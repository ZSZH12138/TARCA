# Stage1B 运行调度与真实监控修复设计

> 状态：已获用户批准，等待设计文件复核
> 日期：2026-08-27
> 实施分支：`codex/stage1b-runtime-supervision-fix`
> 基线分支：`codex/stage1b-v2-official-runtime`

## 1. 目标

修复 Stage1B v2 服务器运行时的两个阻断问题：

1. 调度器必须从主机总容量中扣除全部正在运行任务的 CPU、内存和 GPU 分配，不能在后续轮询中重复占用同一资源；
2. 监控前端必须显示正式运行产生的真实进程、CPU、内存、GPU、显存、温度、功率和 ETA，不能依赖模拟快照或用零值掩盖遥测缺失。

本修复不改变世界、数据、seed、模型、epoch、batch size、门槛或任何科学身份；不运行完整 Stage1B、E01 或 E02；不修改 `docs/auth`。

## 2. 用户可观察的完成标准

- 两张 RTX 4090 上默认最多各有一个神经训练或模型冻结任务；同一张 GPU 不会因调度器重复轮询而接收第二个声明需要 20 GiB 的训练任务；
- CPU 任务和 GPU 任务的活动分配总和始终不超过 24 个数据核、200 GiB 主机内存和每张 GPU 的可用显存；
- 当资源不足时任务保持排队，已有任务结束后下一次调度立即补位；
- 正式运行每两秒写入一次真实遥测，前端显示采样时间并能识别遥测缺失或过期；
- 训练任务使用真实 `completed_steps / total_steps` 计算任务 ETA；同类型已完成任务的真实耗时用于估计尚未启动任务；数据不足时明确显示“校准中”；
- 监控 API 仍然只读，前端不能修改任务、参数、状态或资格结果；
- 所有新增行为都有先失败、后通过的回归测试。

## 3. 方案选择

采用“严格资源账本 + 安全最大并行 + 真实遥测反馈”，不采用固定激进超额装箱。

原因：每个神经训练任务已经声明最多需要 20 GiB 显存，而 RTX 4090 只有 24 GB。即使某个时刻实际显存较低，也不能据此保证第二个任务在后续 epoch 不会增长到峰值。安全吃满硬件的含义是两张卡各运行一个独立任务、CPU 在合同范围内贪心补位，并通过 DataLoader 和实际遥测发现供数瓶颈，而不是让多个20 GiB任务争抢同一张卡。

已有 `decide_gpu_packing()` 和 `select_ddp_mode()` 保留为未来经过服务器实测后使用的策略工具，本次不把它们接入正式科学调度，也不改变单任务训练语义。

## 4. 架构

### 4.1 活动资源账本

`ExecutionStateStore` 新增只读活动尝试投影，返回每个 `RUNNING` 尝试的任务身份、PID、资源申请和已经绑定的资源分配。

调度器每次 `tick()` 执行以下流程：

1. 轮询已启动子进程，更新已退出进程；
2. 从状态库读取当前 `RUNNING` 分配；
3. 从主机容量扣除活动 CPU、主机内存和逐卡 GPU 显存申请；
4. 只使用剩余容量为 `READY` 任务生成新分配；
5. 原子 claim 成功后启动 worker；claim 冲突时不消耗资源；
6. 下一轮重新从状态库建立账本，不依赖进程内猜测。

资源规则保持：调度和监控预留1核、系统和 I/O 预留3核、数据任务最多24核、任务内存预算最多200 GiB、存储至少100 GiB。GPU任务按声明显存进行准入，同一块卡只有剩余显存足够时才可分配。

### 4.2 运行监督器

新增聚焦单一职责的运行监督组件，放在 `src/tarca/execution/supervision.py`。它不读取模型分数、truth、Gate 或资格结果，只处理运行状态。

每两秒执行：

- 获取全部正在运行尝试及 PID；
- 使用 `PsutilNvmlTelemetryProbe` 采集主机和两张 GPU 的真实数据；
- 为每个活动尝试写入带 `attempt_id` 的进程样本；
- 为整个 run 写入一条主机/GPU总样本，供资源卡片使用；
- 检查进程是否消失、遥测是否过期、CPU/GPU是否长期供数不足或出现内存压力；
- 将去重后的告警写入状态库。

采样失败不能终止科学任务。失败会写入 `TELEMETRY_UNAVAILABLE` 告警，并保留最后一次有效采样；前端必须把过期数据标为过期，不能把缺失值显示成真实的0%。

### 4.3 ETA

ETA分为两层：

- 运行中训练任务：根据真实开始时间、`completed_steps` 和 `total_steps` 计算剩余时间；完成进度必须大于0才可计算；
- 尚未开始或非训练任务：使用同 phase、同 model 的已完成尝试耗时中位数；没有同类样本时保持 `CALIBRATING`。

运行级 ETA 根据当前可见任务、依赖关系和 CPU/GPU lane 估计剩余关键路径。当前批次尚未获得稳定速率时显示 `CALIBRATING`；获得真实速率后显示 `AVAILABLE`；完成或失败时分别显示 `COMPLETE` 或 `FAILED`。

硬件预检收据的粗略估计只作为初始参考，不冒充运行期真实 ETA。

### 4.4 监控投影与前端

`MonitoringRepository` 继续以 SQLite 只读方式工作，但改为：

- job 实际 CPU/RSS 使用该 job 最新的 `attempt_id` 样本；
- GPU卡片使用 run 级最新 NVML 样本；
- job 和 run ETA 使用真实进度/历史耗时投影；
- `sampled_at_utc` 为空时返回“遥测不可用”语义，而非生成0值样本；
- 采样超过10秒时生成过期状态和告警。

前端保持同源 WebSocket，每两秒刷新。增加最后采样时间、`正在采集 / 数据正常 / 数据过期` 状态和缺失值显示；0%只表示真实采样值为0，缺失显示为破折号。

### 4.5 任务图总数

运行摘要必须显示完整74任务，而不是仅显示当前已经入库的就绪批次。运行创建时把完整图节点数写入不可变 run 元数据；已完成、运行、失败来自状态库，未入库的未来节点计入 pending。该元数据绑定现有 `graph_id`，同一 run 不允许用不同任务总数覆盖。

## 5. 数据流

```text
74任务图
→ 状态库保存run总数
→ 调度器读取READY + 当前RUNNING分配
→ 计算真实剩余资源
→ 启动不超额的worker
→ worker上报训练进度
→ 监督器每2秒采集进程/主机/NVML
→ SQLite保存真实进度、遥测和告警
→ 只读监控API生成任务、资源和ETA投影
→ WebSocket推送到中文前端
```

## 6. 故障处理

- 调度 claim 冲突：跳过该任务，下轮重新读取状态；
- worker 正常退出：状态以 worker 已提交结果为准；
- worker 异常退出：现有失败和 retry policy 处理，本修复不扩大科学重试范围；
- NVML 暂时不可用：科学任务继续，前端显示遥测不可用并写告警；
- 遥测过期：显示最后采样值及过期标记，不显示为实时数据；
- ETA无样本：显示校准中，不猜测固定时间；
- SQLite写冲突：沿用 `busy_timeout` 和短事务，监督器下一个采样周期重试；
- 资源合同无法满足：任务保持排队；单个任务本身超过主机上限时明确失败，不死循环。

## 7. 测试策略

### 7.1 调度回归

- 2张GPU、8个20 GiB任务连续执行多次 `tick()`，在前两个任务结束前总运行数必须始终为2；
- 一个任务结束后只补一个任务到释放的GPU；
- 多轮 `tick()` 的活动CPU和内存总和不超过合同；
- CPU 24核任务运行时第二个同类任务保持排队；
- claim冲突不泄漏账本容量。

### 7.2 遥测与ETA

- 假探针返回非零CPU、内存、GPU利用率和显存，状态库与API必须原样投影；
- 每个job只读取自己的进程RSS，GPU卡读取run级样本；
- 无样本与过期样本分别显示不可用和过期，不能显示伪0；
- 训练进度20/100结合真实耗时得到非空ETA；0/100保持校准；
- 完成、失败、24小时以上ETA状态正确；
- 采样器异常不会终止调度器。

### 7.3 前端

- 真实非零资源、缺失资源和过期资源三种状态组件测试；
- ETA校准、可用、完成和失败状态测试；
- E2E保留UI快照测试，并增加后端状态库到API的集成测试，明确区分模拟UI测试与真实数据链测试。

### 7.4 完整验证

- Python 3.10服务器矩阵（排除仅支持Python 3.11/3.12的Stage0 doctor CLI）全部通过；
- Python 3.11 Stage0 CLI通过；
- branch coverage不低于80%；
- Ruff、mypy strict通过；
- 前端Vitest、coverage、build、Playwright通过；
- Compose校验、容器空状态、非root/只读/双GPU声明检查通过；
- 不执行完整Stage1B、E01、E02或冻结。

## 8. 变更边界

预计修改：

- `src/tarca/execution/state.py`
- `src/tarca/execution/resources.py`
- `src/tarca/execution/scheduler.py`
- `src/tarca/execution/supervision.py`（新建）
- `src/tarca/monitoring/repository.py`
- `src/tarca/monitoring/schemas.py`
- `src/tarca/stage1b/runner.py`
- `frontend/stage1b-monitor/src/`中的资源与状态展示
- 对应 `tests/execution/`、`tests/monitoring/`、`tests/stage1b/` 和前端测试
- 服务器交接报告和README的运行说明

明确不修改：

- `docs/auth/**`
- Stage1B世界、数据、seed、模型参数、训练预算、资格门槛
- E01/E02实现和状态
- checkpoint科学内容与冻结语义

## 9. 回退

本修复在 `codex/stage1b-runtime-supervision-fix` 独立分支完成。原始实现保留在 `codex/stage1b-v2-official-runtime` 的 `f2104f7`，出现问题可直接切回，不需要删除任何科学制品。
