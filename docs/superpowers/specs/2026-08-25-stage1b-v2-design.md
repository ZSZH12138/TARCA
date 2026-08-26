# Stage1B v2 官方世界、双 GPU 资格运行与只读监控设计

> 状态：`APPROVED_FOR_IMPLEMENTATION`
> 初版日期：2026-08-25
> 本次修订：2026-08-26
> 协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 科学版本：继续使用 Stage1B v2；不创建 v3
> 当前边界：只完成设计；不执行完整资格、E01 或 E02，也不冻结科学结果

## 1. 目的

本设计在现有 Stage1B v2 上完成四项修正：

1. 世界数据、生成器和神经模型尽量直接使用论文官方仓库的固定版本，不再只做近似复刻；
2. 把世界真值从预测器中彻底分离，确保真值来自世界方程、生成器状态和可重放噪声；
3. 增加适配双 RTX 4090、28 核 CPU、224GB 内存服务器的任务调度与故障恢复；
4. 增加只读可视化前端，显示任务存活、期望/实际资源、进度、告警和预计剩余时间。

这不是为了制造“神经网络必胜”的数据。候选世界必须先满足 TARCA 后续的结构、概念、
干预、传播、regime、概率效应和内部机制定位需求，再按已经登记的 v2 资格标准比较神经
模型与 VAR。

## 2. 权威边界与不变量

本设计服从 `docs/auth`，尤其是：

- Stage1B 必须产生 `DataManifest`、`SCMTruthManifest`、标准物理分区和高层 oracle bridge；
- 事实与反事实使用相同未来外生噪声，无干预时必须逐位相同；
- truth 不得进入普通预测 `WindowBatch`；
- 预测器只能消费 Stage1A 标准 bridge，不能读取生成器私有状态；
- 调度器只能改变并行度、worker placement、GPU/CPU 分配、内存准入和重试安排；
- 调度器不得根据中途 NLL、CRPS、MAE、seed 排名或 sealed truth 改变模型、seed 或任务；
- 监控界面不得展示会影响人工择优的未完成科学排名；
- 服务器 backend 不得改变 scientific identity；
- 当前任务不执行 E01、E02 或 Stage3 以后实验。

现有 v2 资格规则保持不变：至少 40 个独立比较单元、CRPS 胜率至少 65%、总体 CRPS
skill 大于 0、seen/unseen 均为多数胜出，并通过 bootstrap、NLL、MAE、校准、
worst-regime 和内部位置可操作性护栏。本次只修复世界、官方模型、运行和监控实现，不根据
未来结果修改阈值。

## 3. 版本、授权与回退

### 3.1 保持 v2

- 活动配置继续命名为 `worlds_v2.yaml` 和 `qualification_v2.yaml`；
- schema 和 qualification identity 继续属于 v2；
- 不创建 v3；
- v1 只保留 `docs/research/stage1b_world_qualification_report_v1.md`，其余 v1 活动实现由
  v2 覆盖。

### 3.2 正常冻结、授权覆盖

v2 尚未通过资格，因此当前可以在用户授权下原位修正。正式资格通过并经用户确认后，v2
正常进入只读冻结状态。

以后如用户明确授权修改或覆盖：

- 仍可保持 v2 科学系列，不被迫改名为 v3；
- 修改前记录授权原因和受影响范围；
- 产生新的 `revision_id`、配置哈希和资格收据；
- 旧收据只用于审计，不能被静默改写成新结果；
- 受影响的正式资格必须完整重跑，不能只重跑失败 seed。

普通开发失败、探针失败和未进入正式资格的中间产物不冻结为科学历史版本，可以清理或
覆盖。已经启动的正式盲资格必须保留全部成功与失败比较单元，防止选择性重跑。

### 3.3 Git 回退面

- `codex/stage1b-v2-pre-official-runtime` 固定在本次官方运行修订之前；
- `codex/stage1b-v2-official-runtime` 承载本设计和后续实现；
- 分支名称不改变 Stage1B 的科学版本。

## 4. 总体架构

```text
固定的 v2 科学配置与来源清单
          |
          v
官方源码/数据物化与哈希校验
          |
          +--> 官方复现通道 ----------> 复现收据
          |
          +--> TARCA oracle 资格通道
                    |
                    v
             TaskManifest / ExecutionPlan
                    |
        +-----------+------------+
        |                        |
   CPU 数据/VAR worker      GPU 神经训练 worker
        |                        |
        +-----------+------------+
                    |
       heartbeat / progress / resources
                    |
                 SQLite
                    |
          FastAPI + WebSocket
                    |
          React 只读监控前端
```

Science Plane 决定世界、数据、模型、seed、指标和门槛。Execution Plane 只负责把已经固定
的任务放到 CPU/GPU 上运行。Monitoring Plane 只读取运行状态，不参与科学选择。

## 5. 官方来源与数据策略

### 5.1 固定来源

继续使用当前 v2 已登记的官方来源与精确 commit：

| 来源 | 用途 | 使用方式 |
|---|---|---|
| Neural-GC | 稳定 VAR、单尺度 Lorenz-96 | 官方生成器、配置和数据语义 |
| GVAR | Lorenz-96、捕食者—猎物 | 官方生成器与官方存储数据 |
| JMLR 双尺度 L96 仓库 | 双尺度 Lorenz-96 | 官方脚本、参数和数据生成路径 |
| Interfere | CML 与干预基准 | 官方生成器和干预接口 |
| PatchTST | PatchTST 主干 | 固定 commit 的官方模型代码 |
| iTransformer/Time-Series-Library | iTransformer 主干 | 固定 commit 的官方模型代码 |

用户已授权：许可证缺失或不明确不再阻断本次直接使用官方数据与代码。实现仍须记录仓库
URL、commit、文件哈希、获取日期、许可证状态和本次授权；这份记录用于来源审计，不代表
TARCA 对许可证作出新的法律判断。

### 5.2 不修改官方原件

官方源码和原始数据进入只读 source cache：

- 以来源 ID 和 commit 定位；
- 下载或物化后先校验 commit/文件哈希；
- 原始文件保持不变；
- TARCA 只生成格式转换、split、normalization、truth 和 provenance sidecar；
- 正式运行期间不从网络按 batch 读取数据。

### 5.3 两条数据通道

官方数据不能一概当作 TARCA 的“已知答案世界”。因此数据分为两条隔离通道：

1. `OFFICIAL_REPRODUCTION`：按官方数据、官方切分、官方模型和官方超参数复现上游行为，
   用于确认接入没有改坏；该通道不自动产生 TARCA oracle truth。
2. `TARCA_ORACLE_QUALIFICATION`：运行官方生成器或官方方程路径，同时保存初态、状态、
   regime、外生噪声、图、lag 和干预记录，形成 TARCA 需要的 paired oracle。

只有具备完整真值 sidecar 的轨迹才能进入第二条通道。只提供观测值、但缺少初态、噪声或
方程辅助信息的官方静态数据只能用于官方复现或辅助验证，不能冒充反事实真值。

资格内部命名空间与 Stage1A 物理分区做固定映射：`QUAL_TRAIN -> TRAIN`、
`QUAL_TUNE -> VALIDATION`、`QUAL_SEEN -> TEST_SEEN_REGIME`、
`QUAL_UNSEEN -> TEST_UNSEEN_REGIME`。映射只改变契约名称，不重切分、不拼接、不打乱轨迹；
normalization 只在 `QUAL_TRAIN` 拟合。资格数据、E01/E02 数据和 reserved formal seeds 继续
使用不同身份与目录，不能互相读取。

### 5.4 候选世界的项目用途

| 世界 | 主要功能 | 后续价值 | 资格角色 |
|---|---|---|---|
| 稳定稀疏 VAR | 检查基线公平性与泄漏 | 结构负对照、错误机制对照 | 控制；允许 VAR 胜出 |
| Lorenz-96 F=10 | 非线性方向传播与状态干预 | layer/time/lag/variable、regime、概率效应 | 主世界 |
| Lorenz-96 F=40 | 强混沌压力测试 | 极端 regime 和失败边界 | 压力世界 |
| 双尺度 Lorenz-96 | 观测慢变量和潜在快变量 | 隐藏机制、多尺度传播、子空间定位 | 主世界 |
| GVAR 捕食者—猎物 | 有符号二部图和扩散噪声 | signed graph、生态/金融传播、干预 oracle | 辅助世界 |
| 修正 CML | 局部耦合与图传播 | wrong-source、wrong-lag、局部传播负对照 | 辅助世界 |

主世界必须提供两个最小概念语义：

- `trend/persistence`：由官方世界的强迫、增长或持续性参数定义；
- `local_scale/volatility`：由官方世界的动态噪声、观测噪声或快尺度强度定义。

概念值从当时状态与生成参数计算，不能从未来标签反推。只改变 trend 时，scale 的生成
状态和未来噪声保持不变；只改变 scale 时，trend 的生成状态保持不变。世界的原生非线性
可能让最终均值和方差同时响应，但被干预的生成因子必须保持单一、可审计。

为此，`TARCA_ORACLE_QUALIFICATION` 可以在官方生成器外增加最小的 oracle adapter：它只把
预注册的 trend/scale schedule 映射到官方已有的强迫、增长、噪声或快尺度参数，不替换官方
动力学，也不从模型输出构造真值。adapter 关闭时必须与官方生成器逐步一致。每个主世界
必须预先提供至少一组 trend source/base pair 和一组 scale source/base pair；pair 共享初态、
其他生成参数和未来噪声。参数值或范围必须来自论文、官方配置或官方数据统计证据，并写入
来源清单，不能为追求神经胜率临时试填。

若官方随机生成器把噪声隐藏在内部、无法导出或注入同一未来噪声，则该世界在增加并通过
显式噪声 adapter 的一步等价测试前，只能进入官方复现通道，不能进入 oracle 资格通道。

## 6. 官方模型与 TARCA 适配

### 6.1 官方主干优先

PatchTST 和 iTransformer 不再由 TARCA 写“相似结构”代替。每个训练 worker 直接加载固定
commit 的官方主干、官方预处理、官方架构默认和已登记超参数。

两条通道的超参数职责分开：官方复现使用与其官方数据配套的上游命令和配置；TARCA
资格使用冻结的 `qualification_v2.yaml` 参数。官方复现结果不参与 v2 资格门槛，资格参数
也不能在看到 blind 结果后改成官方复现中表现更好的组合。

官方复现任务先验证：

- 输入窗口与官方脚本一致；
- 归一化、patch/token 处理和输出形状一致；
- 关闭 TARCA 扩展时，官方入口与 TARCA wrapper 的确定性输出在容差内一致；
- 使用官方推荐训练方式可以完成最小 forward/backward/checkpoint/reload。

### 6.2 薄适配层

TARCA wrapper 只负责：

- `WindowBatch` 与官方输入之间的显式转换；
- 将官方预测头接到 `ForecastDistribution`；
- 提供有限且为正的概率尺度；
- 固定模型 identity、配置哈希和权重哈希；
- 暴露冻结后的内部位置目录；
- 支持 capture、swap 和不改写权重的 intervention；
- 把所有适配差异写入收据。

PatchTST 负责 layer × time/patch 位置；由于其通道独立设计，不单独承担完整跨变量机制
主张。iTransformer 负责 layer × variable × subspace 位置和跨变量传播。至少一个通过
资格的神经模型必须同时通过相应位置的捕获、交换、identity no-op 和权重不变测试，才能
认为该世界对 Stage3–8 有效。

## 7. 真值与 paired oracle

真值来自世界生成器，不来自 PatchTST、iTransformer、VAR 或任何训练模型。

每条资格轨迹必须持久化或可验证重建：

- 官方来源和生成器 commit；
- `SyntheticConfig` 与配置哈希；
- root seed、轨迹 seed、初态摘要；
- regime sequence；
- 外生噪声与 shock 的内容哈希；
- true graph、signed edge、mechanistic lag 和 latent dimension；
- trend/scale 概念状态；
- boundary clipping、数值积分和观测规则；
- factual/counterfactual 的共享噪声证明。

硬测试包括：

1. 同 seed 完全重放；
2. identity intervention 逐位相同；
3. paired factual/counterfactual 共享未来噪声；
4. 方程一步结果与官方实现一致；
5. lag 来自方程依赖，不把经验峰值错误写成机制最短路径；
6. 状态依赖符号在每个时间点可追踪；
7. latent truth 只能由 oracle 路径解析；
8. 普通 `WindowBatch` 不包含 truth；
9. 轨迹有限、非塌缩、非伪二周期，声明的 clipping 必须计数。

## 8. 容器与运行环境

### 8.1 固定运行镜像

正式服务器 runtime 以以下镜像为基础：

```text
pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04
```

运行时必须验证：

- Python 3.10；
- PyTorch 2.2.2；
- CUDA 12.1 runtime 可用；
- 两张 RTX 4090 均可见且每张约 24GB 显存；
- 两张卡分别完成最小 forward、backward、AMP、checkpoint 和 reload；
- 28 个 CPU 执行单元、224GB 主存和本地高速存储满足准入要求。

### 8.2 Python 3.10 兼容

现有代码中的 Python 3.11 专属用法需要兼容处理：

- `StrEnum` 使用兼容实现；
- `Self` 来自 `typing_extensions`；
- `tomllib` 在 3.10 下使用 `tomli` fallback；
- lint、mypy 和测试覆盖 Python 3.10，开发环境可继续复核 3.11。

服务器依赖不得用 CPU-only PyTorch 覆盖镜像自带 CUDA PyTorch。NumPy 和官方仓库依赖要
使用与 PyTorch 2.2.2 兼容的固定版本；运行收据记录最终 `pip freeze`、镜像 digest、驱动、
CUDA、GPU UUID 摘要和 Git commit。

### 8.3 前端构建

React/TypeScript 前端在镜像构建阶段编译为静态文件。最终 PyTorch runtime 不运行 Node，
FastAPI 只提供已构建的静态资源和只读 API。

## 9. 任务模型与依赖图

科学配置先编译为协议要求的 `ExperimentSpec -> TaskManifest`。内部不可变 job 记录必须能
一一映射到 `TaskSpec/PlannedTask`，至少包含：

- world、model、seed、partition 和输入 artifact hash；
- 依赖任务；
- 期望 CPU、RAM、GPU、VRAM；
- epochs、batches、checkpoint、超时和重试规则；
- precision policy 和 scientific identity；
- 输出 ArtifactRef 类型。

Task 输入引用固定的配置或数据 ArtifactRef，不允许使用“寻找最新 checkpoint/最新数据”这类
会随目录状态变化的路径。

状态机为：

```text
QUEUED -> STARTING -> RUNNING -> SUCCEEDED
                     |  |  |
                     |  |  +-> STALLED
                     |  +----> RETRYING
                     +-------> FAILED
```

`SUCCEEDED/COMPLETED` 必须绑定通过校验的 ArtifactRef。调度器数据库只是运行账本，不能
替代正式科学 artifact。

资格任务按依赖拆分为：来源物化、官方复现、世界健康检查、完整轨迹生成、数据校验、VAR
调优/评分、神经训练、冻结模型检查、评分/bootstrap、资格聚合和收据生成。当前配置预期
形成 12 个相互独立的主 GPU 训练任务，实际任务数必须由冻结 manifest 计算，不能在代码
中硬编码。

## 10. 双 RTX 4090 调度

两张 4090 无 NVLink，因此默认使用任务级并行：每张卡运行不同的 world/model/seed 任务。

### 10.1 动态装箱

每张卡从 1 个任务开始，观察至少 3 分钟的稳定窗口：

- GPU 利用率低于 70% 且单任务显存低于 8GB时，允许尝试每卡 2 个任务；
- 仍低于 80% 且总显存低于 18GB时，允许尝试每卡 3 个任务；
- 显存超过 20GB、发生 OOM、吞吐下降、热限制或数据等待恶化时，降低装箱数；
- 装箱调整只改变并发，不改变 batch size、模型、seed、epoch、horizon 或任务数量。

OOM 可以在降低同卡并发后自动重试一次；不能通过缩小科学配置来掩盖 OOM。

### 10.2 DDP 边界

只有单个任务明显过大、并且同一正式配置的预探针显示双卡 DDP 至少缩短约 30% 墙钟时间
时，才允许为该任务使用 DDP。否则两卡各跑独立任务，以避免通信开销和 4090 无 NVLink
造成的低效。

### 10.3 precision policy

AMP 不能在正式运行中随意开关。服务器预探针只在预声明候选精度中验证数值一致性与
吞吐；选定结果先写入 v2 precision record 并随 TaskManifest 冻结，然后整个资格运行保持
不变。调度器不能根据中途科学指标修改精度。

## 11. CPU、内存和 I/O 调度

服务器期望资源为 28 核 CPU、224GB RAM：

- 1 核预算给调度与监控；
- 1–4 核保留给系统、文件系统和突发 I/O；
- 数据生成阶段可给 CPU worker 分配约 20–26 核；
- GPU 训练时每个训练进程初始分配 2–4 个 DataLoader worker；
- 剩余 CPU 并行执行下一批数据生成、VAR、评分和 bootstrap；
- 使用 CPU affinity，并限制 OMP/MKL 线程，防止进程乘法导致过量订阅；
- 根据 GPU data-wait 和 CPU contention 调整 worker 数，但不改变科学数据。

内存准入上限约 190–200GB，给系统保留 24–32GB。大数组优先使用 mmap/page cache，训练
使用 pinned memory、prefetch 和 persistent workers；扩大 Docker shared memory，数据与
checkpoint 放在本地 NVMe。未发现足够容量和吞吐的本地存储时，启动探针 fail closed。

目标是有任务可运行时保持有用饱和，而不是用无意义计算制造 100% 数字。GPU 队列非空但
卡长期空闲、DataLoader 长期让 GPU 等待、CPU 过量订阅或内存缓存挤压都必须告警。

## 12. 运行账本、遥测与 ETA

调度器使用 SQLite WAL 保存结构化状态：

- run、job、attempt；
- dependency 与 assigned resource；
- heartbeat 和 progress event；
- resource sample；
- alert 与 terminal error category。

worker 每 2 秒发送 heartbeat/进度，采集器读取进程树、psutil 和 NVIDIA NVML。5–10 秒
落盘一次，长期曲线降采样，避免监控本身成为负载。

### 12.1 期望值与实际值

| 项目 | 期望值 | 实际值 |
|---|---|---|
| 进程 | Job 状态与 worker identity | PID、进程树、存活、heartbeat age |
| CPU 核数 | Job 分配核数和 affinity | 实际 affinity；有效忙核=`进程树 CPU% / 100` |
| CPU 占用 | 任务预算 | 进程树 CPU%、宿主机 CPU% |
| 内存 | 任务和全局预算 | RSS/PSS、宿主机已用/可用内存 |
| GPU | 分配的 GPU | CUDA PID 实际所在 GPU |
| 显存 | 预测/保留显存 | 进程显存和整卡显存 |
| GPU 负载 | 目标利用率 | utilization、power、temperature |
| 进度 | epochs/batches/work units | 已完成量、最近吞吐、data-wait |
| ETA | 探针预测 | rolling ETA、全局 critical-path ETA |

### 12.2 ETA

- 初始样本不足时显示“校准中”；
- 对每个 world/model 使用最近 batch 时间的 EWMA；
- 加上 validation、checkpoint、数据等待和重试开销；
- 全局 ETA 按两条 GPU 队列、CPU 依赖和未完成 critical path 计算；
- 显示预计完成时间、剩余小时和不确定区间；
- 预计超过 24 小时时告警，并停止正式资格启动、等待用户决定；
- 不允许通过减少 seed、epoch、horizon 或比较单元把 ETA 人为压到 24 小时内。

## 13. 只读监控前端

### 13.1 技术结构

- FastAPI 提供版本化只读 REST API、WebSocket 和静态文件；
- React + TypeScript 构建页面；
- ECharts 绘制资源和吞吐曲线；
- SQLite 是单一运行状态来源；
- psutil 与 NVIDIA NVML 负责实际资源采集。

最小 API 提供 run summary、jobs、resources、alerts 和 WebSocket snapshot。所有查询参数都
做 schema 校验，日志读取只允许 scheduler 登记的路径。

### 13.2 页面内容

顶部总览显示：

- `Stage1B v2`、终态、已运行时间；
- 总进度、任务状态计数；
- ETA、预计完成时间和超过 24 小时风险。

资源区显示：

- 期望/可分配/有效忙 CPU 核；
- 期望/实际/系统 RAM；
- GPU0/GPU1 当前任务、显存、利用率、功耗、温度；
- 磁盘读写和 DataLoader wait。

任务表显示：

- world、model、seed、PID、存活、状态、GPU；
- 期望与实际 CPU/RAM/VRAM；
- epoch、batch、吞吐、ETA、heartbeat、retry 和错误类别。

告警包括：

- GPU 有待运行任务但长期空闲；
- worker 死亡或 heartbeat 超时；
- OOM、NaN、hash drift、truth 校验失败；
- CPU 过量订阅、DataLoader starvation、内存高水位；
- 热限制、磁盘过慢、ETA 超过 24 小时。

### 13.3 科学与安全边界

前端只能读，不能暂停、重启、杀进程、改配置或改调度。它不显示 partial CRPS/NLL/MAE、
unseen truth、模型排名、最佳 seed 或任何模型选择建议。

容器内 FastAPI 可监听服务端口，但 Docker 只发布到服务器宿主机 `127.0.0.1`。用户可在
自己的可信 SSH 会话中建立本地端口转发；TARCA 自动化服务器接入仍严格服从
`TARCA_SERVER_ACCESS_RUNBOOK.md`，不自行放宽其中的转发限制。不公开监听公网，不挂载
Docker socket，不读取环境秘密。运行日志经过 allowlist 和脱敏，不能浏览任意文件。

监控预算不超过约 1 个 CPU 核和 1GB RAM；监控进程失败不终止训练，恢复后可从 SQLite
继续读取。

## 14. 故障恢复

- 每个 epoch 至少保存一次原子 checkpoint；
- checkpoint 绑定任务、配置、数据、模型和 precision hash；
- scheduler 重启时先识别已有存活 PID，不能重复启动同一任务；
- 已完成且 artifact 验证通过的任务不重跑；
- 暂时性 worker/IO 错误最多自动重试一次；
- OOM 只在降低并发后重试一次；
- hash mismatch、truth mismatch、NaN、partition leakage 和 scientific identity drift
  直接 fail closed，不自动继续；
- 所有失败保留错误类别、最后 heartbeat 和尝试号，不记录秘密。

## 15. 正式服务器流程

实现完成并获得运行授权后，服务器按以下顺序执行：

1. 校验镜像、Git、官方来源、数据缓存、CPU、RAM、GPU 和本地存储；
2. 两张 GPU 分别执行最小 CUDA/AMP/checkpoint 探针；
3. 运行短数据、VAR 和官方模型校准，选择并冻结并发、DataLoader 和 precision policy；
4. 若 critical-path ETA 超过 24 小时，则停止并报告，不缩减科学工作量；
5. 启动 scheduler、worker、遥测、FastAPI 和只读前端；
6. 完成官方数据/官方模型复现通道；
7. 复现通过后运行 Stage1B v2 资格通道；
8. reserved formal seeds 保持未使用；
9. 所有任务完成后才聚合 VAR/神经结果和 bootstrap；
10. 生成来源、硬件、失败、资源和资格收据；
11. 用户审阅后才能冻结 v2；
12. E01、E02 继续不运行。

## 16. 实现边界

后续实现计划需要覆盖以下独立模块，但共同服从同一个 TaskManifest：

1. Python 3.10/CUDA 服务器环境与启动探针；
2. 官方来源物化、哈希和两条数据通道；
3. 官方世界与 `SCMTruthManifest` bridge；
4. 官方 PatchTST/iTransformer 薄适配；
5. Science-blind scheduler、worker 与恢复；
6. SQLite 遥测、资源采集和 ETA；
7. FastAPI 只读 API 与 React 前端；
8. 资格聚合、收据和冻结流程；
9. 单元、集成、GPU 探针和关键前端 E2E 测试。

不修改 Stage1A 公共 Schema；如确实需要新的跨阶段字段，必须先走 CCP。Stage1B 内部类型
需要通过显式 adapter 映射到现有公共契约，不能创建第二套跨阶段数据语言。

## 17. 验收标准

设计落地后至少满足：

- 官方源码/数据 commit 与哈希可复核，原件不被修改；
- 官方复现与 TARCA oracle 资格数据完全隔离；
- 官方模型关闭 TARCA 扩展时通过输出等价检查；
- 真值全部由 generator/oracle 产生，预测器无法读取；
- identity、shared-noise、equation-step、lag、sign、latent 和 leakage 测试通过；
- 同一科学 manifest 在不同 worker placement 下产生相同 scientific identity；
- 两张 4090 均被有效调度，空闲与过量订阅可被检测；
- 进程、期望/实际 CPU/RAM/VRAM、进度、ETA 和告警可在前端查看；
- 前端无科学排名和运行控制入口；
- scheduler/worker/API 的单元与集成测试、前端关键 E2E 测试通过；
- 全项目测试通过，branch coverage 不低于 80%，Ruff 和 mypy 通过；
- 完整资格、E01 和 E02 在没有对应授权前均未执行；
- 只有正式资格通过且用户确认后才写 `FROZEN v2`。

## 18. 已知风险与关闭方式

| 风险 | 关闭方式 |
|---|---|
| 官方静态数据缺少初态或未来噪声 | 仅用于复现；资格使用可记录完整 truth 的官方生成器路径 |
| 官方仓库相互污染 Python import | 每个官方入口在隔离 worker 进程中加载，来源路径和 commit 固定 |
| Python 3.10 与现有 3.11 代码不兼容 | 兼容层和 3.10 CI/容器测试先行 |
| pip 覆盖 CUDA PyTorch | 独立 server constraint 与启动时 torch/CUDA 精确校验 |
| 双 4090 无 NVLink，DDP 可能更慢 | 默认任务级并行；仅用预探针证实收益后启用 DDP |
| 强行追求 100% 导致吞吐下降 | 以完成时间、队列空闲和 data-wait 为准做有用饱和 |
| 24 小时目标无法满足 | 启动前报告 critical-path ETA，不削减科学工作量 |
| 监控泄露 blind 科学结果 | API schema 和 E2E 测试禁止 partial metrics、truth 和排名 |
| 自动恢复造成选择性重跑 | 重试规则预注册，正式比较单元全部保留并一起聚合 |

## 19. 明确不做

- 不创建 Stage1B v3；
- 不在本设计阶段实现代码；
- 不执行完整 Stage1B 资格；
- 不执行 E01 或 E02；
- 不改变 65% 等既有 v2 资格阈值；
- 不因硬件缩减 seed、epoch、horizon、模型或比较单元；
- 不让监控前端控制任务或参与模型选择；
- 不把真实预测数据误称为具有完整反事实真值的世界；
- 不为了显示高利用率而运行无意义计算。
