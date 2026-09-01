# TARCA Stage 2 完成情况与任务交接快照

> 快照日期：2026-09-01
> 阶段状态：`COMPLETED / FROZEN`
> 固定运行：`run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb`
> 固定图：`stage2-graph-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb`
> 最新任务状态：`37/37 COMPLETED`
> 下一实验：E02，当前为 `NOT_RUN_E02_FORMAL`
> 服务器运行事实：`docs/research/stage2_server_run_report_v1.md`

## 1. 功能层结论

Stage 2 已经完成。它的功能不是解释神经网络内部机制，也不是判断 TARCA 的最终方法是否成立，
而是建立后续机制实验必须使用的“冻结预测器套件”：同一份数据切分、同一套概率输出接口、
若干统计/线性基线、PatchTST、iTransformer、固定初始化、验证分数和内容寻址 checkpoint。

本次结果说明，项目已经能够：

1. 在 Stage1B 冻结合成世界上产生开发期训练与验证窗口；
2. 训练并验证 Last Value、Seasonal Naive、VAR、DLinear、PatchTST 和 iTransformer；
3. 用 h1–6 验证 CRPS 固定 strongest-linear 与 primary-iTransformer；
4. 把模型、数据、来源、运行和上游身份绑定成不可变 Stage 2 manifest；
5. 在 E02 formal 数据保持关闭的情况下发布可重算的 Stage 2 freeze receipt。

因此项目当前已经从“真值世界和测量尺准备完成”推进到“预测器套件准备完成”。下一步可以准备
E02，用独立 formal 轨迹检查这些预测器是否满足预注册的 NLL、CRPS、校准和 guardrail 要求。

必须同时保留以下限制：Stage 2 的 `FROZEN` 只表示预测器套件身份完整、固定图完成且可被下游
消费；它不等于 E02 `PASS`，也不构成进入 Stage 3/4 的授权。

## 2. Stage1B、E01、Stage 2 与 E02 的关系

```text
Stage1B FROZEN_V2
  提供：可预测、可复现的 Lorenz-96 双尺度合成世界与反事实 oracle
  ↓
E01 v2/PASS
  提供：已验证的 SCM / paired-effect 测量尺
  ↓
Stage 2 FROZEN（本快照）
  提供：固定概率预测器、checkpoint、验证选择与统一 ForecastDistribution
  ↓
E02 NOT_RUN_E02_FORMAL
  将提供：独立 formal 预测性能、校准、bootstrap、guardrail 和 PASS/FAIL
```

- Stage1B 回答“实验世界是否有可学习信号、真值是否可复现”；
- E01 回答“后续机制效应的测量尺是否可信”；
- Stage 2 回答“哪些预测器和 checkpoint 被固定为后续研究对象”；
- E02 才回答“冻结预测器在未见 formal 轨迹上的概率预测是否合格”。

四者不能互相替代。尤其不得用 Stage 2 的开发/验证 CRPS 代替 E02 formal 结论，也不得把
Stage 2 的训练授权继承为 E02 formal 数据访问授权。

## 3. 固定运行与冻结身份

| 项目 | 固定值 |
| --- | --- |
| protocol | `TARCA-E2E-STAGE-PROTOCOL-2.0` |
| experiment | `stage2_probabilistic_forecasting_v1` |
| world | `lorenz96_twoscale_v2` |
| run | `run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb` |
| Stage 2 scientific config | `8a0509edfd1487dc36188e8d12ca088d52f0287804f4808215ff0f7c279c069f` |
| Stage 2 scientific identity | `c2df021d248c2ffcdcf6133179f4b88c86ea88ae4e3f72630f302b88402e0e32` |
| Stage1B manifest | `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25` |
| E01 receipt identity | `16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14` |
| source capsule archive | `24ed91eea0789554b4b9417d1fdae367084aecac978a709686f6a0e3302e47dc` |
| source receipt identity | `7a1cb3bfca3cd74a2839dda9b0b175fa4190c5468d7e82ccbbbdd45b491ffc71` |
| data manifest | `90b8ea4b94d08263030f9ddbbc5da0c776220e4003922db09c37c41b2b1f9166` |
| normalizer | `42d3e883ef665e4dbbbc3077500a7fec76a4410e7b54d46ec53528b315190acd` |
| runtime identity | `b6da571ceeca5c724e3fca13737e3b1ce5126b5de39a2da33621328188a05bba` |
| freeze manifest identity | `ff50bb15819dea13bd0f31cdb3fc331f02b2ed528509022b0d1aa676d3d8e5d2` |
| freeze receipt identity | `37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166` |
| receipt status | `FROZEN` |
| formal access events | `0` |

机器可核验的小型发布文件为：

```text
artifacts/stage2/frozen/v1/stage2_freeze_receipt.json
artifacts/stage2/frozen/v1/stage2_manifest.json
```

freeze receipt 文件外层 SHA-256 为
`5ec77ab844ef0bc793bf8543db57f01856ab603718a21fd1a19c42bf0947d8e5`；manifest 文件外层
SHA-256 为 `6d9ef496956a714e956c57800f0c1cf479a042624f757f54f4882a99f8d132d4`。外层文件哈希与
JSON 内部 scientific/receipt/manifest 身份作用不同，后续任务不得混用。

## 4. 实验任务、模型和选择结果

固定图由 37 个节点组成，最终数据库中每个阶段的最新 attempt 均为 `COMPLETED`：

| 阶段 | 任务数 | 功能 |
| --- | ---: | --- |
| `UPSTREAM_VERIFY` | 2 | 校验 Stage1B manifest 和 E01 receipt |
| `SOURCE_VERIFY` | 4 | 校验四个固定官方来源及授权边界 |
| `DEV_DATA` | 1 | 生成三组开发数据并只在训练期拟合 normalizer |
| `BASELINE_FIT` | 4 | Last Value、Seasonal Naive、VAR、DLinear |
| `NEURAL_TRAIN` | 6 | PatchTST 与 iTransformer，各三个初始化 |
| `CHECKPOINT_VALIDATE` | 6 | 校验完整 checkpoint、identity 和固定批预测 |
| `VALIDATION_PREDICT` | 10 | 四个基线加六个神经初始化的 h1–6 概率预测 |
| `MODEL_SELECT` | 2 | 固定 strongest-linear 和 primary-iTransformer |
| `FREEZE_CANDIDATE` | 1 | 汇总模型、数据、运行和上游身份 |
| `STAGE2_RECEIPT` | 1 | 原子发布最终冻结 receipt |

模型范围和固定配置入口为 `configs/stage2/stage2_v1.yaml`：

- Last Value 和 Seasonal Naive 提供最低成本参照；
- VAR 在 lag 1/2/4/8/16 与 ridge 候选中拟合；
- DLinear 使用官方实现、100 epoch 上限和 patience 20；
- PatchTST 使用 3 层、`d_model=128`、16 个 head、patch length 16、stride 8；
- iTransformer 使用 3 层、`d_model=512`、8 个 head；
- PatchTST 与 iTransformer 的初始化种子均为 `1797287582`、`883082243`、`1933050005`。

固定选择结果：

| 选择 | 结果 | h1–6 validation CRPS |
| --- | --- | ---: |
| strongest linear（VAR 与 DLinear 中选择） | `VAR` | `0.3996080756187439` |
| primary iTransformer initialization | seed `1797287582` | `0.2084132879972458` |

数值越低表示验证 CRPS 越好。这里的分数只参与 Stage 2 固定选择；它们来自开发/验证边界，
不能替代 E02 formal 指标，也不能单独证明“神经预测器正式合格”。

## 5. 设备不一致事故与受控恢复

### 5.1 原本应该发生什么

每个神经训练任务应当在指定 GPU 完成训练，原子写入带 `COMPLETE` 标记和内容哈希的 checkpoint，
随后在相同模型设备上执行固定验证批预测，最后把 attempt 标记为 `COMPLETED`。

### 5.2 实际发生了什么

六个神经任务都完成了训练并写入完整 checkpoint，但训练尾部的固定验证输入仍在 CPU，模型已经
位于 CUDA。PyTorch 因输入和权重设备不一致抛出错误，使六个 attempt-1 记录为
`FAILED / WORKER_ERROR`。失败范围覆盖：

- PatchTST：三个固定初始化种子；
- iTransformer：三个固定初始化种子。

这不是六个模型没有完成训练，也不是 checkpoint 损坏；错误发生在完整 checkpoint 形成之后的
验证预测边界。旧失败记录必须保留，不能改写成成功或用新 `launch` 隐藏。

### 5.3 为什么会发生

- 原训练函数没有把固定验证 batch 显式移动到模型所在的唯一设备；
- 本地 CPU 测试和假资源测试无法触发真实 CUDA/CPU 组合；
- 原服务器 preflight 只验证训练环境，没有在两张 GPU 上加载真实 `COMPLETE` checkpoint 并走
  零训练步预测路径。

### 5.4 如何修复和补救

1. 新增模型单设备检查和 `forecast_fixed_batch_on_model_device()`，显式把固定批移动到模型设备；
2. 用 `DeviceContractError` 区分设备契约错误，不用广泛 fallback 隐藏问题；
3. 冻结 `configs/stage2/stage2_device_mismatch_recovery_v1.json`，精确绑定同一 run、六个任务、
   六个源 attempt、六个 checkpoint 和机器 profile；
4. 导入固定恢复归档，保留 attempt-1，再为每个任务追加唯一 attempt-2；
5. attempt-2 从 `COMPLETE` checkpoint 进入零训练步路径，要求 checkpoint 哈希不变且不重写；
6. 恢复 preflight 在 GPU 0/GPU 1 上并发加载真实 checkpoint 并执行预测；
7. 服务器已经直接处于目标 PyTorch 容器时，使用
   `deploy/stage2/recovery_bootstrap_direct.sh`，不在容器内嵌套安装 Docker；
8. 同一 run 的六个 attempt-2 最终全部 `COMPLETED`，下游 15 个剩余节点随之完成并冻结。

恢复账本中六条事件统一绑定：

| 项目 | 固定值 |
| --- | --- |
| recovery ID | `stage2-device-mismatch-recovery-v1` |
| reason | `DEVICE_MISMATCH_V1` |
| recovery spec SHA-256 | `6c18b133a9a9861b8d2e804e1d216c48361e595120313aa28152a2a3b96eeb1e` |
| 恢复时服务器代码 bundle | `c0c0b8da1804e982a26234181d5daa83db0c76cf22bef409f788d78fa2c89dc4` |
| 容器直跑入口 SHA-256 | `7298cbc50740e25c8c1f34a22a771e1717304f224022ed250949c4c98533b89f` |

当前本地重新构建的完整服务器 bundle 已包含上述修复，其 SHA-256 为
`1d05dd8a98178ef111990131b682552e5b9cd51e1b23f79397bc4a4fec99deee`。

## 6. 双 GPU 调度与只读前端监督

冻结目标 profile 为：

| 资源 | 运行合同 |
| --- | --- |
| 镜像 | `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` |
| GPU | `2 × RTX 4090`，每张实际可见显存不少于 23 GiB |
| 单 GPU 任务 ceiling | 20 GiB VRAM、4 CPU threads、32 GiB host RAM |
| CPU | 28 核；24 核工作预算、1 核调度/监控、3 核系统/I/O |
| RAM | 224 GiB；调度 ceiling 200 GiB |
| 存储 | 最低 200 GiB free，推荐 300 GiB NVMe |
| reset | 24 小时清空，保留 1 小时安全边界 |

调度器把两个可运行神经任务分别投放到 GPU 0 和 GPU 1；依赖满足的 CPU-only 任务使用剩余预算
backfill。并行目标是缩短固定科学图的关键路径，不通过复制任务、减少 seed、降低 epoch 或伪造
利用率来“吃满”硬件。科学任务 identity 与 worker placement 分离，因此 GPU 分配不改变结果身份。

监控前端只绑定 `127.0.0.1:8765`，只提供 GET/HEAD：

- 展示 GPU 0/GPU 1 利用率、显存、功率、温度和计算进程；
- 展示 CPU 忙核、任务亲和性、真实进度和 ETA 来源；
- 缺失遥测显示“不可用”，不伪装成 0；
- 只显示每个任务的最新 attempt，但旧失败仍保存在数据库；
- 没有停止、删除、重试或改写科学任务的按钮。

实验进程、监控进程和 SSH 隧道彼此独立。用户关闭浏览器或隧道不应终止实验；服务器实验通过
后台进程/容器和持久 runtime 目录继续运行。

## 7. 最终归档与本地保管边界

下列大文件是本地审计/复现证据，不进入 GitHub：

| 内容 | 本地相对路径 | SHA-256 |
| --- | --- | --- |
| Stage 2 最终完整服务器归档 | `artifacts/stage2/server-results/stage2-v1-complete-20260901T011423Z/tarca-stage2-v1-complete-20260901T011423Z.tar.gz` | `7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a` |
| 事故恢复输入归档 | `artifacts/stage2/server-archives/tarca-stage2-recovery-20260831T102151Z.tar.gz` | `79c6cb2c0f8fd8a1801d378fb779212b66f3774d8372df7b360b1721b3f9b126` |
| 当前服务器 bundle | `artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz` | `1d05dd8a98178ef111990131b682552e5b9cd51e1b23f79397bc4a4fec99deee` |
| 官方来源 capsule | `artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz` | `24ed91eea0789554b4b9417d1fdae367084aecac978a709686f6a0e3302e47dc` |

每个归档/bundle 必须与同目录 `.sha256` 或 receipt 一起保管。最终服务器归档包含执行数据库、
checkpoint、内容寻址 store、失败与恢复历史、冻结 receipt/manifest 和运行日志。删除安全解压副本
不删除原归档；需要审计时必须先核验外层 SHA-256，再解压到新的受控目录。

本地清理允许删除缓存、测试临时导入、重复 verify bundle 和最终归档的可再生 `extracted/`；不得
删除本节四个固定保管对象及其 sidecar/receipt。

## 8. GitHub 发布边界

GitHub 应发布：

- Stage 2/E02 源代码、配置、测试、部署入口和只读监控前端源码；
- 恢复规格、事故说明、服务器运行报告和本权威快照；
- `artifacts/stage2/frozen/v1/` 中两个小型 JSON。

GitHub 不发布：

- checkpoint、模型权重、SQLite 数据库和 content-addressed store；
- 服务器归档、恢复归档、服务器 bundle、来源 capsule 和 wheelhouse；
- 原始/生成数据、第三方仓库缓存、日志、密钥、连接地址或代理值；
- `node_modules`、构建缓存、覆盖率、Playwright 输出和解压副本。

`.gitignore` 对 `artifacts/stage2/**` 采用默认本地保管，只显式放行 freeze receipt 和 manifest。
后续任务不得使用 `git add -f` 绕过该边界。

## 9. 当前代码、配置和操作入口

| 功能 | 入口 |
| --- | --- |
| Stage 2 scientific config | `configs/stage2/stage2_v1.yaml` |
| E02 scientific config | `configs/e02/e02_v1.yaml` |
| 固定恢复规格 | `configs/stage2/stage2_device_mismatch_recovery_v1.json` |
| Stage 2 CLI | `scripts/run_stage2_v1.py` |
| E02 CLI | `scripts/run_e02_v1.py` |
| Stage 2 runtime | `src/tarca/stage2/runtime.py` |
| Stage 2 任务图/执行器 | `src/tarca/stage2/tasks.py`、`src/tarca/stage2/jobs.py` |
| 神经训练/设备边界 | `src/tarca/stage2/training.py` |
| 恢复授权/导入/探针 | `src/tarca/stage2/recovery.py`、`recovery_archive.py`、`recovery_probe.py` |
| 共享执行面 | `src/tarca/execution/` |
| 只读监控 API | `src/tarca/monitoring/` |
| 前端 | `frontend/stage1b-monitor/` |
| Docker 宿主入口 | `deploy/stage2/recovery_bootstrap.sh` |
| 目标容器直跑入口 | `deploy/stage2/recovery_bootstrap_direct.sh` |
| 服务器包构建 | `scripts/prepare_stage2_v1_server_bundle.py` |
| 安全接入规则 | `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md` |

服务器入口必须先判断 SSH 落点是 Docker 宿主机还是目标 PyTorch 容器，选择对应 bootstrap；不得在
目标容器内为了复用 Compose 而嵌套安装 Docker。

## 10. 独立验证证据

本快照采用以下完成判据，而不是“GPU 曾经很忙”或“前端曾经显示绿色”：

1. 最终归档外层 SHA-256 与 sidecar 一致；
2. 安全路径检查证明归档不能越界解压；
3. 执行数据库中 37 个固定节点的最新 attempt 全为 `COMPLETED`；
4. 六个 attempt-1 `WORKER_ERROR` 和六个 attempt-2 `COMPLETED` 同时存在；
5. 六条 recovery event 精确绑定源 attempt、新 attempt、checkpoint、spec 和 bundle；
6. 零训练步恢复保持 checkpoint 内容哈希不变；
7. manifest、freeze receipt 和所有引用 artifact 的内容哈希可重算；
8. receipt 为 `FROZEN` 且 `formal_access_event_count = 0`；
9. 最终归档中不存在 E02 formal grant、formal trajectory 或 E02 final receipt。

后续代码发布前仍应在本地运行：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -m pytest -q
python -m ruff check . --no-cache
python -m mypy
```

前端还需执行 Vitest、coverage、production build 和 Playwright；四个 Stage 2 Bash 入口需执行
`bash -n`。这些工程检查不改变 Stage 2 冻结身份。

## 11. 已知边界

- Stage 2 使用开发/验证数据选择模型；E02 formal 数据仍密封；
- strongest linear `VAR` 只是在 VAR/DLinear 固定候选中的选择，不表示它优于所有方法；
- primary iTransformer seed 只固定正式比较入口，其他两个初始化仍保留为 E02 配对比较对象；
- Stage 2 未评估内部机制位置、交换干预、OT 定位、DAS 或跨状态 zero-refit；
- Stage 2 `FROZEN` 不表示整个 TARCA 项目通过；
- E02 未运行，因此目前没有 E02 NLL/CRPS、bootstrap、guardrail 或 PASS/FAIL；
- 不得因为 E01 和 Stage 2 均完成就跳过 E02 直接进入 Stage 3/4。

## 12. 下一任务的正确起点

下一任务是 E02，但分成两个权限层：

### 12.1 不打开 formal 数据的准备层

1. 从 Git 中核验两个 Stage 2 frozen JSON；
2. 从本地正式归档或受控传输恢复它们引用的完整 Stage 2 suite；
3. 重新执行 E02 `prepare`、`dry-run` 和 `preflight`；
4. 确认预期 formal trajectory 数为 120，但已执行 formal task 数仍为 0；
5. 检查服务器 GPU、CPU、RAM、磁盘和剩余租期，重新计算 E02 ETA；
6. 启动只读监控，但不得创建 formal grant。

### 12.2 需要用户再次书面授权的 formal 层

只有用户明确授权 E02 后，才可使用确认串：

```text
I_ACKNOWLEDGE_E02_V1_FORMAL_RUN
```

E02 `launch` 才能原子创建 sealed access grant 并首次物化 120 条 formal trajectories。中断后只能
按相同 E02 run identity 执行 `resume`，不能新建 run 或重新开放 formal 数据来隐藏失败。

E02 完成判据是 120/120 trajectory、全部预测/评分/bootstrap/guardrail 节点完成并原子发布
E02 final receipt；在此之前不得给出 E02 合格结论。

## 13. 后续任务交接检查表

- [ ] 先读本快照、`stage2_server_run_report_v1.md` 和服务器 runbook；
- [ ] 核验两个 Git 冻结 JSON 的外层和内部哈希；
- [ ] 核验本地最终归档及 sidecar，禁止直接信任解压目录；
- [ ] 确认 E02 formal grant、formal trajectories 和 final receipt 仍不存在；
- [ ] 不修改 `configs/stage2/stage2_v1.yaml` 或冻结模型选择；
- [ ] E02 准备阶段保持 formal task count 为 0；
- [ ] 在点火前重新做硬件和剩余租期 gate；
- [ ] 只有获得新的 E02 书面授权后才使用 E02 确认串；
- [ ] E02 运行由用户和只读前端监督，Codex 不承担持续轮询；
- [ ] 异常时先回收数据库、已发布 artifact 和 recovery capsule，再决定是否同 run 恢复；
- [ ] E02 final receipt 出现并独立校验前，不进入 Stage 3/4。
