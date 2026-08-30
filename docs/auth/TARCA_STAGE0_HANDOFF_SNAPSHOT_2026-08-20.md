# TARCA Stage 0 历史执行快照与 Stage 1A 交接说明

> 文档性质：历史执行证据 / 阶段交接材料（非规范性文件）
> 快照时间：2026-08-20T12:02:08.4508828Z
> 对应 Git 提交：`2f6c21aa66e137c577635632ee6cc02fefbcbd66`
> 对应远端：`https://github.com/ZSZH12138/TARCA`，分支 `main`
> 协议标识：`TARCA-E2E-STAGE-PROTOCOL-2.0`

## 1. 文档用途与权威边界

本文件记录上述时间点和 Git 提交下的 Stage 0 实际执行结果，用于向 Stage 1A 交接。它只回答“已经产生了哪些可验证输入、当前核验结果是什么、下一阶段可以依赖什么”，不修改项目计划书、具体实施计划、端到端协议、预注册或任何冻结 artifact 的语义。

若本文件与结构化 artifact 不一致，以经哈希校验的结构化 artifact 为准；若与计划书或协议冲突，以权威文档规定的优先级为准。本文件不是持续更新的状态面板，也不能代替 `GateDecision`、`ResearchContractManifest` 或 `Stage0CompletionReceipt`。

## 2. Stage 0 交接结论

| 核验对象 | 快照结果 | 交接含义 |
|---|---|---|
| Stage 0 聚合核验 | `PASS` | 当前 Stage 0 输入、合同、来源、环境和完成凭证可被重新验证 |
| Research contract | `FROZEN` | Stage 1A 应绑定当前研究合同，不得静默改写其输入 |
| Gate 0 | `PASS` | 经人工核验并授权的新颖性 Gate 允许进入下一阶段 |
| Completion receipt | `COMPLETED` | 完成凭证已绑定同版研究合同、Gate decision 和 artifact index |
| 默认执行环境 | `DEFAULT_EXECUTION_PROFILE` | 提供可复现起点，但允许用户授权切换 backend |
| Stage 1+ 实现 | 未包含 | Stage 1A 从统一数据契约和数据边界开始，不继承历史占位实现 |

在不修改冻结研究输入的前提下，本快照支持 Stage 1A 启动。Gate 0 的 `PASS` 是人工授权的新颖性检查结论，不等于 TARCA-C1～C3 已通过后续正式证伪实验；候选贡献仍受预注册、负对照和后续 Gate 约束。

## 3. Stage 0 已建立的功能

| 功能 | 实际能力 | 对后续研究的意义 |
|---|---|---|
| 严格公共契约 | 使用 strict、frozen、`extra="forbid"` 的 Pydantic contract，并提供内容哈希类型 | 防止下游通过临时字典或额外字段静默改变接口语义 |
| Artifact 引用与存储 | 根据实际文件字节生成 `ArtifactRef`，支持安全相对路径、原子写入、重新加载和哈希复核 | 让 Stage 1A 能准确引用输入，而不是依赖文件名或人工记忆 |
| 研究合同冻结 | 将预注册、新颖性声明、假设、术语、环境和相关工作装订为 `ResearchContractManifest` | 为后续实验提供统一、可审计的研究边界 |
| Gate 0 校验 | 校验人工签发的 `GateDecision`、状态、证据类型和 evidence hash；不实现自动新颖性判断 | 保留人工授权边界，同时对缺失或失配证据 fail closed |
| 完成凭证 | `Stage0CompletionReceipt` 绑定研究合同、Gate decision 和 artifact index | 防止后续只看到 `PASS` 字样却使用了不同版本的输入 |
| 默认冻结与授权覆盖 | 冻结产物默认拒绝覆盖；用户可显式授权，并要求理由、归档和前后哈希回执 | 保护可复现性，同时保留受控修订能力 |
| 相关工作矩阵 | 固定 canonical 列结构、必需值、唯一 work ID 和最低工作集合 | 为候选贡献、基线选择和直接碰撞复核提供结构化入口 |
| 第三方来源治理 | 记录官方论文/仓库、commit、许可证状态、允许动作及 dependency release 绑定 | 避免浮动依赖、来源混淆和未知许可证代码复制 |
| 环境发现与 doctor | 检查 Python、PyTorch、POT、pyvene、确定性、hook、Sinkhorn、磁盘和写权限 | 证明最小研究工具链可运行，同时不把本机硬件写成项目上限 |
| Stage 0 聚合检查 | `verify_stage0()` 同时复核研究输入、artifact、Gate 0、环境和完成凭证 | 为本地、CI 和未来服务器提供同一交接门禁 |
| 严格运行报告 | `run_doctor()` 与 `verify_stage0()` 返回命名的冻结报告类型，CLI 再序列化为 JSON | 防止公共检查结果退化为无约束字典 |

## 4. 冻结研究输入

以下引用来自 `artifacts/stage0/artifact_index.json`，哈希算法均为 SHA-256。

| Artifact type | 路径 | 内容哈希 |
|---|---|---|
| `PREREGISTRATION` | `docs/preregistration_v0.md` | `cc21e68c3219aed8dddbebb1b946fd168e2772904f0c57c7a9207083d4dae4af` |
| `NOVELTY_CLAIMS` | `docs/novelty_claims.md` | `5ce1daa1e635e9f85966f90443883202e6d5cf4fac37d09e78e05868f8a13950` |
| `ASSUMPTION_LEDGER` | `docs/assumption_ledger.md` | `553d9f58699c2d0be5dfdd6a217a0c9592c56045c2a8fe52a9caddcbec80499f` |
| `TERMINOLOGY` | `docs/terminology.md` | `4a4dbfb5b44151b8009ed94ff4a83847f978e787dcbfac57068a7bdafee74439` |
| `ENVIRONMENT_BUNDLE` | `artifacts/stage0/environment_bundle.json` | `0b2dd4e9c5bba08feb8a245442621db2b5d20ad8333a096554eaa96c3d855e88` |
| `RELATED_WORK_BUNDLE` | `artifacts/stage0/related_work_bundle.json` | `ec16e67f43a090b7804cbadb7e9bf40af7c3bdc5e4bccebf43ac68c471a7ed00` |
| `RELATED_WORK_MATRIX` | `docs/related_work_matrix.csv` | `48f6da6efb4cdbebb45ce2fb78f0b76120dedd003eed50126ab4cc33df1dc345` |
| `THIRD_PARTY_VERSIONS` | `third_party_manifest/sources.yaml` | `ac6c1dab6c66aea0d4cc29038defbc45f8f3168e1b636c5e72e43c37034a27c6` |
| `PYPROJECT` | `pyproject.toml` | `e1b6010c06f78221104ec8f0da9e3d5ba28914d209927c50b3d0dea744d84bb4` |
| `UV_LOCK` | `uv.lock` | `0b11479749a005b9d967650f47c34c303251d0c6e9a0a156b44e223aa415c52c` |
| `ENVIRONMENT_PROFILE` | `artifacts/stage0/environment_profile.json` | `5a33add92fc8cb6d892b85557ebc9cd2b51f52f7dca57c839227cf216327c994` |

## 5. 交接控制产物

| 产物 | 状态/作用 | 文件 SHA-256 |
|---|---|---|
| `research_contract_manifest.json` | `FROZEN`；统一研究合同 | `514602c477de95329738b3012d1707decba7beb39b3296123cc58b6e9c2c45a3` |
| `gate0_decision.json` | `GATE_0_NOVELTY = PASS`；人工授权决策 | `55a9fda0ea422b740dcf7b93973b58f0165e46328d371739b7458180f3d0c0e8` |
| `artifact_index.json` | 列出并绑定冻结研究输入 | `e51f9ece729df5bf2c34f1a5f26579cdea8752e89ac77b9d0eb957c199ee5053` |
| `environment_profile.json` | 默认可复现环境，不是算力上限 | `5a33add92fc8cb6d892b85557ebc9cd2b51f52f7dca57c839227cf216327c994` |
| `environment_bundle.json` | 绑定 pyproject、lock 和默认 profile | `0b2dd4e9c5bba08feb8a245442621db2b5d20ad8333a096554eaa96c3d855e88` |
| `related_work_bundle.json` | 绑定相关工作矩阵和第三方来源清单 | `ec16e67f43a090b7804cbadb7e9bf40af7c3bdc5e4bccebf43ac68c471a7ed00` |
| `stage0_completion_receipt.json` | `COMPLETED`；绑定研究合同、Gate 和 index | `2057bb50e67bf48bbd39c223f402e776daf9ecab1989b736f8a8d1a302ff7847` |

完成凭证记录的完成时间为 `2026-08-20T10:25:59.472496Z`。

## 6. 快照时重新执行的验证

### 6.1 聚合门禁

命令：

```powershell
.\.venv\Scripts\python.exe scripts/check_stage0.py --json
```

结果摘要：

- `status = PASS`；
- `research_contract_status = FROZEN`；
- `gate0_status = PASS`；
- `completion_status = COMPLETED`；
- related-work matrix：22 行、22 个唯一 work ID；
- third-party manifest：14 个来源；
- 精确冻结的 dependency release：2 个，lock 绑定：2 个；
- doctor 的 7 项检查全部为 `PASS`；
- `gpu_required = false`；
- `execution_backend_replaceable = true`。

### 6.2 自动化测试与覆盖率

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=tarca --cov-report=term --cov-fail-under=80 -q
```

结果：40 项测试全部通过，总覆盖率 87.63%，满足项目配置的 80% 下限。

### 6.3 静态与依赖检查

| 检查 | 结果 |
|---|---|
| Ruff lint | `PASS` |
| Ruff format check | `PASS` |
| Mypy strict | `PASS`，13 个 source files 无问题 |
| `uv lock --check` | `PASS`，解析 120 个包 |

### 6.4 第三方远程来源复核

命令：

```powershell
.\.venv\Scripts\python.exe scripts/run_reference_smoke.py --network
```

结果：`REMOTE_VERIFIED / PASS`，复核 14 个来源、27 个远程入口和 2 个 release tag-to-commit 绑定。5 个来源的许可证状态为 `UNKNOWN`；契约禁止把这些来源作为 dependency 或复制代码，只允许按清单声明的 reference/static 边界使用。

### 6.5 Git 与 CI 证据

- Git commit：`2f6c21aa66e137c577635632ee6cc02fefbcbd66`；
- commit message：`chore: publish clean stage0 baseline`；
- Git commit time：`2026-08-20T19:54:45+08:00`；
- 本地 `main` 与 `origin/main` 在取证开始时一致；
- GitHub Actions workflow：`stage0-ci`；
- CI conclusion：`success`；
- CI run：`https://github.com/ZSZH12138/TARCA/actions/runs/32366172323`。

## 7. 默认环境与算力边界

冻结的默认 profile 为 `stage0-default-local`：Windows/AMD64、CPython 3.11.15、12 个逻辑 CPU、约 16.9 GB RAM；当前 profile 未检测到 CUDA，已验证 PyTorch `float32` 和 `float64`。

该 profile 只证明一个可复现的最小 CPU 起点：

- `profile_role = DEFAULT_EXECUTION_PROFILE`；
- `execution_backend_replaceable = true`；
- `compute_boundary_fixed = false`；
- 无 GPU 不阻断 Stage 0；
- Stage 1+ 可在用户授权后使用本地其他环境、单机服务器或多卡服务器；
- 更换 backend 不得改变 seed、split、dataset、checkpoint、metric、Gate 或其他 scientific identity。

## 8. 明确未交接为“已实现”的能力

Stage 0 没有实现或预置以下 Stage 1+ 科学能力：

- Stage 1A 的数据 registry、WindowBatch、Arrow schema 和 sealed-data enforcement；
- 合成 regime-switching SCM、counterfactual oracle 或联合位置真值；
- PatchTST、iTransformer 或其他正式概率预测模型；
- mechanistic adapter、内部表示 capture/intervene 或 intervention pair 执行；
- OT/UOT 定位、DAS/HyperDAS、effect bridge 或抽象指标；
- DRO、sequential unseen-regime 训练/评估；
- 通用时间序列实验、金融压力测试、正式统计分析或发布流水线。

这些能力必须按照项目计划书、协议书和具体实施计划从相应 Stage 开始实现，不得把 Stage 0 的合同骨架解释为科学功能已经存在。

## 9. Stage 1A 可直接消费的输入

Stage 1A 可以直接使用：

1. `ResearchContractManifest` 及其六类顶层研究引用；
2. 预注册中的数据、划分、泄漏、指标和变更控制边界；
3. 术语表中关于 causality、horizon/lag、fit/freeze/apply 和 zero-refit 的定义；
4. 假设台账中的 Stage 1A 数据、pair 和身份风险；
5. `src/tarca/contracts/` 中唯一的基础 contract、artifact 和 Gate 类型；
6. `ArtifactStore` 的内容哈希、原子发布、相对路径和 reload/verify 机制；
7. `pyproject.toml`、`uv.lock`、默认环境 profile 和可替换 backend 策略；
8. 第三方来源清单中已经冻结的引用、依赖和许可证边界。

## 10. Stage 1A 建议启动顺序

使用功能层语言描述，下一阶段应按以下顺序开始：

1. **确认交接输入没有变化**：运行 `scripts/check_stage0.py`，确保合同、Gate 和完成凭证仍可通过；
2. **建立统一数据入口**：让数据只能通过明确的数据集标识和 registry 被找到，不让科学代码直接拼接任意路径；
3. **定义窗口数据合同**：明确历史长度、预测期、变量名称、目标、mask、时间边界、partition、device 和 dtype；
4. **阻断未来信息泄漏**：保证 scaler、窗口生成和环境定义只在允许的训练范围拟合；
5. **固定数据身份**：为原始数据、切分、窗口和 schema 计算可复核哈希；
6. **建立持久化格式**：使用协议规定的严格 Manifest 与 Arrow/Parquet schema 保存数据边界；
7. **验证 Stage 1A 输出可以重新加载**：确认 Stage 1B 和 Stage 2 只消费标准 bridge，不读取 Stage 1A 内部实现；
8. **保持范围隔离**：Stage 1A 不提前实现 SCM、预测模型、内部干预、定位或鲁棒优化。

## 11. 必须保持的交接不变量

- 不得修改、复制或重建第二套 `StrictContractModel`、`ArtifactRef`、Gate 或基础哈希类型；
- 不得把 Gate 0 的人工 `PASS` 改写为自动新颖性证明；
- 不得把 `PROVISIONAL` 候选贡献写成已经被实验证实；
- 不得绕过 `ResearchContractManifest` 直接读取未绑定的研究输入；
- 不得把默认本地环境解释成 CPU、GPU、显存或服务器数量上限；
- 不得在测试集或 sealed scope 上重新拟合 scaler、切分、mapping 或其他 scientific state；
- 不得把计划书、协议书或实施计划改写成阶段状态报告；
- Stage 1A 的新契约应扩展 `src/tarca/contracts/`，不应创建重复的基础契约层。

## 12. 需要重新打开 Stage 0 的情形

以下事件触发受控复核，而不是静默继续：

- 用户要求重新核验 Gate 0，或项目收到覆盖现有窄声明的明确直接碰撞证据；
- 预注册、新颖性声明、假设台账、术语、相关工作 bundle 或依赖锁发生内容变化；
- artifact 文件缺失、内容哈希失配、引用路径逃逸或 completion receipt 绑定过期；
- 第三方 dependency 的 package version、release tag、release commit 或许可证边界变化；
- Stage 1A 发现现有研究合同无法表达必要的数据或身份边界。

研究合同和 Stage 0 artifact 默认冻结。需要替换活动版本时，必须获得用户显式授权并给出理由，按当前协议执行归档和前后哈希审计；仅切换等价 execution backend 不构成自动重开 Stage 0 的理由。

## 13. 交接时建议执行的命令

```powershell
.\.venv\Scripts\python.exe scripts/check_stage0.py --json
.\.venv\Scripts\python.exe -m pytest --cov=tarca --cov-report=term --cov-fail-under=80 -q
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\ruff.exe format --check src tests scripts
.\.venv\Scripts\mypy.exe src
D:\software\MyAnaconda\envs\tarca-local-py311\python.exe -m uv lock --check
.\.venv\Scripts\python.exe scripts/run_reference_smoke.py --network
```

远程来源检查需要网络；Stage 0 的核心合同、测试和 offline gate 不依赖服务器或 GPU。

## 14. 快照限制

本快照描述的是提交 `2f6c21aa66e137c577635632ee6cc02fefbcbd66` 在 2026-08-20 的执行证据。快照文件本身没有写入冻结 artifact index，也不会反向改变任何 content hash。后续若代码、合同、artifact 或依赖发生变化，应生成新的带日期/版本快照，不得覆盖本文件后继续沿用其中的旧哈希和验证结论。
