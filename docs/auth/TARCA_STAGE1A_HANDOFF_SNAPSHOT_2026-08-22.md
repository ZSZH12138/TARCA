# TARCA Stage 1A 执行与交接历史快照

> **快照编号**：`TARCA-STAGE1A-HANDOFF-2026-08-22`<br>
> **快照日期**：2026-08-22<br>
> **阶段状态**：`PASS`（仅指 Stage 1A 工程边界验收）<br>
> **代码基线提交**：`15aa79f`（`fix: close stage1a artifact and partition integrity gaps`）<br>
> **稳定协议身份**：`TARCA-E2E-STAGE-PROTOCOL-2.0`<br>
> **契约 Schema 版本**：`1.0.0`<br>
> **目标分支**：本地与远端 `main`<br>
> **远端仓库**：`https://github.com/ZSZH12138/TARCA`

---

## 1. 快照目的

本文件记录 Stage 1A 从协议兼容性修正、实施方案、TDD 实现、完整性修复、验收复核到 GitHub 同步的历史状态，用于后续 Stage 1B、Stage 2、Stage 10、Stage 11 和复现审计交接。

它是历史快照和工程验收材料，不是新的项目计划书、端到端协议、研究契约、GateDecision 或科学结果。若本文件与上位文件冲突，必须服从第 2 节列出的权威层级。

---

## 2. 权威依据与优先级

Stage 1A 的实现和本次判定以以下文件为依据：

1. `docs/auth/TARCA_项目计划书.md`：决定研究问题、Gate、证据层级、实验顺序和停止条件；
2. `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`：决定 Stage I/O、跨模块类型、标准函数和 Artifact/Gate/Execution 连接；
3. 冻结的 Stage 0 研究契约和完成制品：约束 Stage 1+ 正式实验边界；
4. `docs/auth/TARCA_具体实施计划.md`：在不与上位文件冲突时指导工程执行；
5. 本快照、测试结果和检查脚本输出：只能证明对应实现满足哪些验收条件，不能反向修改科学或接口协议。

协议兼容性修订采用已批准的 `docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0001.md`：

- SHA-256 wire format 统一为 64 位小写十六进制，不带 `sha256:` 前缀；
- `SealedAccessGrant` 绑定 dataset、scope、partition、授权 ArtifactRef 和 UTC 有效期；
- sealed 读取必须在物理 I/O 前验证，registry 保护不能被调用者降级；
- 稳定协议身份保持 `TARCA-E2E-STAGE-PROTOCOL-2.0`。

---

## 3. Stage 1A 的功能定位

Stage 1A 建立项目共用的“数据、模型接口和实验产物语言”，回答以下问题：

1. 一份数据如何被准确识别和验证？
2. 一个时间窗口必须包含哪些 Tensor、名称、掩码和时间边界？
3. 预测器、机制模型、概念提取器和干预请求如何接入？
4. 预测、干预、效应、指标和定位结果如何保存和复核？
5. sealed 数据如何授权，训练、验证和测试分区如何防止泄漏？

Stage 1A 不负责生成正式合成 SCM，不下载正式数据，不训练预测模型，不执行模型内部干预，不运行 OT/DAS/DRO，也不产生可以支撑 TARCA 科学主张的实验结果。

---

## 4. 实际执行流程

```text
核验 Stage 0 交接与协议兼容性
  -> 冻结统一契约版本和严格基础类型
  -> 定义数据 registry、manifest 和 WindowBatch
  -> 定义预测、概念、干预、指标和 adapter 接口
  -> 固定运行目录与五类 Arrow Schema
  -> 实现类型化原子 ArtifactStore
  -> 实现 registry 驱动的 persisted dataset repository
  -> 建立 sealed-before-I/O、hash 和 partition 防泄漏边界
  -> 完成最小 typed data 到 verified artifact 闭环
  -> 定向审计并修复高风险完整性问题
  -> 冻结 Stage 1A 边界并形成交接快照
```

整个流程按 TDD 执行：先建立能够复现缺口的失败测试，再写最小实现使测试通过，最后做局部重构和回归检查。

---

## 5. 已交付的主要模块

| 能力 | 主要位置 | 实际交付 |
|---|---|---|
| 统一基础契约 | `src/tarca/contracts/base.py` | 严格 Pydantic 基类、SHA-256、UTC、canonical JSON、协议和 Schema 版本 |
| 数据访问契约 | `src/tarca/contracts/data_access.py` | `DatasetSpec`、物理分区、`AccessScope`、`SealedAccessGrant`、sealed 校验 |
| 数据与窗口契约 | `src/tarca/contracts/data.py` | registry、`DataManifest`、`WindowBatch`、`LeakageAudit`、跨分区隔离审计 |
| 预测契约 | `src/tarca/contracts/forecasts.py` | `ForecastDistribution` 及 mean/scale/quantile/logit/sample 校验 |
| 概念契约 | `src/tarca/contracts/concepts.py` | `ConceptSpec`、`ConceptBatch`、概念干预和 `ConceptExtractor` |
| 干预契约 | `src/tarca/contracts/interventions.py` | site、pair、resolved pair、spec、result 及子空间校验 |
| 模型接口 | `src/tarca/contracts/adapters.py` | `ForecastPredictor` 与 `MechanisticModelAdapter` Protocol |
| 指标契约 | `src/tarca/contracts/metrics.py` | `MetricContext`、`MetricRecord` |
| Arrow Schema | `src/tarca/contracts/arrow_schemas.py` | predictions、intervention pairs、effects、metrics、localization 五套精确 Schema |
| Artifact 元数据 | `src/tarca/contracts/artifacts.py` | `ArtifactRef`、`ArtifactManifest`、`RunManifest` |
| 运行目录 | `src/tarca/artifacts/layout.py` | 固定结果树、相对路径和路径穿越防护 |
| ArtifactStore | `src/tarca/artifacts/store.py` | typed publish/load、fsync、hash、reload、Schema、manifest、completion marker |
| 数据载荷 | `src/tarca/data/payload.py` | 内部 payload manifest、文件角色、路径和哈希约束 |
| 数据解析 | `src/tarca/data/persisted.py` | 从已验证的内存字节解析 `.npy` 和 metadata，不使用 pickle |
| 数据入口 | `src/tarca/data/repository.py` | exact registry、sealed authorization、真实 hash、物理分区读取和 loader audit |
| 阶段检查 | `scripts/check_stage1a.py` | Stage 0 连续性、版本、五类 Schema、冻结目录和阶段边界检查 |
| 范围说明 | `docs/stage1a_scope.md` | Stage 1A/1B 分工、冻结规则和最小验收入口 |

---

## 6. 数据入口和防泄漏边界

### 6.1 数据身份

- registry 只允许精确 `(dataset name, version)` 命中；
- dataset hash 实际绑定 canonical payload manifest；
- manifest 中每个物理文件还必须分别通过字节数和 SHA-256 校验；
- dataset 路径和 payload 路径必须留在各自允许的根目录内；
- `.npy` 只能用 `allow_pickle=False` 解析，object dtype 被拒绝。

### 6.2 sealed 权限

- effective sealed 状态为 `registry_entry.sealed OR access.sealed`；
- 调用者不能用 `AccessScope(sealed=False)` 降级 registry 已声明的保护；
- grant 必须匹配 dataset、scope、partition 和有效时间；
- `hash_dataset()` 读取整个数据集时，grant 必须覆盖所有声明的物理分区；
- 任一授权检查失败都发生在 payload I/O 前，并 fail closed。

### 6.3 物理分区

- `build_windows()` 只读取请求的已有物理分区；
- TEST 不由 TEST_SEEN_REGIME 与 TEST_UNSEEN_REGIME 自动拼接；
- loader 不执行 re-window、re-normalize、cast、shuffle、fill、encode 或 re-split；
- `audit_partition_isolation()` 检查分区标签，并拒绝不同物理分区复用同一 `window_id`；
- 审计函数只检查，不修改、不复制、不拼接 Tensor。

---

## 7. 运行时契约和模型接口

### 7.1 `WindowBatch`

实现检查以下功能边界：

- `x`、`y`、observed covariates、known-future covariates 的 rank 和轴长度；
- batch、history、forecast horizon、target 和 covariate 维度必须一致且必要维度为正；
- feature/target/covariate 名称数量正确、非空、唯一；
- known-future covariate 名称不能与 target 名称重叠；
- mask 形状与数据一致且 dtype 为 `torch.bool`；
- scientific floating Tensor 必须 finite；
- 所有时间为 UTC，窗口边界有序，forecast time 严格递增并位于预测区间；
- validator 不 clone、detach、cast、move device 或改变 `requires_grad`。

### 7.2 预测、概念和干预

- `ForecastDistribution` 检查非空 mean、正 scale、合法 quantile level 和 quantile crossing；
- `ConceptBatch` 检查二维值、bool mask、名称、窗口身份和 definition version；
- `InterventionSite` 与 `InterventionSpec` 明确分离位置目录和执行请求；
- `SUBSPACE_SWAP` 只接受有限、二维、满足冻结容差的正交基；
- `ForecastPredictor`、`MechanisticModelAdapter` 和 `ConceptExtractor` 仅定义接口，不提前绑定具体模型或 hook。

---

## 8. 结果 Schema 与 ArtifactStore

### 8.1 五类表格

以下 Schema 固定列顺序、Arrow 类型、nullability 和 metadata：

1. `PREDICTIONS_SCHEMA`；
2. `INTERVENTION_PAIRS_SCHEMA`；
3. `EFFECTS_SCHEMA`；
4. `METRICS_SCHEMA`；
5. `LOCALIZATION_SCHEMA`。

共同 metadata 包括：

```text
contract_schema_version = 1.0.0
protocol_id = TARCA-E2E-STAGE-PROTOCOL-2.0
artifact_type = <exact table type>
```

Schema 校验比较字段、顺序、类型、nullability 和 metadata，不能以“pandas 能读”代替协议验证。

### 8.2 ArtifactStore 原子发布

实际发布顺序为：

```text
temporary write
-> flush/fsync
-> content hash
-> reload
-> contract/Arrow validation
-> atomic publish
-> manifest
-> completion marker binding manifest hash
-> final verification
-> ArtifactRef
```

任何 hash、reload、Schema、manifest 或 completion marker 校验失败都不得返回 `ArtifactRef`。

加载路径已经加固为“读取一次、验证一次、消费同一份字节”：`load_contract()`、`load_arrow()`、`load_bytes()` 和 `verify_artifact()` 不再在验证后重新打开数据路径，从而消除 hash-to-load 竞态窗口。

---

## 9. 高风险问题及修复历史

### 9.1 协议兼容性问题

1. SHA-256 wire format 的文字和冻结 Stage 0 artifact 不一致；
2. 协议引用 sealed grant，但最初没有定义字段和读取函数参数。

处理结果：通过批准的 CCP-0001 采用兼容性修订，不重写 Stage 0 artifact 字节，不改变稳定协议身份。

### 9.2 首轮 Stage 1A 完整性复核

| 问题 | 修复 |
|---|---|
| `WindowBatch` 曾允许 `H=0` | horizon 必须严格大于 0 |
| dataset 文件通过哈希后又从路径重新打开 | loader 解析通过哈希验证的同一份内存字节 |
| ArtifactManifest 语义篡改不能被发现 | completion marker 绑定 manifest SHA-256 |
| 新 loader 没有具体 `LeakageAudit` 输出 | 增加审计 sidecar，审计失败时标准 loader 同样 fail closed |

对应提交：`11cda34`。

### 9.3 第二轮 Stage 1A 完整性复核

| 问题 | 修复 |
|---|---|
| ArtifactStore load 存在 hash-to-load 竞态 | `_verified_payload()` 对同一份 bytes 完成 hash、manifest 绑定和反序列化 |
| 只检查批内 ID，缺少跨物理分区隔离审计 | 增加 `audit_partition_isolation()`，检查标签和跨分区 `window_id` 交集 |

对应提交：`15aa79f`。

---

## 10. 测试和验收证据

### 10.1 完整实施验收

Stage 1A 初始完成并合并时记录的完整验证：

| 检查 | 结果 |
|---|---|
| 全项目 pytest | `114 passed` |
| 覆盖率 | `83.01%`，高于项目 80% 门槛 |
| Ruff lint/format | PASS |
| mypy | PASS，29 个 source files |
| `scripts/check_stage0.py --json` | PASS |
| `scripts/check_stage1a.py --json` | PASS |
| Stage 0 Gate | PASS，研究契约保持 FROZEN |
| 锁定依赖审计 | 项目锁定运行依赖无已知漏洞 |

### 10.2 最终完整性修复后的复核

在代码基线 `15aa79f` 上记录：

| 检查 | 结果 |
|---|---|
| 可运行的 Stage 1A 测试 | `62 passed` |
| Artifact/partition 定向测试 | `29 passed` |
| Ruff lint | PASS |
| Ruff format | PASS，55 个文件格式正确 |
| `scripts/check_stage1a.py --json` | PASS |
| Stage 0 continuity | PASS |
| Arrow Schema 数量 | 5 |
| 正式数据生成 | false |
| 模型训练 | false |

最终复核使用的既有 Conda 环境没有安装 `mypy`，因此依赖 mypy 的 adapter 静态测试没有在最后一轮重复执行；adapter 代码未在两次完整性修复中改动，且初始完整验收已经记录 mypy PASS。该环境差异属于非强阻断复核限制，不改变 Stage 1A 的工程状态。

---

## 11. 兼容性实现决策

以下决策用于落地协议未展开的工程细节，不改变公开科学语义：

- Adapter 使用协议正式名称 `ForecastPredictor` 和 `MechanisticModelAdapter`，不再复制实施计划中的概念性别名；
- registry 作为 `PersistedDatasetRepository` 的构造依赖注入，避免全局变量或工作目录推断；
- `hash_dataset()` 的全量读取要求 sealed grant 覆盖全部 available partitions；
- persisted 测试载荷采用单数组 `.npy` 加 canonical JSON manifest，不使用 pickle 或不确定 zip 元数据；
- 协议中的 Arrow `float` 持久化为 `float64`，horizon 为 `int32`，时间为 `timestamp[ns, UTC]`；
- localization 的 `score/cost` 展开为两个 nullable 字段，并要求至少一个存在；
- 协议未定义 `ArtifactLayout` 的公开字段，因此实现内部安全 layout validator，不臆造新的跨阶段模型；
- Stage 1A 只识别 `STAGE1_SYNTHETIC_CONFIG` 并 fail closed，生成器归 Stage 1B。

---

## 12. 明确没有执行的事项

截至本快照，以下事项没有发生：

- 没有下载正式时间序列或金融数据；
- 没有生成 Stage 1B 正式合成 SCM 数据；
- 没有拟合 normalization、encoder 或 temporal split policy；
- 没有训练 PatchTST、iTransformer 或其他预测器；
- 没有执行模型内部 activation intervention；
- 没有运行 source matching、effect metric、OT、DAS 或 DRO；
- 没有生成正式实验表、论文图或 claim-bearing 结果；
- 没有创建协议未定义的 Stage 1A 完成凭证或自动 Gate。

因此 Stage 1A 的 `PASS` 只能解释为“工程基础设施和数据边界可交接”，不能解释为 TARCA 方法已经得到科学验证。

---

## 13. 冻结与变更控制

Stage 1A 合并后，以下范围由 `src/tarca/artifacts/freeze.py` 纳入 Windows 只读目录：

- `docs/auth/` 权威与历史文件；
- Stage 0 artifact 和历史版本；
- `src/tarca/contracts/`；
- `src/tarca/artifacts/`；
- `src/tarca/data/`；
- `src/tarca/stage0/`；
- `tests/stage0/` 和 `tests/stage1a/`；
- Stage0/Stage1A 检查脚本、依赖锁和冻结研究输入。

没有用户授权时不得修改这些文件。获得授权后，应只解除目标文件只读属性，在隔离 worktree 中修改和复核，完成后恢复只读，并按需要更新历史快照、hash 或 CCP。

---

## 14. Git 实施历史

以下提交构成远端 Stage 1A 同步范围：

| 提交 | 功能 |
|---|---|
| `7c49fa1` | 修复协议兼容性边界，落地 CCP-0001 |
| `d356ba8` | 建立 Stage 1A 契约版本边界 |
| `98d3ed8` | 增加数据 metadata 和 registry 契约 |
| `98b426f` | 实现冻结且严格校验的 `WindowBatch` |
| `1b55f9d` | 增加预测、概念和干预契约 |
| `a5dcb44` | 定义模型与概念 Protocol |
| `aed0d98` | 增加 Run/Artifact metadata 和安全 layout |
| `8ba60d1` | 冻结五类 Arrow Schema |
| `5f911ff` | 实现类型化原子 ArtifactStore |
| `fc6dc25` | 实现严格 persisted dataset repository |
| `7b40ee8` | 完成 Stage 1A 最小端到端契约闭环 |
| `91bc9ee` | 为 Stage 1A 重新绑定冻结 Stage 0 环境证据 |
| `8f39b51` | 加强 Stage 1A 边界校验 |
| `ceae07c` | 统一 Stage 1A 实现格式 |
| `0253a26` | 隔离各阶段测试模块 |
| `f7d505c` | 记录 Stage 1A 完成实施方案 |
| `41423b9` | 冻结批准的 Stage 1A 方案 |
| `11cda34` | 修复窗口、数据和 Artifact 完整性问题 |
| `15aa79f` | 闭合 Artifact 加载和跨分区隔离缺口 |

本快照和实施计划同步位于上述代码基线之后的文档提交中。远端同步不得包括本地虚拟环境、缓存、临时测试数据、凭据或未经本次授权的未跟踪文件。

---

## 15. Stage 1B 交接要求

Stage 1B 可以在本快照基础上实现合成 regime-switching SCM，但必须满足：

1. 复用同一 `src/tarca/contracts/`，不得创建第二套数据或基础 artifact 类型；
2. 生成器输出必须绑定 `SyntheticConfig`、dataset hash、normalization record、物理 splits、truth 和 provenance；
3. 所有窗口必须转换为既有 `WindowBatch`，不得绕过 validator；
4. 生成完整物理分区后，显式运行 `audit_partition_isolation()`；
5. train-only transform 只能在 TRAIN 拟合，并通过 typed ArtifactStore 发布状态；
6. seen/unseen truth、future noise 和 scientific identity 继续服从冻结协议；
7. Stage 1B 不得因为实现便利改写 Stage 1A Schema 或接口；如确需协议变更，应先立新的 CCP。

Stage 1B 的核心任务是“把有真值的合成世界装入已经冻结的容器”，而不是重新设计容器。

---

## 16. 最小复核入口

在具有项目锁定依赖的环境中执行：

```powershell
python scripts/check_stage0.py --json
python scripts/check_stage1a.py --json
python -m pytest tests/stage1a -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
git diff --check
```

若本机既有环境缺少测试依赖，不得为追求表面通过而污染冻结环境；应使用项目锁定环境或在报告中明确记录未执行项和原因。

---

## 17. 功能层总结

用简单的话说，Stage 1A 已经完成四类基础设施：

1. **身份证**：明确每份数据、契约和 artifact 到底是谁；
2. **门禁**：决定谁能读取哪个受保护分区，并在读文件前拒绝非法请求；
3. **标准箱子和插头**：规定时间窗口、预测结果、概念、干预和模型接口的统一形状；
4. **保险箱和账本**：用 Schema、hash、manifest、completion marker 和原子发布保存可复核结果。

Stage 1A 没有证明 TARCA 方法有效；它确保从 Stage 1B 开始，后续科学实验不会因为数据含义、权限、接口或文件格式漂移而失去可解释性和可复现性。
