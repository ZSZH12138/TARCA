# TARCA Stage 1A 完成方案

> **执行要求：** 后续实现使用 `superpowers:executing-plans`，并严格遵守 TARCA 的单代理规则；所有步骤由当前主代理串行完成，不使用子代理或并行代理。

**目标：** 在不下载正式数据、不训练模型、不执行内部干预的前提下，建立 Stage 1A 所要求的唯一契约层、严格数据入口、可验证的持久化 Schema 和最小端到端证据，使 Stage 1B、Stage 2、Stage 10、Stage 11 可以依赖稳定接口继续工作。

**架构：** 以 `src/tarca/contracts/` 为跨阶段语义的唯一权威；在其下定义冻结的 Pydantic 元数据契约、冻结的运行时张量契约和 Python Protocol。数据物理读取由一个显式注入 registry 的本地 repository 负责，ArtifactStore 负责原子写入和 Schema 验证。所有边界均 fail-closed，不在验证器或 loader 中隐式重采样、归一化、类型转换、设备迁移、打乱或重新分割。

**技术栈：** Python 3.11、Pydantic v2、PyTorch 2.13、NumPy、PyArrow 25、pytest、mypy、ruff、uv。

**权威依据（优先级从高到低）：**

1. `docs/auth/TARCA_项目计划书.md`：研究阶段、Gate、先后顺序。
2. `docs/auth/TARCA_具体实施计划.md`：Stage 1A 的工作边界、交付物和验收内容。
3. `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`：接口、类型、输入输出和跨阶段语义。
4. `docs/auth/TARCA_STAGE0_HANDOFF_SNAPSHOT_2026-08-20.md` 与 `artifacts/stage0/research_contract_manifest.json`：已完成状态和冻结科学输入的可验证证据。
5. 外部论文、官方资料和 GitHub 仓库：只用于补强工程做法，不得覆盖上述文件。

---

## 1. Stage 1A 的能力定义

Stage 1A 不是“开始跑模型”，而是为之后所有工作修建一条有检查站的数据与接口通道。它必须回答五个功能问题：

1. **这份数据到底是谁？** 用 `(name, version)`、registry 条目和真实 SHA-256 唯一识别。
2. **这批窗口到底长什么样？** 用 `WindowBatch` 固定形状、名字、时间、掩码、dtype、device 和 split 语义。
3. **模型、概念和干预怎样接进来？** 用 Protocol 和请求/结果契约固定调用边界，不绑定具体实现。
4. **结果如何可靠保存和再读取？** 用严格 JSON/Pydantic、Arrow Schema、Parquet 和原子 ArtifactStore 固定持久化格式。
5. **怎样证明没有偷看测试集或悄悄改变数据？** 用 `LeakageAudit`、sealed grant、registry no-downgrade 和身份保持测试给出证据。

### 1.1 包含范围

- 数据 registry、数据 manifest、窗口摘要、泄漏审计。
- `WindowBatch`、`ForecastDistribution`、概念和干预相关运行时契约。
- `ForecastPredictor`、`MechanisticModelAdapter`、`ConceptExtractor` 等结构化接口。
- Run/Artifact/Metric 等严格元数据契约。
- 协议规定的 Arrow 表 Schema 和 Parquet 往返验证。
- 类型化、原子、本地 ArtifactStore。
- 一个只读取测试夹具的最小 persisted-dataset repository。
- 少量但覆盖高风险边界的单元、集成和最小端到端测试。

### 1.2 明确不包含

- 正式数据下载或正式数据产物。
- Stage 1B 的合成数据生成器和真实性 Gate。
- 模型训练、预测质量比较、概念学习或内部激活干预。
- 自动归一化、自动编码、自动补值、滑窗生成、打乱或重分割。
- Kedro、TensorFlow Data Validation、Hugging Face Datasets、PyTorch Forecasting 或 GluonTS 作为运行时依赖。
- 尚未被协议定义的跨阶段公开类型。

---

## 2. 研究结论与选型

本轮采用“有针对性的工程文献综述”，不是系统综述或新颖性检索。检索重点是数据文档、时间序列泄漏、严格 Schema、协议接口和可复现指纹。

| 证据 | 可借鉴做法 | TARCA 中的落点 | 不照搬的部分 |
|---|---|---|---|
| Gebru 等，*Datasheets for Datasets* | 记录来源、组成、处理和维护信息 | `DataManifest`、registry 和 payload 文件清单 | 不引入自由文本大表单作为运行时契约 |
| Kapoor & Narayanan，*Leakage and the Reproducibility Crisis in ML-based Science* | 将时间、预处理和 split 泄漏作为显式审计对象 | `LeakageAudit`、物理 split、train-only 变换边界 | Stage 1A 不实现完整科学审计平台 |
| Bergmeir 等；Cerqueira 等 | 时间序列评估应尊重时间顺序，非平稳场景不能随意使用随机交叉验证 | 禁止 shuffle/resplit；TEST 不可拼回训练 | 不在 Stage 1A 选择最终评估算法 |
| Breck 等，*Data Validation for Machine Learning* | Schema 和数据偏移应在入口处检查并快速失败 | 精确字段、类型、nullability、metadata 和 leakage 验证 | 不引入 TensorFlow/Beam 技术栈 |
| Sculley 等，*Hidden Technical Debt in ML Systems* | 限制边界侵蚀和未声明数据依赖 | `contracts/` 唯一权威、显式 repository 和 store | 不建立大型平台层 |
| Pydantic 官方文档 | `strict`、`frozen`、`extra='forbid'` 的持久化边界 | JSON 元数据契约 | 不把 Tensor 对象塞进 Pydantic JSON 模型 |
| Apache Arrow 官方文档 | Schema 固定字段、类型、nullability 和 metadata | 五类协议表的精确 Schema | “pandas 能读”不算通过 |
| Python `typing.Protocol` | 静态结构化接口 | predictor/adapter/extractor/store | `runtime_checkable` 不能替代 mypy 和行为测试 |
| Hugging Face Datasets fingerprint | 数据和变换共同进入指纹 | 显式 payload manifest + 每文件 SHA-256 | 拒绝依赖 dill/pickle 或失败后随机 fingerprint |
| Kedro DataCatalog | 数据源由一个中心 catalog 解析 | registry 是唯一入口 | 拒绝动态类路径、任意工厂和隐式全局 catalog |
| PyTorch Forecasting、GluonTS、FEV | 已知未来量、历史量、观测掩码、任务指纹等时间序列语义 | `WindowBatch` 和 forecast manifest 的字段语义 | 拒绝其自动缩放、编码、补值和窗口化行为 |
| TensorFlow Data Validation、GIFT-Eval | Schema/偏移验证以及显式 leakage 标志 | `LeakageAudit` 和测试集边界 | 只作设计参考，不增加重依赖 |

### 2.1 依赖决策

Stage 1A 只新增一个核心研究依赖：

```toml
research = [
  "pyarrow>=25,<26",
]
```

它进入 `research` extra，并同步更新 `uv.lock`。基础安装仍应能运行 Stage 0；因此顶层 `tarca.contracts` 不应无条件导入 `pyarrow`，Arrow Schema 通过 `tarca.contracts.arrow_schemas` 显式使用。

### 2.2 GitHub 代码考察结论

- **采用思想、不采用框架：** Kedro 的单一 catalog、Hugging Face 的指纹、GluonTS/PyTorch Forecasting 的字段语义都值得借鉴。
- **保持 TARCA 最小化：** 这些框架含有 TARCA 明确禁止的隐式转换或远超 Stage 1A 范围的依赖。
- **直接依赖成熟底层库：** Pydantic 负责严格 JSON 边界，PyArrow 负责 Arrow/Parquet 边界，PyTorch 只负责运行时 Tensor。
- **固定检索时间：** 仓库考察基于 2026-08-21 当日默认分支；实现时不从远程仓库复制代码，仅按已记录的公开接口思想自行实现。

---

## 3. 关键兼容性决策

以下决策用于消除权威文档之间未完全展开的实现细节，不改变协议公开语义。

### 3.1 Adapter 命名

实施计划出现过 `ForecastModelAdapter` 的概念性名称，协议给出的正式公开名称是：

```python
class ForecastPredictor(Protocol): ...
class MechanisticModelAdapter(Protocol): ...
```

实现只使用协议名称，不创建第二套别名，避免两条接口分叉。

### 3.2 Registry 如何进入自由函数式接口

协议把 `DatasetRegistryManifest` 列为 Stage 1A 输入，但函数签名没有显式 registry 参数。采用不可变 repository，把 registry 作为构造时依赖：

```python
@dataclass(frozen=True, slots=True)
class PersistedDatasetRepository:
    repo_root: Path
    registry: DatasetRegistryManifest

    def resolve_dataset(self, dataset: DatasetSpec) -> DatasetRegistryEntry: ...
    def build_windows(
        self,
        dataset: DatasetSpec,
        partition: DatasetWindowPartition,
        access: AccessScope,
        grant: SealedAccessGrant | None = None,
    ) -> WindowBatch: ...
    def hash_dataset(
        self,
        dataset: DatasetSpec,
        access: AccessScope,
        grant: SealedAccessGrant | None = None,
    ) -> Sha256Hash: ...
```

这样公开调用仍与协议一致，同时没有全局变量、当前工作目录推断或环境变量回退。

### 3.3 Registry no-downgrade

有效 sealed 状态定义为：

```python
effective_sealed = registry_entry.sealed or access.sealed
```

只要任一方要求 sealed，就必须先验证 grant，再进行任何 payload 文件读取。调用方不能用 `access.sealed=False` 降低 registry 已声明的保护级别。

### 3.4 `hash_dataset` 的 sealed 解释

由于协议的 `hash_dataset` 没有单独的 partition 参数，读取整个数据集并验证哈希时，grant 必须覆盖 registry 声明的全部 `available_partitions`。只覆盖单个 partition 的 grant 可用于 `build_windows`，但不能用于全量 `hash_dataset`。

### 3.5 物理数据包与哈希

测试夹具采用以下内部格式；它不是新的跨阶段公开契约：

```text
dataset-root/
  payload_manifest.json
  train/*.npy
  val/*.npy
  test_seen/*.npy
  test_unseen/*.npy
```

- `.npy` 必须 `allow_pickle=False`，一个文件只保存一个数组。
- `payload_manifest.json` 明确列出相对路径、字节数和每个文件的 SHA-256。
- dataset hash 是 canonical `payload_manifest.json` 内容的 SHA-256；验证时还必须逐文件核对内容哈希。
- 不使用 `.npz` 作为身份根，因为 zip 元数据可能引入不必要的非确定性。
- `PERSISTED_STAGE1` 在 1A 实现；`STAGE1_SYNTHETIC_CONFIG` 只被 registry 识别并 fail-closed，真正生成器归 Stage 1B。

### 3.6 Arrow 数值和 localization 列

协议只写了通用 `float`，实现统一持久化为 Arrow `float64`；`horizon` 为 `int32`，时间为 `timestamp[ns, tz=UTC]`。这只影响报告表序列化，不允许 loader 或运行时契约改变原 Tensor dtype。

localization 表将协议的 “score/cost” 展开为 nullable 的 `score` 和 `cost` 两列，并要求至少一列非空；两列允许同时存在，因为后续定位算法可能同时报告目标分数和计算/运输代价。这样不把两个不同物理量压进含义不清的单列。

### 3.7 `ArtifactLayout`

协议没有给出 `ArtifactLayout` 的公开字段，因此不臆造新的跨阶段 Pydantic 模型。实现一个内部路径/layout 验证器，检查协议规定的结果树和安全相对路径；跨阶段可见对象仍是 `ArtifactManifest`、`ArtifactRef` 和 `RunManifest`。

---

## 4. 目标模块结构

```text
src/tarca/
  contracts/
    base.py                 # schema/protocol 常量、StrictContractModel
    data_access.py          # 已有 access/grant/partition 契约
    data.py                 # registry、manifest、WindowBatch、LeakageAudit
    forecasts.py            # ForecastDistribution
    concepts.py             # concept 契约和 ConceptExtractor
    interventions.py        # site/pair/spec/result 契约
    adapters.py             # ForecastPredictor、MechanisticModelAdapter
    metrics.py              # MetricContext、MetricRecord
    arrow_schemas.py        # 精确 Arrow Schema 与验证器
    artifacts.py            # ArtifactManifest、RunManifest 等
  artifacts/
    layout.py               # 固定结果树和路径验证
    store.py                # 类型化 LocalArtifactStore
  data/
    payload.py              # 内部 payload manifest 和哈希
    repository.py           # registry 驱动的数据入口
    persisted.py            # allowlist 中的 persisted backend
tests/stage1a/
  conftest.py
  test_data_metadata_contracts.py
  test_window_batch.py
  test_forecast_and_intervention_contracts.py
  test_adapter_protocols.py
  test_arrow_schemas.py
  test_artifact_store.py
  test_dataset_repository.py
  test_stage1a_integration.py
scripts/check_stage1a.py
```

所有公开类型从 `tarca.contracts` 选择性导出；需要 PyArrow 的内容保留在显式子模块，避免破坏 Stage 0 的基础安装路径。

---

## 5. 分步实施计划

每个任务都按 RED → GREEN → REFACTOR 执行。提交只是后续实现时的逻辑检查点，本方案本身不执行这些提交。

### Task 1：版本常量、严格基类和研究依赖

**文件：**

- 修改：`pyproject.toml`
- 修改：`uv.lock`
- 修改：`src/tarca/contracts/base.py`
- 测试：`tests/stage1a/test_contract_base.py`

**步骤：**

1. 写失败测试，断言 `CONTRACT_SCHEMA_VERSION == "1.0.0"`、`PROTOCOL_ID == "TARCA-E2E-STAGE-PROTOCOL-2.0"`，以及 `StrictContractModel` 拒绝额外字段、严格类型并冻结赋值。
2. 运行：

   ```powershell
   uv run --extra research --group dev pytest tests/stage1a/test_contract_base.py -q
   ```

   预期因常量或依赖缺失而失败。
3. 在 `research` extra 增加 `pyarrow>=25,<26`，运行 `uv lock`。
4. 添加常量和严格基类配置；Pydantic 的 frozen 是浅层冻结，所有集合字段必须使用 tuple/frozenset 或在 validator 中复制成不可变值。
5. 重跑定向测试和 Stage 0：

   ```powershell
   uv run --extra research --group dev pytest tests/stage1a/test_contract_base.py tests/stage0 -q
   ```

6. 实现提交：`feat: establish stage1a contract version boundary`

### Task 2：数据元数据、registry 和 leakage 契约

**文件：**

- 新建：`src/tarca/contracts/data.py`
- 修改：`src/tarca/contracts/__init__.py`
- 新建：`tests/stage1a/test_data_metadata_contracts.py`

**核心公开对象：**

```python
class DatasetSourceKind(str, Enum):
    STAGE1_SYNTHETIC_CONFIG = "STAGE1_SYNTHETIC_CONFIG"
    PERSISTED_STAGE1 = "PERSISTED_STAGE1"

class DatasetRegistryEntry(StrictContractModel): ...
class DatasetRegistryManifest(StrictContractModel): ...
class DataSplitSummary(StrictContractModel): ...
class WindowContractSummary(StrictContractModel): ...
class DataManifest(StrictContractModel): ...

@dataclass(frozen=True, slots=True)
class LeakageAudit:
    passed: bool
    findings: tuple[str, ...]
```

**验证规则：**

- registry 中 `(name, version)` 唯一；source kind 只允许两个枚举值。
- 相对路径不得为空、绝对、含 `..` 或逃出根目录。
- 数据哈希严格匹配 64 位小写十六进制。
- available partition 不重复；`TEST`、`TEST_SEEN_REGIME`、`TEST_UNSEEN_REGIME` 不得互相替代或被自动拼接。
- manifest 的数据身份、窗口摘要和 registry 条目一致。
- UTC 使用 timezone-aware datetime，非 UTC offset 输入规范化到 UTC；naive datetime 拒绝。
- `LeakageAudit` 的 `passed` 必须与 findings 一致，不允许有 findings 却标记通过；真实数据的附加 audit context 由后续持久化 artifact 绑定，不向协议基础类擅加字段。

**测试：** 正例 + 额外字段、类型强转、重复 registry、路径逃逸、非法 hash、naive 时间、矛盾 partition、矛盾 audit 的反例。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_data_metadata_contracts.py -q
uv run --extra research --group dev mypy src/tarca/contracts
```

**实现提交：** `feat: add stage1a data metadata contracts`

### Task 3：`WindowBatch` 的无副作用验证

**文件：**

- 修改：`src/tarca/contracts/data.py`
- 新建：`tests/stage1a/test_window_batch.py`

**运行时原则：** `WindowBatch` 使用 `@dataclass(frozen=True, slots=True)`；`validate()` 只检查，返回 `self`，不得创建、clone、detach、cast、move 或改变 `requires_grad`。

**协议字段原型：**

```python
@dataclass(frozen=True, slots=True)
class WindowBatch:
    x: Tensor
    y: Tensor | None
    observed_covariates: Tensor | None
    known_future_covariates: Tensor | None
    x_observed_mask: Tensor | None
    y_observed_mask: Tensor | None
    observed_covariates_mask: Tensor | None
    known_future_covariates_mask: Tensor | None
    regime: Tensor | None
    window_id: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    observed_covariate_names: tuple[str, ...]
    known_future_covariate_names: tuple[str, ...]
    feature_start: tuple[UtcDatetime, ...]
    feature_end: tuple[UtcDatetime, ...]
    prediction_start: tuple[UtcDatetime, ...]
    label_end: tuple[UtcDatetime, ...]
    forecast_time: tuple[tuple[UtcDatetime, ...], ...]
    metadata: Mapping[str, JSONValue]
```

**必须覆盖：**

- 每个 Tensor 的 rank 和协议规定 shape。
- batch、context、horizon、target、covariate、time/mask 轴一致。
- feature/concept 名称数量正确且唯一。
- timestamp 为 UTC、窗口内单调、窗口之间符合 split 摘要。
- mask 为 bool，且语义与观测值一致。
- 浮点输入有限；协议允许缺失的位置只能由 mask 表达。
- 所有有关 Tensor 的 device 一致；需要一致的 Tensor dtype 组合必须一致。
- split 不得混合 TEST 与训练 partition。

**身份保持测试：** 在验证前后逐个断言：

```python
assert after is before
assert after.data_ptr() == before.data_ptr()
assert after.dtype == before.dtype
assert after.device == before.device
assert after.requires_grad == before.requires_grad
```

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_window_batch.py -q
```

**实现提交：** `feat: validate immutable stage1a window batches`

### Task 4：预测、概念、干预和指标契约

**文件：**

- 新建：`src/tarca/contracts/forecasts.py`
- 新建：`src/tarca/contracts/concepts.py`
- 新建：`src/tarca/contracts/interventions.py`
- 新建：`src/tarca/contracts/metrics.py`
- 修改：`src/tarca/contracts/__init__.py`
- 新建：`tests/stage1a/test_forecast_and_intervention_contracts.py`

**实现对象：**

- `ForecastDistribution`
- `ConceptSpec`、`ConceptBatch`、`ConceptIntervention`
- `InterventionSite`、`PairingSpec`、`InterventionPair`、`InterventionPairSet`
- `ResolvedInterventionPairBatch`、`InterventionSpec`、`InterventionResult`
- `MetricContext`、`MetricRecord`

**高风险验证：**

- 预测 mean/scale/quantiles 的 batch、horizon、target 轴一致。
- scale 严格为正且有限；quantile level 严格递增、位于 `(0,1)`，数量与 quantile Tensor 对应。
- concept 名称唯一、轴长度一致、有限值和 mask 语义正确。
- intervention site 的 layer/module/site、time/feature/subspace 选择互不矛盾。
- pair ID 唯一；source/target 配对批次和方向一致。
- `InterventionSpec` 必须显式记录 kind、site、lag 和协议规定的 subspace；scientific identity 由调用它的实验/产物上下文绑定，不擅加到该公开类型。
- runtime Tensor 验证同样保持对象身份、dtype、device 和 `requires_grad`。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_forecast_and_intervention_contracts.py -q
```

**实现提交：** `feat: add forecast concept and intervention contracts`

### Task 5：静态接口和最小行为契约

**文件：**

- 新建：`src/tarca/contracts/adapters.py`
- 修改：`src/tarca/contracts/concepts.py`
- 新建：`tests/stage1a/test_adapter_protocols.py`
- 新建：`tests/stage1a/typecheck_fixtures.py`

**正式接口：**

```python
class ForecastPredictor(Protocol): ...
class MechanisticModelAdapter(Protocol): ...
class ConceptExtractor(Protocol): ...
```

**验证方法：**

- mypy fixture 中定义一个正确 fake 和一个错误 fake，确认参数/返回值不匹配会被静态检查发现。
- 行为测试验证 predictor 不改变 `WindowBatch`，adapter 的位点枚举稳定，extractor 输出通过 `ConceptBatch.validate()`。
- `@runtime_checkable` 只用于轻量 presence check，不把它当签名或语义证明。

**命令：**

```powershell
uv run --extra research --group dev mypy src/tarca tests/stage1a/typecheck_fixtures.py
uv run --extra research --group dev pytest tests/stage1a/test_adapter_protocols.py -q
```

**实现提交：** `feat: define stage1a model and concept protocols`

### Task 6：Run/Artifact 元数据与固定 layout

**文件：**

- 修改：`src/tarca/contracts/artifacts.py`
- 新建：`src/tarca/artifacts/__init__.py`
- 新建：`src/tarca/artifacts/layout.py`
- 新建：`tests/stage1a/test_artifact_metadata.py`

**实现内容：**

- 按协议补齐 `RunManifest`、`ArtifactManifest` 和相关引用字段。
- 所有 hash、schema version、protocol ID、artifact type、relative path 和 scientific identity 都做严格验证。
- layout helper 只接受协议规定的结果树类别；拒绝绝对路径、反斜杠混用、`..`、空段和保留临时文件名。
- 不创建协议未定义的公开 `ArtifactLayout` 模型。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_artifact_metadata.py -q
```

**实现提交：** `feat: add typed run artifact metadata and layout checks`

### Task 7：精确 Arrow Schema 与 Parquet 往返

**文件：**

- 新建：`src/tarca/contracts/arrow_schemas.py`
- 新建：`tests/stage1a/test_arrow_schemas.py`

**Schema 目录：**

```python
PREDICTIONS_SCHEMA: pa.Schema
INTERVENTION_PAIRS_SCHEMA: pa.Schema
EFFECTS_SCHEMA: pa.Schema
METRICS_SCHEMA: pa.Schema
LOCALIZATION_SCHEMA: pa.Schema

def validate_table(table: pa.Table, expected: pa.Schema) -> None: ...
```

**统一 metadata：**

```text
contract_schema_version = 1.0.0
protocol_id = TARCA-E2E-STAGE-PROTOCOL-2.0
artifact_type = <exact table type>
```

**验收：**

- 字段顺序、名称、Arrow 类型、nullability 和 metadata 全部精确相等。
- predictions 的时间是 `timestamp[ns, tz=UTC]`，horizon 是 `int32`。
- 数值报告列是 `float64`。
- localization 每行 `score`/`cost` 至少一个非空。
- JSON metadata、Arrow IPC、Parquet 都执行写入后重读，再次验证精确 Schema 和数据相等。
- 测试故意改变列顺序、float 宽度、timezone、nullability、metadata，均须 fail-fast。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_arrow_schemas.py -q
```

**实现提交：** `feat: freeze stage1a arrow and parquet schemas`

### Task 8：类型化、原子的 LocalArtifactStore

**文件：**

- 新建：`src/tarca/artifacts/store.py`
- 新建：`tests/stage1a/test_artifact_store.py`

**公开能力：**

```python
class ArtifactStore(Protocol):
    def publish_contract(...): ...
    def load_contract(...): ...
    def publish_arrow(...): ...
    def load_arrow(...): ...
    def publish_bytes(...): ...
    def publish_text(...): ...
    def verify_artifact(...): ...
```

**原子发布顺序：**

1. 在目标同一文件系统创建受控临时文件。
2. 写入、flush、fsync。
3. canonical serialization。
4. 计算 SHA-256。
5. 从临时文件重读。
6. 重新验证 Pydantic 或 Arrow Schema。
7. 原子 rename 到最终数据路径。
8. 原子发布 manifest。
9. 最后返回 `ArtifactRef`。

任何一步失败都不得返回 ref；临时文件清理由 `finally` 完成。现有 Stage 0 store 和 artifact 字节保持不变，Stage 0 回归测试证明兼容性。

**故障注入测试：** 重读 Schema 错误、hash 不匹配、rename 前异常、路径逃逸、已有目标冲突、损坏后 verify 失败。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_artifact_store.py tests/stage0 -q
```

**实现提交：** `feat: add atomic typed artifact store`

### Task 9：registry 驱动的 persisted dataset repository

**文件：**

- 新建：`src/tarca/data/__init__.py`
- 新建：`src/tarca/data/payload.py`
- 新建：`src/tarca/data/persisted.py`
- 新建：`src/tarca/data/repository.py`
- 新建：`tests/stage1a/conftest.py`
- 新建：`tests/stage1a/test_dataset_repository.py`

**执行顺序：**

```text
resolve exact (name, version)
  -> compute effective sealed
  -> validate grant before payload I/O
  -> allowlist source kind
  -> safe-resolve every relative path
  -> verify payload manifest hash and file hashes
  -> load requested physical partition only
  -> torch.from_numpy without cast/copy policy
  -> construct WindowBatch
  -> WindowBatch.validate()
```

**关键测试：**

- 精确 version 命中；不存在或模糊 version 拒绝。
- registry sealed 不能被 caller 降级。
- grant 数据集、scope、partition、expiry 任一不符时，在任何 payload `open/stat` 之前失败；用 spy backend 证明调用顺序。
- 全量 hash 要求 grant 覆盖所有可用 partition。
- TEST 不得由 TEST_SEEN/TEST_UNSEEN 自动拼接，也不得合入 TRAIN/VAL。
- `.npy` 只用 `allow_pickle=False`；object dtype 拒绝。
- loader 不归一化、不滑窗、不补值、不打乱、不重分割、不 cast。
- registry expected hash 与实际 hash 不同立即失败。
- `STAGE1_SYNTHETIC_CONFIG` 在 Stage 1A 给出明确 unsupported/fail-closed 错误，不生成数据。
- 测试夹具全部位于 pytest 临时目录，不进入正式 artifacts。

**命令：**

```powershell
uv run --extra research --group dev pytest tests/stage1a/test_dataset_repository.py -q
```

**实现提交：** `feat: add strict stage1a persisted dataset repository`

### Task 10：最小端到端闭环、检查脚本和文档

**文件：**

- 新建：`tests/stage1a/test_stage1a_integration.py`
- 新建：`scripts/check_stage1a.py`
- 新建：`docs/stage1a_scope.md`
- 修改：`README.md`

**最小闭环：**

```text
临时 persisted fixture
  -> registry 精确解析
  -> sealed/partition 检查
  -> dataset hash
  -> WindowBatch 无副作用验证
  -> fake ForecastPredictor
  -> ForecastDistribution 验证
  -> predictions Arrow Table
  -> typed ArtifactStore 原子发布
  -> 重读、Schema 校验、hash verify
```

`scripts/check_stage1a.py --json` 只输出瞬时检查结果，不创造协议未定义的完成 receipt，也不写正式数据。

**完整验证：**

```powershell
uv sync --frozen --extra research --group dev
uv run --frozen --extra research --group dev ruff check src tests scripts
uv run --frozen --extra research --group dev ruff format --check src tests scripts
uv run --frozen --extra research --group dev mypy src
uv run --frozen --extra research --group dev pytest -q
uv run --frozen --extra research --group dev python scripts/check_stage0.py --json
uv run --frozen --extra research --group dev python scripts/check_stage1a.py --json
git diff --check
```

**实现提交：** `test: close the stage1a contract and data boundary`

### 建议执行节奏

| 工作日 | 主任务 | 当日退出条件 |
|---|---|---|
| Day 1 | Task 1–2：版本、严格基类、data metadata/registry | 元数据契约定向测试和 Stage 0 回归通过 |
| Day 2 | Task 3–5：运行时张量契约和 Protocol | 身份保持、形状、量化预测、adapter 静态/行为测试通过 |
| Day 3 | Task 6–8：artifact metadata、Arrow、store | JSON/Arrow/Parquet 往返和原子故障注入通过 |
| Day 4 | Task 9：persisted repository 和 sealed 边界 | sealed-before-I/O、hash、partition 隔离测试通过 |
| Day 5 | Task 10：最小闭环、文档、全量复核和 worktree 合并准备 | 全部 Gate 形成新鲜验证证据；没有正式数据或训练副作用 |

这是按项目计划“Stage 1A 约一周”的执行预算，不是硬性工期承诺。若协议级歧义或 Stage 0 回归失败，应停在对应检查点解决，不挤占后续 Gate。

---

## 6. 验收 Gate

Stage 1A 只有在以下条件同时满足时才算完成：

- [ ] `src/tarca/contracts/` 是所有跨阶段类型的唯一来源，没有重复定义。
- [ ] `CONTRACT_SCHEMA_VERSION` 为 `1.0.0`，协议 ID 为稳定 ID。
- [ ] registry 只做 exact `(name, version)` 解析，真实数据 hash 与预期一致。
- [ ] sealed grant 在任何受保护 payload I/O 前验证，且 registry 保护不可降级。
- [ ] `WindowBatch` 和所有运行时 validator 不改变 Tensor 身份、dtype、device、storage 或 `requires_grad`。
- [ ] loader 不执行 rewindow、renormalize、cast、shuffle、resplit、fill 或 encode。
- [ ] TEST 从未合入 TRAIN/VAL，TEST_SEEN/TEST_UNSEEN 不被隐式拼装成 TEST。
- [ ] 预测、概念、干预、指标和 adapter 接口符合协议正式命名与轴语义。
- [ ] 五类 Arrow Schema 的字段、顺序、类型、nullability 和 metadata 精确验证。
- [ ] JSON、Arrow IPC 和 Parquet 往返后内容和 Schema 一致。
- [ ] ArtifactStore 在返回引用前完成 fsync、hash、重读和 Schema 验证。
- [ ] Stage 1A 最小端到端闭环通过，Stage 0 全部回归通过。
- [ ] ruff、mypy、pytest、Stage 0 check、Stage 1A check 和 `git diff --check` 全通过。
- [ ] 没有下载正式数据、没有训练、没有内部干预、没有创建正式研究产物。

---

## 7. 测试预算与停止规则

用户已明确要求“测试少量即可，不要在这个上面花费大量精力”。因此测试采用风险优先，而不是追求穷举：

- 每个核心契约至少一个有效样例和一个代表性失败样例。
- identity/device/dtype/`requires_grad`、sealed-before-I/O、split leakage、hash、Arrow 精确 Schema 作为不可削减测试。
- 端到端只跑一个小型 CPU fixture，规模控制在 KB/MB 级，不运行 GPU、训练、下载或大规模参数化组合。
- 覆盖率只用于发现明显空白；不为了数字重复编写低价值测试。若项目全局 80% 门槛与少量测试冲突，优先保证新增 Stage 1A 核心模块达到门槛，不扩大到无关目录。
- 完整验证正常应在数分钟内结束；若单项出现异常耗时，先定位死锁/I/O/依赖问题，不以扩大硬件负载解决。

---

## 8. 主要风险与控制

| 风险 | 后果 | 控制 |
|---|---|---|
| 文档中的概念名称与协议正式类型不完全一致 | 出现两套 adapter/contract | 协议命名优先，不建别名 |
| Pydantic frozen 被误当深不可变 | 内部 list/dict 仍可被修改 | 集合字段使用 tuple/frozenset，并测试赋值和内部修改 |
| validator 为方便而 cast/move/clone | 掩盖上游错误、破坏梯度或身份 | 只检查；身份保持测试不可削减 |
| registry sealed 被调用者关闭 | 未授权读取 TEST | `entry.sealed OR access.sealed`，I/O 前验证 |
| hash 只覆盖 manifest 不覆盖文件 | 数据被替换仍看似同一数据集 | canonical manifest hash + 每文件 hash 双重验证 |
| Arrow 表“能读”但 Schema 漂移 | 后续统计脚本含义改变 | 精确 Schema/metadata/nullability 验证 |
| 新 ArtifactStore 破坏 Stage 0 | 已封存证据不可复现 | 不改 Stage 0 字节；每次 store 任务都跑 Stage 0 回归 |
| 重框架引入隐式转换 | 违反 Stage 1A 禁令 | 只依赖 Pydantic/PyArrow/PyTorch/NumPy 的小型自有层 |
| Stage 1B 逻辑提前进入 1A | 范围膨胀、Gate 混乱 | synthetic source 在 1A 只识别并 fail-closed |

---

## 9. Stage 1A 完成后对后续阶段的直接价值

- **对 Stage 1B：** 合成数据生成器只需产出已固定的 payload、manifest 和 `WindowBatch`，不用重新定义数据语义。
- **对 Stage 2：** 模型接入只需实现 `ForecastPredictor`/`MechanisticModelAdapter`，预测输出直接进入冻结的 Arrow Schema。
- **对 Stage 10：** 定位和指标结果已有稳定表结构、scientific identity 和 artifact lineage。
- **对 Stage 11：** 报告与统计层读取的是已验证、可追溯、无 split 降级的数据和结果，不再依赖临时 Python 对象。
- **对整个项目：** 把最容易在后期变成隐性技术债的“数据含义、接口形状、持久化格式、访问权限”提前冻结，使后续算法变化不会破坏证据链。

---

## 10. 外部资料索引

### 论文与官方资料

- Gebru et al., *Datasheets for Datasets*: https://www.microsoft.com/en-us/research/uploads/prod/2019/01/1803.09010.pdf
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*: https://doi.org/10.1016/j.patter.2023.100804
- Bergmeir, Hyndman & Koo, *A Note on the Validity of Cross-validation for Evaluating Autoregressive Time Series Prediction*: https://doi.org/10.1016/j.csda.2017.11.003
- Cerqueira et al., *Evaluating time series forecasting models*: https://link.springer.com/article/10.1007/s10994-020-05910-7
- Breck et al., *Data Validation for Machine Learning*: https://research.google/pubs/data-validation-for-machine-learning/
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*: https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- Apache Arrow data model and Schema: https://arrow.apache.org/docs/python/data.html
- Apache Arrow Parquet: https://arrow.apache.org/docs/python/parquet.html
- Pydantic models and strict mode: https://docs.pydantic.dev/latest/concepts/models/ and https://docs.pydantic.dev/latest/concepts/strict_mode/
- Python Protocol: https://docs.python.org/3/library/typing.html#typing.Protocol

### GitHub 仓库

- Pydantic: https://github.com/pydantic/pydantic
- Apache Arrow: https://github.com/apache/arrow
- Kedro DataCatalog: https://github.com/kedro-org/kedro/blob/main/kedro/io/data_catalog.py
- Hugging Face Datasets fingerprint: https://github.com/huggingface/datasets/blob/main/docs/source/about_cache.mdx
- PyTorch Forecasting TimeSeriesDataSet: https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/data/timeseries/_timeseries.py
- GluonTS: https://github.com/awslabs/gluonts
- TensorFlow Data Validation: https://github.com/tensorflow/data-validation
- AutoGluon FEV: https://github.com/autogluon/fev
- GIFT-Eval: https://github.com/SalesforceAIResearch/gift-eval

---

## 11. 执行建议

按 Task 1 → Task 10 串行实施，每个任务在定向测试通过后才进入下一项。Task 3、Task 7、Task 8、Task 9 是风险最高的四个检查点，必须保留其失败测试；其余测试保持小而有代表性。全部 Gate 通过后先做一次本地复核，再按用户要求通过隔离 worktree 的正常 Git 流程合并进 `main`。
