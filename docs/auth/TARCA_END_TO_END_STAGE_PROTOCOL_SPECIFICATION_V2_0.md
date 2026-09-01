# TARCA End-to-End Stage Protocol Specification v2.0.1

> **中文名**：TARCA 全阶段标准输入/输出与接口契约协议书
> **协议版本**：v2.0.1
> **稳定协议身份**：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> **制定日期**：2026-08-20
> **兼容性修订**：`CCP-0001`；纠正 SHA-256 文字记号并补齐 sealed-access 授权边界，不改变既有 wire format 或科学身份。
> **文档角色**：面向整个 TARCA 科研链路的规范性阶段接口协议；定义每个 Stage 的标准输入、标准输出、公共类、必要函数、持久化产物、Gate 连接与执行面边界。
> **范围**：本文件只定义接口和验证规则；Stage/Gate 的执行记录与科学结论不属于本文。
> **与计划书关系**：项目计划书决定“研究什么、按什么顺序验证、哪些 Gate 决定继续/停止”；本协议决定“每一步具体接收什么、返回什么、类里必须有什么字段、函数如何连接”。
> **服务器关系**：远程服务器只是 Execution Plane 的一种 backend，不得改变 Science Plane 的输入、输出、seed、checkpoint、split、metric、Gate 或 scientific identity。服务器接入安全规则继续以 `TARCA_SERVER_ACCESS_RUNBOOK.md` 为唯一网络/凭据操作规范。

---

# 0. 协议目标

本协议解决一个核心工程问题：

> **任何一个 Stage 做完以后，下一 Stage 必须能够只依赖冻结的类型、ArtifactRef 和公开函数开始工作，而不需要猜测文件名、目录、变量含义、Tensor 轴语义、模型 checkpoint、数据 split、实验 ID、Gate 阈值或服务器状态。**

全链路必须满足：

```text
Stage N 标准输入
  -> 边界校验
  -> Stage N 科学实现
  -> 标准输出对象
  -> Schema/identity/hash 验证
  -> 原子发布 ArtifactRef
  -> Stage N 完成收据
  -> Stage N+1 仅消费标准输出/ArtifactRef
```

禁止出现：

```text
上游函数返回 dict/object/任意路径
  -> 下游凭经验猜字段
  -> 隐式重新切分/重新归一化/重新选模型
  -> 结果无法复现或接口语义漂移
```

---

# 1. 权威层级与冲突处理

## 1.1 文件优先级

发生冲突时按下列规则处理：

1. **项目计划书**：决定研究问题、Gate、证据层级、实验顺序和停止条件。
2. **本协议 v2.0**：决定 Stage I/O、跨模块类型、标准函数、Artifact/Gate/Execution 连接。
3. **Stage 0 未来创建并冻结的研究契约**：创建后约束 Stage 1+ 的正式实验，但在创建前不作为现存文件或当前冲突依据。
4. **执行证据与阶段验收材料**：只能证明对应实现满足哪些验收条件，不能反向修改科学或接口协议。
5. **具体代码实现**：必须服从以上协议；代码不能凭“能够运行”成为新权威。

## 1.2 实施状态中立

本协议不记录、推断或汇总任何 Stage 的实施进度。所有 Stage 章节均为规范性定义，而不是当前状态声明。

- **接口语义**：以协议为准；
- **实施进度**：在协议之外管理，不写入本协议；
- **科学证据等级**：只能由 Gate evidence 给出；
- **任何实施进度变化都不得静默改变协议语义、Gate 或科学身份。**

当前实施事实不写入协议正文；E01 的独立状态与收据记录在
`TARCA_E01_HANDOFF_SNAPSHOT_2026-08-30.md`，Stage 2 完成身份记录在
`TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`，E02 的正式结论与后续交接入口记录在
`TARCA_E02_HANDOFF_SNAPSHOT_2026-09-01.md`。不得用这些快照反向修改本协议接口语义。

---

# 2. 外部证据对协议设计的约束

本协议吸收但不复制以下外部工作/官方工具的关键工程原则：

1. **Causal Abstraction / IIT / DAS**：高层变量与低层表示之间的关系必须由干预一致性定义，而不是只靠相关性或 probe；DAS 需要独立的可干预子空间接口。
2. **PLOT**：定位应允许 coarse-to-fine 的 stage 状态机；OT/UOT 的输入（cost、marginals）和输出（transport plan）必须成为显式对象，而不是隐藏在 notebook 中。
3. **DiRoCA**：distributional robustness 是独立的 fit/apply/evaluate 层，不能与基础预测器或定位器隐式合并。
4. **pyvene**：内部表示干预是独立 capability；普通 predictor 不应因为可预测就自动获得 `capture/intervene`。
5. **POT**：OT solver 是 backend；TARCA 公共契约不能把 POT 的具体类泄露到上层。
6. **PatchTST**：时间序列以 patch token 表示且 channel-independent，因此其时间 patch 轴可标准化，但不能强行把跨变量语义解释成模型原生变量交互。
7. **iTransformer**：每个 variate 序列形成 variate token，因此变量轴需要在 `InterventionSite` 中显式声明。
8. **Pydantic v2**：持久化 Manifest 使用 strict/frozen/extra-forbid。
9. **Python Protocol**：用于静态结构接口；`runtime_checkable` 只做成员存在检查，因此建议配套静态类型检查和行为测试。
10. **Apache Arrow/Parquet**：持久化表必须固定字段名、类型、nullable 和 schema metadata，不能只比较列名。
11. **fev/GIFT-Eval**：时间序列 benchmark 应把 task/window/horizon/data fingerprint 等信息显式写入评估上下文，而不是依赖脚本隐式参数。

这些外部工作是协议设计依据，不是 TARCA 的创新声明。

---

# 3. 不可违反的全局规则

## 3.1 Closed-world Type Rule

所有公开边界输入/输出只能是：

- 本协议明确命名的 TARCA class / enum / Protocol；
- 标准 Python primitive：`str/int/float/bool/bytes/PathLike`；
- 明确科学类型：`torch.Tensor`、`numpy.ndarray`、`pyarrow.Table`、`pyarrow.Schema`；
- `Mapping` / `Sequence` 仅在元素类型也被明确约束时使用。

**禁止使用 `object` 作为已经启用的科学接口参数或返回值。**

## 3.2 Contract Class Rule

- 运行时 Tensor 载荷：`@dataclass(frozen=True, slots=True)`。
- 持久化 Manifest：Pydantic v2 strict + frozen + `extra="forbid"`。
- Contract class 默认是**被动数据对象**，不放入训练、搜索、网络访问、磁盘扫描等科学/执行逻辑。
- 科学逻辑优先放在模块函数或 Protocol 实现中。
- class 的成员方法只允许纯函数性质的 identity/validation helper，不允许隐藏 I/O。

## 3.3 Scientific Identity Rule

以下任一字段变化都视为**新的 scientific identity**，不能作为 retry：

- dataset hash / split；
- model/checkpoint/config hash；
- seed；
- forecast horizon / history length；
- concept definition；
- intervention semantics；
- localization search space；
- effect definition/normalizer policy；
- robustness environment definition；
- metric/Gate specification；
- precision/determinism policy（若会改变 claim-bearing 数值）。

Retry 只能改变：

- attempt id；
- worker；
- GPU/CPU placement；
- launcher；
- scheduling order；
- non-scientific telemetry。

## 3.4 Immutable Completion Rule

```text
COMPLETED_TASK_POLICY = NEVER_RERUN
```

已经完成并通过 artifact 验证的 task：

- 不因服务器中断重新运行；
- 不因 retry 修改 seed/config；
- 新算法/新数据/新协议必须生成新的 `ScientificIdentity` 和 task_id。

## 3.5 No Hidden Fit Rule

任何会“学习/拟合/选择”的对象必须显式区分：

```text
fit(train)
freeze()
transform(validation/test)
```

禁止在 validation/test：

- 重拟合 scaler；
- PCA；
- source matching index；
- effect normalizer；
- concept threshold；
- HMM/regime model；
- OT calibration threshold；
- subspace；
- DRO radius/cost scaler；
- model checkpoint。

除非计划书明确把该实验定义为 oracle/per-regime upper bound，并单独报告。

## 3.6 Science / Governance / Execution 分层

```text
contracts
  ├─ science: data/models/concepts/interventions/effects/localization/robustness/metrics
  ├─ governance: artifacts/gates/authorization
  └─ execution: orchestration/runtime/monitoring/backends
```

硬规则：

- Science 不读 SSH、GPU 数量、worker 数、服务器 host；
- Execution 不读未来标签、sealed truth、partial NLL/CRPS 用于模型选择；
- Governance 不实现训练/OT/DAS/DRO；
- 所有跨 plane 连接必须通过 `ArtifactRef`、`TaskSpec`、`GateDecision`、`SealedAccessGrant` 等契约。

---

# 4. 类型与 Shape 记号

| 记号 | 含义 |
|---|---|
| `B` | batch size > 0 |
| `L` | history length > 0 |
| `H` | forecast horizon > 0 |
| `D` | input feature count |
| `Dy` | target count |
| `C_o` | observed covariate count |
| `C_f` | known future covariate count |
| `K` | concept count |
| `P` | patch count |
| `R` | subspace rank |
| `S` | forecast sample count |
| `N_pair` | intervention pair count |
| `N_site` | localization candidate count |
| `E` | environment count |
| `Sha256Hash` | 64 lowercase hex（不带算法前缀） |
| `UtcDatetime` | timezone-aware UTC datetime |

Tensor 原则：

- scientific floating tensor 必须 finite；
- mask 必须 `torch.bool`；
- validator 不得隐式 clone、detach、cast、move device、改变 `requires_grad`；
- 如果缺失数据存在，用 mask 表达，不用 NaN 作为跨模块协议；
- device/dtype 必须被调用方显式控制，不由 scientific module 根据服务器资源自行决定。

---

# 5. 公共 Contract Class Registry

本节定义所有 Stage 允许使用的核心类。标题中的 `PROTOCOL` 表示结构化接口，不是 concrete implementation。

---

## 5.1 `ResearchContractManifest`

**用途**：Stage 0 的机器可校验输出，后续所有正式实验必须绑定。

```python
class ResearchContractManifest(StrictContractModel):
    schema_version: str
    protocol_id: str
    preregistration_ref: ArtifactRef
    novelty_claims_ref: ArtifactRef
    assumption_ledger_ref: ArtifactRef
    terminology_ref: ArtifactRef
    environment_lock_ref: ArtifactRef
    related_work_ref: ArtifactRef
    created_at: UtcDatetime
    status: Literal["FROZEN", "SUPERSEDED"]
```

**成员函数**：无科学成员函数。

**标准 helper**：

```python
validate_research_contract(x: ResearchContractManifest) -> ResearchContractManifest
research_contract_hash(x: ResearchContractManifest) -> Sha256Hash
```

---

## 5.1A 基础枚举与支持类型

### `StrictContractModel`

持久化 contract 的共同基类：

```python
class StrictContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )
```

它不实现科学逻辑，仅统一严格校验、冻结字段和拒绝额外字段。

### `SplitPartition`

```python
TRAIN
VALIDATION
TEST
```

用于持久化 metric/pair 的逻辑 partition。它与 `DatasetWindowPartition` 不同：后者可以进一步区分 `TEST_SEEN_REGIME` 与 `TEST_UNSEEN_REGIME`。

### `AccessScope`

```python
class AccessScope(StrictContractModel):
    sealed: bool
    scope_name: str
```

成员函数：无。
`sealed=True` 只表示访问请求涉及 sealed scope，不等价于授权；真正 sealed 访问仍需要 `SealedAccessGrant`。

### `SealedAccessGrant`

```python
class SealedAccessGrant(StrictContractModel):
    grant_id: str
    dataset: DatasetSpec
    scope_name: str
    allowed_partitions: tuple[DatasetWindowPartition, ...]
    authorization_ref: ArtifactRef
    issued_at: UtcDatetime
    expires_at: UtcDatetime
```

约束：

- `grant_id`、`scope_name` 必须非空；dataset 必须是精确的 logical `(name, version)`；
- `allowed_partitions` 必须非空且无重复；
- `authorization_ref.artifact_type` 必须为 `SEALED_ACCESS_AUTHORIZATION`；
- `expires_at` 必须晚于 `issued_at`；
- grant 必须与 dataset、scope 和请求 partition 全部精确匹配，并在访问时刻有效；
- 缺失、过期或不匹配时，必须在任何物理读取前 fail closed；
- registry 中 `sealed=True` 的 dataset 必须始终按 sealed 处理；调用方不得用 `AccessScope(sealed=False)` 将其降级为 unsealed；
- grant 只授权读取，不授权在 validation/test 上 fit，也不改变 scientific identity。

### `DatasetSourceKind`

```text
STAGE1_SYNTHETIC_CONFIG
PERSISTED_STAGE1
```

只能选择硬编码 allowlisted loader chain；不得包含 Python import path、shell command 或动态 callable。

### `RegimeRelation`

```text
SAME
CROSS
UNKNOWN
```

用于 intervention pair 的 base/source regime 关系。

### `InterventionKind`

```text
FULL_SWAP
SUBSPACE_SWAP
```

### `GateStatus`

```text
PASS
FAIL
BLOCKED
```

### `TaskState`

```text
PENDING
LEASED
RUNNING
PUBLISHING
COMPLETED
FAILED
```

---

## 5.1B `DataSplitSummary` 与 `WindowContractSummary`

```python
class DataSplitSummary(StrictContractModel):
    partition: SplitPartition
    split_hash: Sha256Hash
    count: int

class WindowContractSummary(StrictContractModel):
    history_length: int
    horizon: int
    input_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    observed_covariate_names: tuple[str, ...]
    known_future_covariate_names: tuple[str, ...]
    timezone: Literal["UTC"]
    missingness_protocol: str
```

两类均为数据 Manifest 的被动成员对象，无科学成员函数。

---

## 5.1C Stage 1B Physical Types

这些类定义 Stage 1B 的合成数据物理表示。它们可以在 Stage 1B 内部出现，但除本节列出的标准 bridge 外不得直接跨越模型/定位边界。

### `SyntheticConfig`

具体实现的精确 constructor 由 Stage 1 源码定义；跨阶段语义至少必须固定：

```python
class SyntheticConfig:
    name: str
    D: int
    L: int
    H: int
    regimes: int
    true_delay: int | tuple[int, ...]
    root_seed: int
    burn_in: int
    total_steps: int
    generation_settings: Mapping[str, JSONValue]
    normalization_settings: Mapping[str, JSONValue]
```

如果 concrete class 字段名与上述语义名不同，必须通过显式 adapter 映射；不得让下游读取 concrete implementation 私有字段。

### `SyntheticDataset`

```python
class SyntheticDataset:
    config: SyntheticConfig
    config_hash: Sha256Hash
    dataset_hash: Sha256Hash
    truth: Mapping[str, numpy.ndarray]
    normalization: NormalizationRecord
    splits: tuple[PhysicalSplit, ...]
    manifest: DataManifest
    provenance: SyntheticProvenance
```

它只在 Stage1B data implementation 内部使用；普通 predictor 不得接收它。

### `PersistedSyntheticDataset`

```python
class PersistedSyntheticDataset:
    output_root: PathLike
    dataset_hash: Sha256Hash
    files: Mapping[str, PathLike]
    checksums: Mapping[str, Sha256Hash]
```

### `Stage2Splits`

```python
class Stage2Splits:
    manifest: DataManifest
    data_hash: Sha256Hash
    train: WindowBatch
    validation: WindowBatch
    test_seen_regime: WindowBatch
    test_unseen_regime: WindowBatch
```

它是 Stage 1→Stage 2 的标准 bridge。Architecture-facing path 仍优先使用 `DatasetSpec + DatasetWindowPartition -> build_windows()`。

---

## 5.1D `RunManifest`

```python
class RunManifest(StrictContractModel):
    experiment_id: str
    run_id: str
    config_hash: Sha256Hash
    data_hash: Sha256Hash
    git_commit: str
    schema_version: str
    created_at: UtcDatetime
    status: str
```

`git_commit` 必须绑定实际代码 revision；run id 不能替代 scientific identity。

---

## 5.1E `ExperimentSummary`

```python
class ExperimentSummary:
    experiment_id: str
    results: tuple[TaskResult, ...]
```

成员函数：无。
任何 aggregate metric 必须通过 `MetricRecord` 单独发布，不能藏在未定义的 summary dict 中。

---

## 5.1F `ResourceSnapshot`

```python
class ResourceSnapshot(StrictContractModel):
    snapshot_id: str
    cpu_threads_available: int
    host_memory_gib_available: float
    gpu_count: int
    gpu_memory_gib: tuple[float, ...]
    backend_id: str
    captured_at: UtcDatetime
    snapshot_hash: Sha256Hash
```

只含运行资源事实；不得包含 scientific metric、target、model rank 或 sealed truth。

---

## 5.1G Stage 0 运行核验报告

以下 strict/frozen/extra-forbid 对象是 Stage 0 公共检查函数的标准输出，只报告运行核验结果，不是科学证据或计划实施状态：

```python
class DoctorCheckResults(StrictContractModel):
    pot_sinkhorn: Literal["PASS"] | None
    python_version: Literal["PASS"] | None
    pyvene_import: Literal["PASS"] | None
    torch_basic: Literal["PASS"] | None
    torch_hook: Literal["PASS"] | None
    workspace_disk: Literal["PASS"] | None
    workspace_write: Literal["PASS"] | None

class DoctorVersions(StrictContractModel):
    numpy: str | None
    pot: str | None
    pyvene: str | None
    torch: str | None

class DoctorResources(StrictContractModel):
    logical_cpu_count: int
    memory_total_bytes: int | None
    disk_total_bytes: int
    disk_free_bytes: int
    python_version: str
    python_executable: str

class DoctorReport(StrictContractModel):
    status: Literal["PASS", "FAIL"]
    gpu_required: Literal[False]
    checks: DoctorCheckResults
    versions: DoctorVersions
    resources: DoctorResources | None
    cuda_available: bool | None
    cuda_device_count: int | None
    default_profile_id: str | None
    execution_backend_replaceable: Literal[True] | None
    tested_torch_dtypes: tuple[Literal["float32", "float64"], ...]
    error: str | None

class Stage0VerificationReport(StrictContractModel):
    status: Literal["PASS"]
    row_count: int
    unique_work_ids: int
    dependency_release_count: int
    locked_dependency_count: int
    source_count: int
    research_contract_status: Literal["FROZEN"]
    gate0_status: Literal["PASS"]
    completion_status: Literal["COMPLETED"]
    doctor: DoctorReport | None
```

`DoctorReport` 在 `PASS` 时必须包含完整 capability 字段且所有检查为 `PASS`；在 `FAIL` 时必须包含非空 `error`。默认环境只是可替换的执行起点，以上报告不得固定服务器或算力边界。

---

## 5.2 `ArtifactRef`

```python
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    content_hash: Sha256Hash
    schema_version: str
    relative_path: str | None
```

**纯函数成员/属性**：

```python
identity_key() -> tuple[str, str, str]
# (artifact_type, content_hash, schema_version)
```

`relative_path` 不属于内容身份。

---

## 5.3 `ArtifactManifest`

```python
class ArtifactManifest(StrictContractModel):
    artifact: ArtifactRef
    media_type: str
    serializer_id: str
    producer_stage: str
    producer_task_id: str
    scientific_identity_hash: Sha256Hash
    dependencies: tuple[ArtifactRef, ...]
    size_bytes: int
    created_at: UtcDatetime
```

不得包含任意 Python import path 或可执行命令。

---

## 5.4 `DatasetSpec`

```python
class DatasetSpec(StrictContractModel):
    name: str
    version: str
```

约束：logical key，不是路径；禁止 `/`, `\\`, drive prefix, NUL, `.`/`..` path semantics。

---

## 5.5 `DatasetWindowPartition`

```python
class DatasetWindowPartition(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    TEST_SEEN_REGIME = "TEST_SEEN_REGIME"
    TEST_UNSEEN_REGIME = "TEST_UNSEEN_REGIME"
```

它表示**已有物理窗口视图**，不能代替 temporal split policy。

---

## 5.6 `DatasetRegistryEntry` / `DatasetRegistryManifest`

```python
class DatasetRegistryEntry(StrictContractModel):
    dataset: DatasetSpec
    source_kind: DatasetSourceKind
    relative_location: str
    expected_dataset_hash: Sha256Hash
    sealed: bool
    available_partitions: tuple[DatasetWindowPartition, ...]

class DatasetRegistryManifest(StrictContractModel):
    registry_id: str
    registry_version: str
    entries: tuple[DatasetRegistryEntry, ...]
```

---

## 5.7 `WindowBatch`

```python
@dataclass(frozen=True, slots=True)
class WindowBatch:
    x: Tensor                              # [B, L, D]
    y: Tensor | None                       # [B, H, Dy]
    observed_covariates: Tensor | None     # [B, L, C_o]
    known_future_covariates: Tensor | None # [B, H, C_f]
    x_observed_mask: Tensor | None
    y_observed_mask: Tensor | None
    observed_covariates_mask: Tensor | None
    known_future_covariates_mask: Tensor | None
    regime: Tensor | None                  # [B]
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

**类方法**：无科学方法。所有校验由 `validate_window_batch()` 完成。

---

## 5.8 `DataManifest`

必须至少绑定：

```python
class DataManifest(StrictContractModel):
    schema_version: str
    dataset_name: str
    dataset_version: str
    dataset_hash: Sha256Hash
    splits: tuple[DataSplitSummary, ...]
    window_contract: WindowContractSummary
    source_description: str
    created_at: UtcDatetime
```

---

## 5.9 `SCMTruthManifest`

**用途**：Stage 1B 将合成世界的“真值”以持久化 artifact 形式交给机制植入、Gate 和评估，但不把 truth tensor 偷塞进普通 `WindowBatch`。

```python
class SCMTruthManifest(StrictContractModel):
    schema_version: str
    dataset_hash: Sha256Hash
    generator_config_hash: Sha256Hash
    concept_names: tuple[str, ...]
    regime_ids: tuple[str, ...]
    true_lags: Mapping[str, tuple[int, ...]]
    true_graph_ref: ArtifactRef
    latent_concepts_ref: ArtifactRef
    regime_sequence_ref: ArtifactRef
    exogenous_noise_ref: ArtifactRef
    shock_sequence_ref: ArtifactRef | None
    oracle_protocol_hash: Sha256Hash
    sealed: bool
```

只有 synthetic/oracle evaluation 路径可以解析这些 truth artifacts。

---

## 5.10 `ForecastDistribution`

```python
@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    mean: Tensor                    # [B, H, Dy]
    scale: Tensor | None            # [B, H, Dy], > 0
    quantiles: Mapping[float, Tensor]
    logits: Tensor | None           # [B, H, Dy, C]
    samples: Tensor | None          # [S, B, H, Dy]
    window_id: tuple[str, ...] | None
    target_names: tuple[str, ...]
```

校验：finite、shape/device/dtype aligned、quantile level ∈ (0,1)、quantile non-crossing。

---

## 5.11 `ForecastPredictor` — PROTOCOL

```python
class ForecastPredictor(Protocol):
    @property
    def adapter_name(self) -> str: ...

    @property
    def model_hash(self) -> str: ...

    @property
    def is_frozen(self) -> bool: ...

    def predict_distribution(
        self,
        batch: WindowBatch,
    ) -> ForecastDistribution: ...
```

建议通过行为测试验证：输入不被修改，输出 shape/window_id/target_names 对齐。

---

## 5.12 `InterventionSite`

```python
@dataclass(frozen=True, slots=True)
class InterventionSite:
    site_name: str
    layer: int | None
    tensor_rank: int
    batch_axis: int
    variable_axis: int | None
    patch_axis: int | None
    feature_axis: int
    shape_template: tuple[int | None, ...]
```

补充 v2 语义：

- `site_name` 是稳定主键；
- `variable_axis`/`patch_axis` 可以为空，但不能伪造；
- PatchTST adapter 可声明 patch axis；channel-independent 不能自动推出跨变量交互；
- iTransformer adapter 应把 variate token 对应的 variable axis 显式声明；
- 模型-specific 内部 tensor 先转换为 adapter 的 canonical view，再暴露 axis。

---

## 5.13 `MechanisticModelAdapter` — PROTOCOL

```python
class MechanisticModelAdapter(Protocol):
    @property
    def adapter_name(self) -> str: ...
    @property
    def model_hash(self) -> str: ...
    @property
    def is_frozen(self) -> bool: ...

    def predict_distribution(
        self, batch: WindowBatch
    ) -> ForecastDistribution: ...

    def list_intervention_sites(
        self,
    ) -> tuple[InterventionSite, ...]: ...

    def capture(
        self,
        batch: WindowBatch,
        sites: tuple[InterventionSite, ...],
    ) -> Mapping[str, Tensor]: ...

    def intervene(
        self,
        base: WindowBatch,
        source: WindowBatch,
        spec: InterventionSpec,
    ) -> ForecastDistribution: ...
```

普通 `ForecastPredictor` 不得自动视为 `MechanisticModelAdapter`。

---

## 5.14 `ConceptSpec`

```python
class ConceptSpec(StrictContractModel):
    name: str
    definition_version: str
    required_history: int
    history_only: bool
    source_kind: Literal["ANALYTIC", "WEAK_SUPERVISION", "CONSTRAINED_LEARNED", "SYNTHETIC_TRUTH"]
    intervention_semantics: str
    valid_range: tuple[float | None, float | None]
    expected_effect_components: tuple[str, ...]
    definition_hash: Sha256Hash
```

---

## 5.15 `ConceptBatch`

```python
@dataclass(frozen=True, slots=True)
class ConceptBatch:
    values: Tensor             # [B, K]
    valid_mask: Tensor         # bool [B, K]
    names: tuple[str, ...]
    window_id: tuple[str, ...]
    computed_from_history_only: bool
    definition_version: str
```

---

## 5.16 `ConceptExtractor` — PROTOCOL

```python
class ConceptExtractor(Protocol):
    def compute(self, batch: WindowBatch) -> ConceptBatch: ...
    def leakage_audit(self, batch: WindowBatch) -> LeakageAudit: ...
```

---

## 5.17 `ConceptIntervention`

```python
class ConceptIntervention:
    concept_name: str
    delta: float
```

对于 source-value swap，`delta` 可由 base/source concept value 派生，但必须记录最终值。

---

## 5.18 `PairingSpec`

```python
class PairingSpec(StrictContractModel):
    spec_id: str
    partition: SplitPartition
    concept_name: str
    regime_relation: RegimeRelation
    min_concept_delta: float
    distance_metric: Literal["STRATIFIED_RANDOM", "EUCLIDEAN", "MAHALANOBIS", "LOCAL_OT"]
    matching_feature_names: tuple[str, ...]
    max_source_reuse: int
    max_time_overlap: int
    seed: int
    spec_hash: Sha256Hash
```

---

## 5.19 `InterventionPair`

```python
class InterventionPair(StrictContractModel):
    schema_version: str
    pair_id: str
    partition: SplitPartition
    base_window_id: str
    source_window_id: str
    concept_name: str
    regime_relation: RegimeRelation
    matching_distance: float
    concept_delta: float
```

`base_window_id != source_window_id`；distance finite and non-negative。

---

## 5.20 `InterventionPairSet`

```python
class InterventionPairSet:
    pair_ids: tuple[str, ...]
    source_label: str
```

v2 要求其所有 ID 必须通过 `PairRegistry` 解析，不得按窗口位置隐式配对。

---

## 5.21 `ResolvedInterventionPairBatch`

```python
@dataclass(frozen=True, slots=True)
class ResolvedInterventionPairBatch:
    pairs: tuple[InterventionPair, ...]
    base: WindowBatch
    source: WindowBatch
    base_row_for_pair: tuple[int, ...]
    source_row_for_pair: tuple[int, ...]
    dataset_hash: Sha256Hash
```

约束：

- `len(pairs) == len(base_row_for_pair) == len(source_row_for_pair)`；
- 每个索引能恢复 pair 的 base/source window_id；
- pair partition 与物理 WindowBatch partition 一致；
- train/validation/test pair 不共享 window。

---

## 5.22 `InterventionSpec`

```python
@dataclass(frozen=True, slots=True)
class InterventionSpec:
    site_name: str
    layer: int | None
    variable_index: int | None
    patch_index: int | None
    lag: int
    subspace_basis: Tensor | None
    intervention_kind: InterventionKind
```

`SUBSPACE_SWAP` 必须携带有限、二维、正交基；`FULL_SWAP` 不携带 basis。

---

## 5.23 `InterventionResult`

```python
class InterventionResult:
    pair_id: str
    spec: InterventionSpec
    factual: ForecastDistribution
    intervened: ForecastDistribution
```

这是**低层神经干预结果**。

---

## 5.24 `HighLevelInterventionResult`

**用于把 SCM/高层概念干预与低层神经干预放到同一 effect pipeline。**

```python
@dataclass(frozen=True, slots=True)
class HighLevelInterventionResult:
    pair_id: str
    concept_name: str
    factual: ForecastDistribution
    intervened: ForecastDistribution
    oracle: bool
    expected_lag: int | None
    high_level_model_hash: Sha256Hash
```

---

## 5.25 `EffectSignature`

```python
@dataclass(frozen=True, slots=True)
class EffectSignature:
    delta_mean: Tensor
    delta_scale: Tensor | None
    delta_quantiles: Mapping[float, Tensor]
    horizon: int
```

shape 至少保留 `[H, Dy]` 或 `[B,H,Dy]` 的协议化形式；具体 aggregation 必须由 `EffectComputationSpec` 决定。

---

## 5.26 `EffectComputationSpec`

```python
class EffectComputationSpec(StrictContractModel):
    spec_id: str
    aggregation: Literal["PER_PAIR", "MEAN_OVER_PAIRS"]
    include_mean: bool
    include_scale: bool
    quantile_levels: tuple[float, ...]
    horizon_weights: tuple[float, ...] | None
    spec_hash: Sha256Hash
```

禁止把单样本 calibration error 放入 effect signature。

---

## 5.27 `EffectRecord`

```python
@dataclass(frozen=True, slots=True)
class EffectRecord:
    pair_id: str
    concept_name: str
    source_kind: Literal["HIGH_LEVEL", "LOW_LEVEL"]
    model_id: str | None
    candidate_id: str | None
    signature: EffectSignature
    effect_spec_hash: Sha256Hash
```

高层 effect：`model_id=None`, `candidate_id=None`。
低层 effect：必须绑定 model_id/candidate_id。

---

## 5.28 `EffectNormalizationSpec`

```python
class EffectNormalizationSpec:
    train_only: bool   # 必须 True
```

## 5.29 `EffectNormalizerState`

```python
class EffectNormalizerState(StrictContractModel):
    normalizer_id: str
    method: Literal["ROBUST_SCALE", "STD", "MAD", "IDENTITY"]
    train_pair_set_hash: Sha256Hash
    parameters_ref: ArtifactRef
    effect_spec_hash: Sha256Hash
    frozen: bool
```

validation/test 只能 `transform`，不能 fit。

---

## 5.30 `MechanismTruthSite`

```python
class MechanismTruthSite(StrictContractModel):
    concept_name: str
    site_name: str
    layer: int | None
    variable_indices: tuple[int, ...]
    patch_indices: tuple[int, ...]
    causal_lags: tuple[int, ...]
    subspace_rank: int
    basis_ref: ArtifactRef
    forecast_horizon_support: tuple[int, ...]
```

注意：forecast horizon 是输出效应索引，不与 causal lag 合并。

---

## 5.31 `PlantedMechanismManifest`

```python
class PlantedMechanismManifest(StrictContractModel):
    manifest_id: str
    model_id: str
    model_hash: Sha256Hash
    checkpoint_hash: Sha256Hash
    dataset_hash: Sha256Hash
    planting_method: Literal["HARD_PLANT", "POST_TRAIN_LOW_RANK_ADAPTER"]
    truth_sites: tuple[MechanismTruthSite, ...]
    concept_definition_hashes: Mapping[str, Sha256Hash]
    manifest_hash: Sha256Hash
```

---

## 5.32 `LocalizationStage`

```text
COARSE_LAYER
TIME_PATCH
VARIABLE
SUBSPACE
DAS_REFINEMENT
HELDOUT_EVAL
```

---

## 5.33 `LocalizationCandidate`

```python
class LocalizationCandidate(StrictContractModel):
    candidate_id: str
    stage: LocalizationStage
    site_name: str
    layer: int | None
    variable_index: int | None
    patch_index: int | None
    lag: int | None
    subspace_rank: int | None
    basis_ref: ArtifactRef | None
    parent_candidate_id: str | None
```

candidate 是“假设位置”，不是实验结果。

---

## 5.34 `OTProblem`

```python
@dataclass(frozen=True, slots=True)
class OTProblem:
    problem_id: str
    high_level_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    source_weights: Tensor          # [K]
    target_weights: Tensor          # [N_site]
    cost_matrix: Tensor             # [K, N_site]
    solver_kind: Literal["BALANCED_SINKHORN", "UNBALANCED_SINKHORN"]
    epsilon: float
    unbalanced_reg: float | None
```

---

## 5.35 `OTResult`

```python
@dataclass(frozen=True, slots=True)
class OTResult:
    problem_id: str
    transport_plan: Tensor          # [K, N_site]
    objective_value: float
    converged: bool
    iterations: int | None
    solver_backend: str
    diagnostics: Mapping[str, float | int | str | bool]
```

必须验证 transport mass、finite、shape 与 problem 匹配。

---

## 5.36 `LocalizationRequest`

```python
class LocalizationRequest(StrictContractModel):
    request_id: str
    stage: LocalizationStage
    concept_names: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    high_level_effect_ref: ArtifactRef
    low_level_effect_ref: ArtifactRef
    normalizer_ref: ArtifactRef
    parent_localization_ref: ArtifactRef | None
    localization_config_hash: Sha256Hash
```

---

## 5.37 `LocalizationTrace`

```python
class LocalizationTrace(StrictContractModel):
    trace_id: str
    stage: LocalizationStage
    request_hash: Sha256Hash
    candidate_ids_in: tuple[str, ...]
    ot_problem_ref: ArtifactRef | None
    ot_result_ref: ArtifactRef | None
    selected_candidate_ids: tuple[str, ...]
    selection_rule: str
    selection_threshold: float | None
    cost_summary: Mapping[str, float]
    runtime_seconds: float
```

---

## 5.38 `LocalizationResult`

```python
class LocalizationResult:
    stage: LocalizationStage
    candidate_ids: tuple[str, ...]
```

v2 规定：它只是轻量运行时结果。Claim-bearing 持久化结果必须由 `LocalizationTrace` 保存完整 evidence。

---

## 5.39 `DASRefinementSpec`

```python
class DASRefinementSpec(StrictContractModel):
    spec_id: str
    candidate_ids: tuple[str, ...]
    rank_grid: tuple[int, ...]
    train_pair_set_hash: Sha256Hash
    validation_pair_set_hash: Sha256Hash
    heldout_pair_set_hash: Sha256Hash
    capacity_penalty: float
    orthogonality_tolerance: float
    optimization_hash: Sha256Hash
```

---

## 5.40 `EnvironmentSpec`

```python
class EnvironmentSpec:
    environment_id: str
    definition_hash: Sha256Hash
```

---

## 5.41 `EnvironmentAssignmentBatch`

```python
@dataclass(frozen=True, slots=True)
class EnvironmentAssignmentBatch:
    pair_ids: tuple[str, ...]
    environment_ids: tuple[str, ...]
    features: Tensor               # [N_pair, E_feat]
    feature_names: tuple[str, ...]
    definition_hash: Sha256Hash
    fitted_on_partition: SplitPartition
```

真实数据环境定义若需要 HMM/threshold，必须 train-fit 后再生成 val/test assignment。

---

## 5.42 `RobustnessSpec`

至少包含：

```python
class RobustnessSpec:
    train: EnvironmentSpec
    validation: EnvironmentSpec
    test: EnvironmentSpec
```

额外算法配置放 `RobustnessTrainingSpec`。

---

## 5.43 `RobustnessTrainingSpec`

```python
class RobustnessTrainingSpec(StrictContractModel):
    spec_id: str
    method: Literal["ERM", "BALANCED", "GROUP_DRO", "WASSERSTEIN_DRO"]
    radius_grid: tuple[float, ...]
    cost_feature_names: tuple[str, ...]
    zero_refit_required: bool
    validation_selection_metric: str
    spec_hash: Sha256Hash
```

---

## 5.44 `RobustnessModelState`

```python
class RobustnessModelState(StrictContractModel):
    state_id: str
    method: str
    training_spec_hash: Sha256Hash
    train_environment_hash: Sha256Hash
    normalizer_ref: ArtifactRef
    localization_ref: ArtifactRef
    parameter_ref: ArtifactRef
    selected_radius: float | None
    cost_scaler_ref: ArtifactRef | None
    frozen: bool
    zero_refit: bool
```

---

## 5.45 `RobustnessEvaluation`

```python
class RobustnessEvaluation(StrictContractModel):
    evaluation_id: str
    state_ref: ArtifactRef
    partition: SplitPartition
    environment_metric_refs: tuple[ArtifactRef, ...]
    mean_abstraction_error: float
    worst_environment_error: float
    mean_prediction_metric: float | None
    zero_refit_verified: bool
```

---

## 5.46 `MetricContext`

```python
class MetricContext(StrictContractModel):
    experiment_id: str
    run_id: str
    split: SplitPartition
    data_hash: Sha256Hash
    model_id: str | None
    protocol_id: str
    gate_scope: str | None
```

---

## 5.47 `MetricRecord`

```python
class MetricRecord(StrictContractModel):
    experiment_id: str
    run_id: str
    split: SplitPartition
    metric_name: str
    value: float
    regime: str | int | None
    horizon: int | None
    concept: str | None
```

所有值 finite。

---

## 5.48 `GatePredicate`

```python
class GatePredicate(StrictContractModel):
    predicate_id: str
    metric_name: str
    scope: Mapping[str, str | int | float | bool]
    operator: Literal[">", ">=", "<", "<=", "=="]
    threshold: float
    aggregation: Literal["ALL", "MEAN", "WORST", "MEDIAN"]
```

正式实验前 threshold 不允许仍是“明显”“较高”“稳定”等自然语言。

---

## 5.49 `GateSpec`

```python
class GateSpec(StrictContractModel):
    gate_id: str
    protocol_id: str
    gate_version: str
    predicates: tuple[GatePredicate, ...]
    required_evidence_types: tuple[str, ...]
    failure_action: Literal["REPAIR", "STOP", "DROP_CLAIM", "CONTINUE_EXPLORATORY"]
    spec_hash: Sha256Hash
```

---

## 5.50 `GateDecision`

```python
class GateDecision:
    gate_id: str
    status: GateStatus      # PASS / FAIL / BLOCKED
    rationale: str
    evidence: tuple[ArtifactRef, ...]
```

除下述人工新颖性例外外，GateDecision 必须绑定同一版本 `GateSpec` 的 hash（通过 evidence/receipt）。`GATE_0_NOVELTY 是人工新颖性 Gate，不要求 GateSpec`；它直接绑定同一冻结版本的 novelty claims 与 related-work bundle，仓库仅核验决定结构、状态、证据类型和 content hash。

---

## 5.51 `LeakageAudit`

```python
class LeakageAudit:
    passed: bool
    findings: tuple[str, ...]
```

正式金融/真实数据实验必须额外持久化 audit context：data hash、fit interval、release-time alignment hash、pair separation evidence。

---

## 5.52 `StatisticalTestSpec`

```python
class StatisticalTestSpec(StrictContractModel):
    test_id: str
    metric_name: str
    test_kind: Literal["BLOCK_BOOTSTRAP", "PAIRED_BOOTSTRAP", "DIEBOLD_MARIANO"]
    block_length: int | None
    confidence_level: float
    correction: Literal["NONE", "HOLM", "BENJAMINI_HOCHBERG"]
    seed: int
```

---

## 5.53 `StatisticalTestResult`

```python
class StatisticalTestResult(StrictContractModel):
    test_id: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float | None
    adjusted_p_value: float | None
    effect_size: float | None
    sample_count: int
```

---

## 5.54 `EfficiencyRecord`

```python
class EfficiencyRecord(StrictContractModel):
    experiment_id: str
    run_id: str
    method: str
    gpu_hours: float
    cpu_hours: float
    intervention_count: int
    peak_gpu_memory_gib: float | None
    peak_host_memory_gib: float
    cache_bytes: int
```

---

## 5.55 `AblationSpec`

```python
class AblationSpec(StrictContractModel):
    ablation_id: str
    parent_experiment_id: str
    changed_factors: Mapping[str, str | int | float | bool]
    unchanged_identity_hash: Sha256Hash
    seed_policy: str
```

任何消融必须明确“改了什么、没改什么”。

---

## 5.56 `ScientificIdentity`

```python
class ScientificIdentity:
    protocol_id: str
    experiment_id: str
    task_id: str
    model_id: str
    data_id: str
    seed: int
```

---

## 5.57 `ResourceRequest`

```python
class ResourceRequest:
    cpu_threads: int
    gpu_count: int
    gpu_memory_gib: float
    host_memory_gib: float
```

它不能包含模型/数据选择逻辑。

---

## 5.58 `TaskSpec` / `TaskManifest`

```python
class TaskSpec:
    identity: ScientificIdentity
    phase: str
    inputs: tuple[ArtifactRef, ...]
    output_artifact_type: str
    resource_request: ResourceRequest

class TaskManifest:
    manifest_id: str
    tasks: tuple[TaskSpec, ...]
    completed_task_policy: Literal["NEVER_RERUN"]
```

---

## 5.59 `ResourceAllocation`

```python
class ResourceAllocation(StrictContractModel):
    cpu_threads: int
    gpu_ids: tuple[int, ...]
    host_memory_gib_limit: float
    worker_id: str
```

GPU id 不是 scientific identity。

---

## 5.60 `PlannedTask`

```python
class PlannedTask(StrictContractModel):
    task_id: str
    attempt_id: str
    executor_key: str
    allocation: ResourceAllocation
    input_refs: tuple[ArtifactRef, ...]
    expected_output_artifact_type: str
```

`executor_key` 必须来自 allowlisted runtime registry；禁止任意 shell string。

---

## 5.61 `ExecutionPlan`

```python
class ExecutionPlan(StrictContractModel):
    plan_id: str
    task_manifest_id: str
    backend_id: str
    planned_tasks: tuple[PlannedTask, ...]
    max_concurrency: int
    resource_snapshot_hash: Sha256Hash
    created_at: UtcDatetime
```

---

## 5.62 `ExecutionContext`

```python
class ExecutionContext:
    run_id: str
    task_id: str
    attempt_id: str
    runtime_identity: str
    worker_identity: str
```

不允许包含替代 scientific identity 的字段。

---

## 5.63 `TaskResult`

```python
class TaskResult:
    task_id: str
    attempt_id: str
    state: TaskState
    artifact: ArtifactRef | None
```

`COMPLETED` 必须有经过验证的 ArtifactRef。

---

## 5.64 `MonitoringSnapshot`

```python
class MonitoringSnapshot:
    phase: str
    terminal_status: str | None
    task_counts: Mapping[str, int]
    resource_summary: Mapping[str, float]
    heartbeat_age_seconds: float
    eta_status: str
```

禁止包含目标值、partial metrics、模型排名、sealed truth。

---

## 5.65 `ReleaseManifest`

```python
class ReleaseManifest(StrictContractModel):
    release_id: str
    protocol_id: str
    git_commit: str
    environment_lock_ref: ArtifactRef
    research_contract_ref: ArtifactRef
    gate_decisions: tuple[ArtifactRef, ...]
    datasets: tuple[ArtifactRef, ...]
    models: tuple[ArtifactRef, ...]
    experiment_summaries: tuple[ArtifactRef, ...]
    statistics: tuple[ArtifactRef, ...]
    reproduction_levels: Mapping[str, str]
    created_at: UtcDatetime
```

---

# 6. 标准 Artifact Store 协议

Artifact Store 不得使用接收任意 `object` 的发布接口；应采用以下 typed methods。

## 6.1 `ArtifactStore` — PROTOCOL / v2

```python
T = TypeVar("T")

class ArtifactStore(Protocol):
    def publish_contract(
        self,
        value: StrictContractModel,
        artifact_type: str,
    ) -> ArtifactRef: ...

    def publish_arrow(
        self,
        table: pyarrow.Table,
        expected_schema: pyarrow.Schema,
        artifact_type: str,
    ) -> ArtifactRef: ...

    def publish_bytes(
        self,
        value: bytes,
        artifact_type: str,
        media_type: str,
        schema_version: str,
    ) -> ArtifactRef: ...

    def publish_text(
        self,
        value: str,
        artifact_type: str,
        media_type: str,
        schema_version: str,
    ) -> ArtifactRef: ...

    def load_contract(
        self,
        ref: ArtifactRef,
        expected_type: type[T],
    ) -> T: ...

    def load_arrow(
        self,
        ref: ArtifactRef,
        expected_schema: pyarrow.Schema,
    ) -> pyarrow.Table: ...

    def load_bytes(self, ref: ArtifactRef) -> bytes: ...

    def verify_artifact(self, ref: ArtifactRef) -> bool: ...
```

## 6.2 原子发布顺序

```text
temporary write
-> flush/fsync (适用时)
-> canonical serialization
-> content hash
-> reload
-> schema/type validation
-> atomic rename/publish
-> manifest/completion marker
-> ArtifactRef
```

如果 reload/schema/hash 任一失败，不得返回 ArtifactRef。

---

# 7. Pair Registry 与解析协议

## 7.1 `PairRegistry` — PROTOCOL

```python
class PairRegistry(Protocol):
    def resolve(self, pair_id: str) -> InterventionPair: ...

    def resolve_set(
        self,
        pair_set: InterventionPairSet,
        dataset: DatasetSpec,
        partition: DatasetWindowPartition,
        access: AccessScope,
    ) -> ResolvedInterventionPairBatch: ...
```

硬规则：

- 不允许 `pair_ids[i]` 默认对应 `WindowBatch[i]`；
- 不允许从文件名猜 base/source；
- pair record、dataset hash、partition、window IDs 必须交叉验证；
- sealed 数据仍受 grant 管理。

---

# 8. Effect Bridge 协议

## 8.1 必须实现的函数

```python
build_effect_signature(
    factual: ForecastDistribution,
    intervened: ForecastDistribution,
    spec: EffectComputationSpec,
) -> EffectSignature
```

功能：只计算预测分布变化：

```text
delta_mean
delta_scale
delta_quantiles
```

不得加入单样本 calibration。

```python
build_high_level_effect_record(
    result: HighLevelInterventionResult,
    spec: EffectComputationSpec,
) -> EffectRecord
```

```python
build_low_level_effect_record(
    result: InterventionResult,
    concept_name: str,
    model_id: str,
    candidate_id: str,
    spec: EffectComputationSpec,
) -> EffectRecord
```

标准链路为：

```text
InterventionResult
  -> build_low_level_effect_record
  -> EffectRecord(signature=EffectSignature)
```

---

# 9. Metrics 与 Gate 标准协议

## 9.1 Metrics

所有 metric functions 必须显式接收 `MetricContext`：

```python
forecasting_metrics(
    forecast: ForecastDistribution,
    target: WindowBatch,
    context: MetricContext,
) -> tuple[MetricRecord, ...]
```

```python
abstraction_metrics(
    high_level: Sequence[EffectRecord],
    low_level: Sequence[EffectRecord],
    context: MetricContext,
) -> tuple[MetricRecord, ...]
```

```python
localization_metrics(
    localization: LocalizationTrace,
    truth: PlantedMechanismManifest,
    context: MetricContext,
) -> tuple[MetricRecord, ...]
```

```python
calibration_metrics(
    forecast: ForecastDistribution,
    target: WindowBatch,
    context: MetricContext,
) -> tuple[MetricRecord, ...]
```

calibration 只允许按 fold/horizon/subgroup/regime 聚合。

## 9.2 Gate

```python
evaluate_gate(
    spec: GateSpec,
    records: Sequence[MetricRecord],
    evidence: Sequence[ArtifactRef],
) -> GateDecision
```

```python
verify_gate(
    spec: GateSpec,
    decision: GateDecision,
) -> None
```

必须验证：

- gate id/version/hash 对齐；
- predicate 使用的 metric 都存在；
- evidence type 完整；
- threshold 已数值冻结；
- 不得从 test 后验修改 GateSpec。

---

# 10. OT Backend 标准协议

```python
class OTBackend(Protocol):
    def solve(self, problem: OTProblem) -> OTResult: ...
```

TARCA 上层永远只看 `OTProblem/OTResult`。

允许 backend：

- POT balanced Sinkhorn；
- POT UOT；
- 后续经 CCP 批准的 GPU/自定义 solver。

更换 backend 只有在：

- 数值容差内等价；
- solver semantics 不变；
- problem identity 不变；

时可视为执行优化；否则必须新 scientific identity。

---

# 11. Stage 总览与标准交接矩阵

| Stage | 名称 | 核心输入 | 核心输出 | 下一消费者 |
|---|---|---|---|---|
| 0 | 研究契约与新颖性冻结 | 计划书/文献/环境 | `ResearchContractManifest`, Gate0 | 所有 Stage |
| 1A | 数据契约与数据边界 | Dataset registry + contract | `DataManifest`, `WindowBatch`, dataset hash | 1B/2/10/11 |
| 1B | 合成 SCM 与反事实真值 | Synthetic config + 1A | `SCMTruthManifest`, high-level oracle effects | 3/4/5/9 |
| 2 | 基础预测器 | `WindowBatch`, model registry | `ForecastDistribution`, frozen predictor identity | 3/4/10/11 |
| 3 | 机制植入与 mechanistic adapter | Stage1 truth + Stage2 model | `PlantedMechanismManifest`, sites, mechanistic model | 4/6/7/8 |
| 4 | 固定位置交换干预 | pair batch + adapter + spec | `InterventionResult` | 5 |
| 5 | Matching / Effect / Gate A | high/low intervention results | `EffectRecord`, normalizer, metrics, Gate A | 6 |
| 6 | 层 × 时间/lag 渐进定位 | normalized effects + candidates | `LocalizationTrace` | 7 |
| 7 | 变量轴与子空间 | Stage6 trace + adapter | candidate/subspace trace | 8 |
| 8 | DAS 精修与 Gate1 | Stage7 + pair splits | final localization + Gate1 | 9/10/11 |
| 9 | Regime robustness + theory | localization + effects + env | `RobustnessModelState`, evaluation, Gate2 | 10/11 |
| 10 | 通用时间序列验证 | generic datasets + frozen method state | domain summaries/metrics | cross-domain check / 11 |
| 11 | 金融压力测试 | finance data + leakage audit + frozen pipeline | finance summary/Gate3 input | 12 |
| 12 | 消融、统计与效率 | all metrics/summaries | stats/ablations/Gate4 candidate | 13 |
| 13 | 复现发布 | all passed/recorded evidence | `ReleaseManifest` | paper/release |

---

# 12. Stage 0：研究契约、环境与 Gate 0

## 12.1 标准输入

- 项目计划书；
- 具体实施计划；
- 本协议；
- Stage 0 检索得到的一手论文、官方资料与官方仓库证据；
- 本地 Python、Conda、操作系统和硬件环境事实。

## 12.2 标准输出

1. `ResearchContractManifest`
2. `GateDecision(gate_id="GATE_0_NOVELTY")`
3. `Stage0CompletionReceipt`：在研究合同、Gate 0 与 artifact index 全部通过后发布，分别绑定三者的 `ArtifactRef`；
4. 以下 `ArtifactRef`：
   - preregistration；
   - novelty claims；
   - assumption ledger；
   - terminology；
   - environment lock：绑定 `pyproject.toml`、`uv.lock` 与一个默认 runtime environment profile 的 bundle；该 profile 是可复现起点，不是固定算力边界，用户可授权改用服务器或其他 execution backend；
   - third-party versions：绑定官方仓库 commit；依赖项另绑定 package version、release tag 与 release commit。

## 12.3 必要函数

```python
freeze_research_contract(...) -> ResearchContractManifest
verify_research_contract(...) -> None
run_doctor(workspace: PathLike) -> DoctorReport
verify_stage0(...) -> Stage0VerificationReport
```

`GateDecision(GATE_0_NOVELTY)` 由经人工核验并授权的外部决策流程签发。仓库只验证其 schema、`PASS/FAIL/BLOCKED`、所需 evidence 类型与 content hash，不实现自动新颖性判断器。

## 12.4 不变量

- 不能以 README 代替 preregistration；
- PLOT-guided DAS 明确作为 baseline；
- 一般 Wasserstein causal abstraction 不能作为 TARCA 新颖性；
- 金融只是压力测试，不能关闭 Gate A/1/2；
- 若明确的直接碰撞证据进入项目并覆盖核心 claim，先更新 novelty claim，再继续对应代码；协议不要求例行重跑文献检索。
- 研究合同与 Stage 0 artifact 默认拒绝覆盖；只有用户显式授权并给出理由时才可替换活动版本，且必须先归档旧版本并生成前后 hash 审计回执。

## 12.5 Exit

只有 Gate 0 `PASS` 或“收缩 claim 后 PASS”，且原子发布并重新核验 `Stage0CompletionReceipt` 后，才进入新的 formal experiment。

---

# 13. Stage 1A：统一数据契约与数据边界

## 13.1 标准输入

```text
ResearchContractManifest
DatasetSpec
DatasetRegistryManifest
DatasetWindowPartition
AccessScope
```

## 13.2 标准输出

```text
DatasetRegistryEntry
DataManifest
Sha256Hash(data_hash)
WindowBatch (按请求 partition)
LeakageAudit (真实数据/新 loader 必须)
```

## 13.3 必要函数

```python
resolve_dataset(dataset: DatasetSpec) -> DatasetRegistryEntry

build_windows(
    dataset: DatasetSpec,
    partition: DatasetWindowPartition,
    access: AccessScope,
    grant: SealedAccessGrant | None = None,
) -> WindowBatch

hash_dataset(
    dataset: DatasetSpec,
    access: AccessScope,
    grant: SealedAccessGrant | None = None,
) -> Sha256Hash

validate_sealed_access(
    dataset: DatasetSpec,
    partition: DatasetWindowPartition,
    access: AccessScope,
    grant: SealedAccessGrant | None,
    accessed_at: UtcDatetime,
) -> None
```

未来 artifact transform：

```python
temporal_split(...) -> ArtifactRef
fit_transform_train_only(...) -> ArtifactRef
transform(...) -> WindowBatch
```

只有在 typed ArtifactStore 实现后才允许启用。

## 13.4 验收

- registry exact `(name, version)`；
- hash 实际计算并匹配；
- TEST 不允许擅自拼接 seen/unseen；
- unsealed 读取不要求 grant；sealed 读取必须在物理 I/O 前验证 grant；
- effective sealed 状态必须由 registry 与 access scope 共同决定；registry 已标记 sealed 时，调用方不能降级；
- sealed grant 必须精确匹配 dataset、scope、partition 和有效时间，否则 fail closed；
- sealed grant 不得绕过 train-only fit、sealed truth 或 scientific identity 规则；
- 不 re-window / re-normalize / cast / shuffle / re-split；
- `WindowBatch` shape/time/name/mask 全部通过。

## 13.5 下游

Stage 2、Stage 10、Stage 11 只能通过 `DatasetSpec + partition -> WindowBatch` 获取科学输入。

---

# 14. Stage 1B：合成 Regime-Switching SCM 与 Counterfactual Oracle

## 14.1 标准输入

```text
ResearchContractManifest
SyntheticConfig
DatasetSpec
Data contract / schema
random seed plan
```

## 14.2 标准输出

```text
DataManifest
SCMTruthManifest
WindowBatch(TRAIN/VALIDATION/TEST_SEEN/TEST_UNSEEN)
HighLevelInterventionResult[]
EffectRecord(source_kind=HIGH_LEVEL)[]
```

## 14.3 必要函数

```python
build_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset
persist_synthetic_dataset(...) -> PersistedSyntheticDataset
splits_from_synthetic_dataset(...) -> Stage2Splits
```

v2 新增标准 oracle bridge：

```python
build_scm_truth_manifest(
    dataset: SyntheticDataset,
) -> SCMTruthManifest
```

```python
high_level_intervene(
    pair: InterventionPair,
    truth: SCMTruthManifest,
    concept: ConceptIntervention,
    shared_future_noise: bool = True,
) -> HighLevelInterventionResult
```

```python
build_high_level_effect_record(...) -> EffectRecord
```

## 14.4 必须保持的 paired counterfactual 规则

- factual / counterfactual 使用相同未来外生噪声；
- 无干预时结果相同；
- trend intervention 不得暗改 scale latent；
- causal lag 真值保留；
- truth 不进入普通 prediction `WindowBatch` metadata。

## 14.5 Exit

E01/SCM oracle 工程验证通过；若 oracle 本身不稳定，禁止进入 Stage 3/4。

---

# 15. Stage 2：基础预测器与概率输出

## 15.1 标准输入

```text
DatasetSpec
WindowBatch(TRAIN/VALIDATION/...)
model_id / model registry entry
explicit device + dtype placement request
training config (仅训练路径)
```

## 15.2 标准输出

```text
ForecastPredictor (frozen for claim-bearing downstream)
ForecastDistribution
model_id
model_hash
checkpoint_hash
config_hash
RunManifest / model artifact refs
MetricRecord[] (forecasting validation)
```

## 15.3 必要函数

```python
resolve_predictor(model_id: str) -> ForecastPredictor
validate_predictor(predictor: ForecastPredictor) -> ForecastPredictor
prepare_stage2_prediction_batch(...) -> WindowBatch
predict_distribution(batch: WindowBatch) -> ForecastDistribution
```

训练路径必须独立：

```python
train_forecaster(...) -> ArtifactRef  # checkpoint/model artifact
```

不得由 `ForecastPredictor.predict_distribution()` 内部自动训练/选 checkpoint。

## 15.4 Exit

- 至少一个神经 predictor 满足预注册预测门槛；
- output contract 完整；
- claim-bearing 模型冻结；
- Stage 3/4 使用固定 model/checkpoint identity。

---

# 16. Stage 3：机制植入网络与 Mechanistic Adapter

## 16.1 标准输入

```text
frozen base model identity
WindowBatch
ConceptSpec[] / ConceptBatch
SCMTruthManifest
planting config
```

## 16.2 标准输出

```text
mechanistic model_id/checkpoint
MechanisticModelAdapter
InterventionSite[]
PlantedMechanismManifest
oracle intervention evidence
```

## 16.3 必要函数

```python
build_planted_model(...) -> ArtifactRef
resolve_mechanistic_adapter(model_id: str) -> MechanisticModelAdapter
list_intervention_sites(...) -> tuple[InterventionSite, ...]
validate_sites(...) -> None
build_planted_mechanism_manifest(...) -> PlantedMechanismManifest
```

## 16.4 Adapter 轴语义要求

每个 site 必须公开：

```text
site_name
layer
tensor_rank
batch_axis
variable_axis?
patch_axis?
feature_axis
shape_template
```

不能让 Stage 6/7 直接依赖某个模型的内部 module path 和 tensor layout。

## 16.5 Exit

oracle-site intervention 能控制目标效应；random/orthogonal complement 明显作为负对照。

---

# 17. Stage 4：固定位置时序交换干预

## 17.1 标准输入

```text
MechanisticModelAdapter
ResolvedInterventionPairBatch
InterventionSpec
InterventionSite registry
```

## 17.2 标准输出

```text
InterventionResult[]
optional activation-cache ArtifactRef
execution/identity metadata
```

## 17.3 必要函数

```python
validate_spec_against_site(spec, site) -> None

apply_intervention(
    pair_id: str,
    base: WindowBatch,
    source: WindowBatch,
    spec: InterventionSpec,
    model: MechanisticModelAdapter,
) -> InterventionResult
```

```python
capture(...) -> Mapping[str, Tensor]
```

## 17.4 行为不变量

- `source=base` -> effect≈0；
- no-op -> original forward；
- model parameters unchanged；
- hook cleanup in `finally`；
- lag 边界不足时丢弃，不循环 padding；
- full-rank subspace swap 与 full swap 在数值容差内一致。

---

# 18. Stage 5：Source Matching、Effect、Metrics、Gate A

## 18.1 标准输入

```text
ConceptBatch
PairingSpec
InterventionPair records
HighLevelInterventionResult[]
InterventionResult[]
EffectComputationSpec
MetricContext
GateSpec(GATE_A)
```

## 18.2 标准输出

```text
InterventionPairSet
ResolvedInterventionPairBatch
EffectRecord(HIGH_LEVEL)[]
EffectRecord(LOW_LEVEL)[]
EffectNormalizerState
MetricRecord(IIC/Cause/Isolation/Completeness/diagnostics)[]
GateDecision(GATE_A)
```

## 18.3 必要函数

```python
build_pairs(
    base: ConceptBatch,
    source: ConceptBatch,
    spec: PairingSpec,
) -> tuple[InterventionPair, ...]
```

```python
resolve_pair_set(...) -> ResolvedInterventionPairBatch
build_effect_signature(...) -> EffectSignature
build_high_level_effect_record(...) -> EffectRecord
build_low_level_effect_record(...) -> EffectRecord
```

```python
fit_normalizer(
    train_effects: Sequence[EffectRecord],
    spec: EffectNormalizationSpec,
) -> EffectNormalizerState
```

```python
normalize_effects(
    effects: Sequence[EffectRecord],
    state: EffectNormalizerState,
) -> tuple[EffectRecord, ...]
```

```python
abstraction_metrics(...) -> tuple[MetricRecord, ...]
evaluate_gate(...) -> GateDecision
```

## 18.4 Gate A 必须至少有数值冻结 predicate

- oracle-site held-out IIC vs random-site；
- Cause；
- Isolation；
- source=base effect；
- true lag vs wrong lag；
- random model/concept negative control；
- train vs held-out gap。

如果 threshold 未冻结，Gate 状态只能 `BLOCKED`，不能 `PASS`。

---

# 19. Stage 6：层 × 时间 Patch / Lag 渐进 OT 定位

## 19.1 标准输入

```text
Gate A PASS
normalized HighLevel EffectRecord[]
normalized LowLevel EffectRecord[]
MechanisticModelAdapter site catalog
LocalizationCandidate(stage=COARSE_LAYER/TIME_PATCH)[]
LocalizationRequest
```

## 19.2 标准输出

```text
OTProblem
OTResult
LocalizationTrace(COARSE_LAYER)
LocalizationTrace(TIME_PATCH)
selected candidate IDs
EfficiencyRecord
```

## 19.3 必要函数

```python
enumerate_layer_candidates(...) -> tuple[LocalizationCandidate, ...]
enumerate_time_candidates(...) -> tuple[LocalizationCandidate, ...]
```

```python
build_ot_problem(
    high_effects: Sequence[EffectRecord],
    low_effects: Sequence[EffectRecord],
    candidates: Sequence[LocalizationCandidate],
    config: LocalizationRequest,
) -> OTProblem
```

```python
solve_ot(problem: OTProblem, backend: OTBackend) -> OTResult
select_candidates(result: OTResult, request: LocalizationRequest) -> LocalizationResult
persist_localization_trace(...) -> LocalizationTrace
```

## 19.4 不变量

- 第一轮 coarse layer 不偷偷加入 variable/subspace 搜索；
- time patch 与 causal lag 显式区分；
- selection rule/threshold 在 validation 冻结；
- PLOT 算法/仓库是 baseline/参考，不是 hidden dependency。

---

# 20. Stage 7：变量轴与受限子空间

## 20.1 标准输入

```text
Stage6 selected candidates
MechanisticModelAdapter
model-specific canonical site axis
train activation data
EffectRecord[]
```

## 20.2 标准输出

```text
LocalizationCandidate(stage=VARIABLE)[]
LocalizationCandidate(stage=SUBSPACE)[]
LocalizationTrace(VARIABLE)
LocalizationTrace(SUBSPACE)
subspace basis ArtifactRef[]
```

## 20.3 必要函数

```python
enumerate_variable_candidates(...) -> tuple[LocalizationCandidate, ...]
fit_pca_subspaces(train_activations, rank_grid) -> tuple[ArtifactRef, ...]
build_random_orthogonal_subspaces(...) -> tuple[ArtifactRef, ...]
build_probe_subspace(...) -> ArtifactRef
```

PCA/probe 只能在 train activation 上 fit。

## 20.4 PatchTST / iTransformer 规则

- PatchTST：主用于 temporal patch；变量轴不能由 channel-independent 表示强行解释为内部 cross-variable mechanism。
- iTransformer：variate token 可作为 variable candidate；adapter 仍需公开 canonical axis，而不能由 localization 代码硬编码 iTransformer tensor layout。

---

# 21. Stage 8：DAS Refinement、联合真值与 Gate 1

## 21.1 标准输入

```text
Stage7 candidates
DASRefinementSpec
MechanisticModelAdapter
train/validation/heldout pair sets
PlantedMechanismManifest
PLOT / Full-DAS baseline configs
```

## 21.2 标准输出

```text
LocalizationTrace(DAS_REFINEMENT)
LocalizationTrace(HELDOUT_EVAL)
MetricRecord(localization + abstraction + efficiency)[]
StatisticalTestResult[]
GateDecision(GATE_1)
```

## 21.3 必要函数

```python
refine_das(
    candidates: Sequence[LocalizationCandidate],
    spec: DASRefinementSpec,
    model: MechanisticModelAdapter,
) -> LocalizationTrace
```

```python
evaluate_joint_truth(
    localization: LocalizationTrace,
    truth: PlantedMechanismManifest,
    context: MetricContext,
) -> tuple[MetricRecord, ...]
```

## 21.4 Gate 1 关注对象

必须同时评估：

```text
layer
variable
patch/time
causal lag
subspace
forecast horizon effect
```

并与：

```text
PLOT
PLOT-guided DAS
Full DAS
random localization
oracle-site DAS
```

比较。

---

# 22. Stage 9：环境定义、DRO、Zero-Refit 与理论

## 22.1 标准输入

```text
Gate1 PASS
frozen localization/subspace
train/validation/test EffectRecord[]
EnvironmentSpec / train-fit environment model
EnvironmentAssignmentBatch
RobustnessTrainingSpec
EffectNormalizerState
```

## 22.2 标准输出

```text
RobustnessModelState
RobustnessEvaluation(validation)
RobustnessEvaluation(test_seen)
RobustnessEvaluation(test_unseen)
MetricRecord[]
GateDecision(GATE_2)
theory assumption/proof ArtifactRef[]
```

## 22.3 必要函数

```python
define_environments_train(...) -> ArtifactRef  # fitted environment definition
assign_environments(...) -> EnvironmentAssignmentBatch
```

```python
fit_robustness(
    train_effects: Sequence[EffectRecord],
    train_env: EnvironmentAssignmentBatch,
    validation_effects: Sequence[EffectRecord],
    validation_env: EnvironmentAssignmentBatch,
    spec: RobustnessTrainingSpec,
) -> RobustnessModelState
```

```python
apply_robustness(
    state: RobustnessModelState,
    effects: Sequence[EffectRecord],
    environments: EnvironmentAssignmentBatch,
) -> RobustnessEvaluation
```

## 22.4 Zero-refit 验证

进入 test/unseen 前必须冻结并 hash：

- predictor；
- localization candidate/site；
- subspace basis；
- effect normalizer；
- concept mapping；
- environment definition；
- cost scaler；
- radius；
- robustness parameters。

如果 test 发生任一 fit，`zero_refit_verified=False`，Gate2 主张不能通过。

## 22.5 Wasserstein solver 单元测试（推荐）

适用且资源允许时，建议为小问题提供精确 primal/LP oracle，并检查：

- primal/dual 数值一致；
- radius=0 接近 ERM；
- cost non-negative；
- diagonal 0；
- train-only scaler；
- rho selection validation-only。

---

# 23. Stage 10：通用时间序列跨域验证

## 23.1 标准输入

```text
Gate1/Gate2 required evidence
DatasetSpec(Weather/Electricity/Traffic/fev/GIFT-Eval task)
WindowBatch
ConceptSpec
frozen predictor/mechanistic adapter
frozen localization
optional frozen RobustnessModelState
```

## 23.2 标准输出

```text
ExperimentSummary per domain/model
MetricRecord[]
LeakageAudit
EfficiencyRecord[]
failure-region diagnostics ArtifactRef
cross-domain check GateDecision-like evidence
```

## 23.3 关键原则

Stage10 不定义新一套解释数据类型。它**复用 Stage1–9 同一协议**：

```text
WindowBatch
-> ForecastDistribution
-> ConceptBatch
-> InterventionPair
-> InterventionResult
-> EffectRecord
-> LocalizationTrace
-> RobustnessEvaluation
-> MetricRecord
```

这正是避免 synthetic 与 real-data 两套 pipeline 对接不上的核心规则。

---

# 24. Stage 11：金融数据治理与压力测试

## 24.1 标准输入

```text
financial DatasetSpec
DataManifest
release/vintage metadata
rolling split spec
purging/embargo spec
frozen TARCA method state
```

## 24.2 强制先行输出

在任何 claim-bearing finance run 前必须先产生：

```text
LeakageAudit(passed=True)
```

并有以下 evidence refs：

- feature/label interval audit；
- scaler fit interval；
- state-model fit interval；
- release-time alignment；
- pair partition separation；
- universe/survivorship policy；
- corporate-action policy（适用时）。

## 24.3 标准科学输出

仍然复用：

```text
ForecastDistribution
EffectRecord
LocalizationTrace
RobustnessEvaluation
MetricRecord
StatisticalTestResult
ExperimentSummary
```

## 24.4 Gate 3

Gate 3 是 exploratory prediction benefit：

- 不能关闭 Gate A/1/2；
- finance failure 不自动否定已通过的 mechanistic claim；
- 只有金融收益也不能反向证明通用方法。

---

# 25. Stage 12：全量消融、统计与效率

## 25.1 标准输入

```text
ExperimentSummary[]
MetricRecord[]
EfficiencyRecord[]
AblationSpec[]
StatisticalTestSpec[]
```

## 25.2 标准输出

```text
StatisticalTestResult[]
ablation table ArtifactRef
efficiency table ArtifactRef
final result table ArtifactRef
GateDecision(GATE_4 candidate)
```

## 25.3 必要函数

```python
run_ablation(spec: AblationSpec, parent: ExperimentSpec) -> ExperimentSummary
run_statistical_test(spec: StatisticalTestSpec, records: Sequence[MetricRecord]) -> StatisticalTestResult
aggregate_efficiency(records: Sequence[EfficiencyRecord]) -> ArtifactRef
build_paper_tables(...) -> tuple[ArtifactRef, ...]
```

---

# 26. Stage 13：可复现发布

## 26.1 标准输入

```text
ResearchContractManifest
all DataManifest / model manifests
GateDecision[]
ExperimentSummary[]
StatisticalTestResult[]
source git commit
environment lock
licenses
```

## 26.2 标准输出

```text
ReleaseManifest
REPRODUCE.md
DATA.md
INSTALL.md
CITATION.cff
result table refs
figure-data refs
```

## 26.3 Reproduction Level

```text
LEVEL_0_CPU_SMOKE
LEVEL_1_SINGLE_GPU_CORE
LEVEL_2_FULL_PAPER
```

每一个 level 必须在 `ReleaseManifest.reproduction_levels` 绑定到明确入口和 evidence。

---

# 27. Experiment / Task 编译协议

科学 experiment 不能直接变成 shell command。

## 27.1 `ExperimentSpec`

```python
class ExperimentSpec:
    experiment_id: str
    protocol_id: str
    tasks: tuple[TaskSpec, ...]
```

## 27.2 编译

```python
validate_experiment(spec: ExperimentSpec) -> None
compile_experiment(spec: ExperimentSpec) -> TaskManifest
```

TaskSpec 输入必须全部是 ArtifactRef 或 frozen identity；不能包含“去某目录找最新 checkpoint”。

## 27.3 执行计划

```python
inspect_resources() -> ResourceSnapshot
plan_execution(
    manifest: TaskManifest,
    resources: ResourceSnapshot,
    backend_id: str,
) -> ExecutionPlan
```

ExecutionPlan 只解决：

- 并行度；
- worker placement；
- GPU assignment；
- CPU thread allocation；
- memory admission；
- attempt scheduling。

不能改变：

- model config；
- seed；
- input artifact；
- precision policy（若 scientific identity 已冻结）；
- task count；
- metric/gate。

---

# 28. 服务器执行协议

## 28.1 Science Contract 与服务器完全解耦

同一个 `TaskManifest` 应能够被：

```text
LocalBackend
SingleGPUBackend
MultiGPUBackend
RemoteServerBackend
```

执行，而 scientific output identity 不因 backend 名称变化。

## 28.2 RemoteServerBackend 只负责

```text
secure connection
artifact staging
resource inspection
ExecutionPlan launch
heartbeat/telemetry
result retrieval
artifact verification
cleanup
```

服务器接入严格遵守 `TARCA_SERVER_ACCESS_RUNBOOK.md`：

- 连接事实来自既有环境变量；
- 白名单解析 SSH；
- 固定 probe；
- 不打印秘密；
- 临时 key/known_hosts/proxy script 清理；
- 主机密钥冲突/认证异常 fail closed。

## 28.3 禁止

- science module import server code；
- data adapter 读取 SSH；
- model registry 根据 GPU 数切换模型；
- scheduler 看到 partial metric 后改变 model/seed；
- runtime 访问 sealed truth；
- dashboard 展示 claim-bearing partial scientific metrics 参与人工择优。

---

# 29. Monitoring 协议

```python
class ReadOnlyMonitor(Protocol):
    def read_status(self) -> MonitoringSnapshot: ...
    def read_resources(self) -> MonitoringSnapshot: ...
    def read_telemetry(self) -> MonitoringSnapshot: ...
```

允许：

- pending/running/completed/failed count；
- GPU/CPU/RAM utilization；
- heartbeat；
- ETA；
- worker health；
- terminal status。

禁止：

- partial NLL/CRPS/MAE 排名；
- unseen/sealed truth；
- model-selection recommendation；
- “哪个 seed 当前最好”之类会污染预注册决策的信息。

---

# 30. Arrow / Parquet Schema 最低要求

所有正式表格必须带 schema metadata：

```text
contract_schema_version
protocol_id
artifact_type
```

至少定义：

## 30.1 predictions

```text
window_id: string non-null
split: string non-null
forecast_time: timestamp[UTC] non-null
horizon: int32 non-null
target: string non-null
y_true: float nullable
mean: float non-null
scale: float nullable
```

## 30.2 intervention_pairs

完整对应 `InterventionPair`。

## 30.3 effects

```text
pair_id
concept_name
source_kind
model_id nullable
candidate_id nullable
horizon
target
effect_component
quantile_level nullable
value
```

## 30.4 metrics

```text
experiment_id
run_id
split
metric_name
value
regime nullable
horizon nullable
concept nullable
```

## 30.5 localization

```text
trace_id
stage
candidate_id
parent_candidate_id nullable
selected
score/cost
transport_mass nullable
runtime_seconds
```

Schema 不匹配必须 fail-fast；禁止“能被 pandas 读出来就算通过”。

---

# 31. Gate 规范

## 31.1 Gate 0 — Novelty

输出：`GateDecision`。
如果核心 claim 被覆盖：`DROP_CLAIM/STOP`，不能通过改数据集名称继续。

## 31.2 Gate A — Fixed-site Causal Intervention

必须验证：

- oracle vs random site；
- Cause；
- Isolation；
- source=base；
- true vs wrong lag；
- random model/concept；
- held-out pairs。

## 31.3 Gate 1 — Joint Localization + Anti-vacuity

必须验证：

- intervention truth；
- layer/variable/patch/lag/subspace joint truth；
- capacity frontier；
- random model/concept；
- PLOT/DAS baselines；
- held-out generalization；
- efficiency。

## 31.4 Gate 2 — Sequential Unseen Regime Zero-Refit

必须验证：

- zero-refit hash closure；
- worst-regime error vs ERM/Group-DRO/DiRoCA-style/random reweighting；
- average performance tradeoff；
- random environment negative control。

## 31.5 Cross-domain Check

不是 Gate 3。至少两个非金融域提供机制稳定 evidence。

## 31.6 Gate 3 — Exploratory Prediction Benefit

仅附加 claim；失败可继续写纯机制论文。

## 31.7 Gate 4 — Paper/Release Completeness

理论、算法、基准、真实实验、负对照、统计、复现材料完整。

---

# 32. 错误码与 Fail-Closed

```text
PROTOCOL_ERROR
CONTRACT_ERROR
DATA_ERROR
SCIENTIFIC_FAIL
RESOURCE_BLOCKED
RUNTIME_ERROR
ARTIFACT_INVALID
SEALED_ACCESS_VIOLATION
ARCHITECTURE_VIOLATION
UNIMPLEMENTED_CAPABILITY
AUTHORIZATION_BLOCKED
```

新增建议细分（若 CCP 批准）：

```text
IDENTITY_MISMATCH
PAIR_RESOLUTION_FAILED
EFFECT_CONTRACT_INVALID
LOCALIZATION_CONTRACT_INVALID
GATE_SPEC_INCOMPLETE
ZERO_REFIT_VIOLATION
SCHEMA_MISMATCH
```

若不新增 error enum，可映射到上面的通用大类，并在 rationale 中记录 subtype。

---

# 33. 公共函数矩阵

| 模块 | 必要函数 | 输入 | 输出 |
|---|---|---|---|
| contracts | `validate_*` | named contract | same/None |
| data | `resolve_dataset` | DatasetSpec | DatasetRegistryEntry |
| data | `build_windows` | DatasetSpec+partition+scope | WindowBatch |
| data | `hash_dataset` | DatasetSpec+scope | hash |
| synthetic | `build_synthetic_dataset` | SyntheticConfig | SyntheticDataset |
| synthetic | `build_scm_truth_manifest` | SyntheticDataset | SCMTruthManifest |
| models | `resolve_predictor` | model_id | ForecastPredictor |
| models | `resolve_mechanistic_adapter` | model_id | MechanisticModelAdapter |
| concepts | `compute` | WindowBatch | ConceptBatch |
| concepts | `leakage_audit` | WindowBatch | LeakageAudit |
| pairs | `build_pairs` | ConceptBatch+PairingSpec | InterventionPair[] |
| pairs | `resolve_set` | PairSet+dataset+partition | ResolvedInterventionPairBatch |
| interventions | `apply_intervention` | pair/base/source/spec/model | InterventionResult |
| high-level | `high_level_intervene` | pair+truth+concept | HighLevelInterventionResult |
| effects | `build_effect_signature` | factual+intervened+spec | EffectSignature |
| effects | `fit_normalizer` | train EffectRecord[] | EffectNormalizerState |
| effects | `normalize_effects` | effects+state | EffectRecord[] |
| localization | `enumerate_*` | site catalog/parent candidates | LocalizationCandidate[] |
| localization | `build_ot_problem` | effects+candidates | OTProblem |
| backend | `solve` | OTProblem | OTResult |
| localization | `select_candidates` | OTResult+request | LocalizationResult |
| localization | `refine_das` | candidates+DAS spec | LocalizationTrace |
| robustness | `assign_environments` | data/pairs+train-fitted def | EnvironmentAssignmentBatch |
| robustness | `fit_robustness` | train/val effects+env | RobustnessModelState |
| robustness | `apply_robustness` | frozen state+effects+env | RobustnessEvaluation |
| metrics | `forecasting_metrics` | forecast+target+context | MetricRecord[] |
| metrics | `abstraction_metrics` | high/low effects+context | MetricRecord[] |
| metrics | `localization_metrics` | trace+truth+context | MetricRecord[] |
| governance | `evaluate_gate` | GateSpec+metrics+evidence | GateDecision |
| artifacts | `publish_*` | typed payload | ArtifactRef |
| experiments | `compile_experiment` | ExperimentSpec | TaskManifest |
| runtime | `plan_execution` | TaskManifest+resources | ExecutionPlan |
| runtime | `execute_task` | ExecutionContext | TaskResult |
| stats | `run_statistical_test` | spec+metrics | StatisticalTestResult |
| release | `build_release_manifest` | all evidence | ReleaseManifest |

---

# 34. 依赖方向规则

允许：

```text
contracts <- all modules

data -> contracts
models -> contracts
concepts -> contracts
interventions -> contracts

effects -> contracts
localization -> contracts + effects interfaces + backends interfaces
robustness -> contracts + effects interfaces
metrics -> contracts
experiments -> contracts
```

禁止：

```text
data -> models
models -> data concrete implementation
models -> concepts concrete implementation
science -> runtime/server/ssh
runtime -> scientific solver internals
monitoring -> target/metric ranking
contracts -> concrete science implementation
```

必要的对象协作通过调用方 orchestration 注入，而不是模块互相 import concrete implementation。

---

# 35. Stage Acceptance Test 推荐清单（非强制）

以下测试均为推荐项，而不是 Stage 合格或实现 PR 的强制条件。适用且资源允许时建议实现；未实现这些测试本身不构成阻断，但已经实现的测试应保持通过，并可作为验收证据：

1. 正确输入成功；
2. 错误 type fail；
3. 错误 shape/schema fail；
4. identity/hash mismatch fail；
5. 上下游 round-trip；
6. deterministic seed evidence（适用时）；
7. train/test separation；
8. forbidden dependency test；
9. fail-closed test；
10. artifact reload+verify；
11. public API signature test；
12. no hidden execution/environment dependency；
13. baseline regression；
14. completion receipt 校验测试。

本节只规定测试实现的推荐范围，不取消协议其他章节对契约、产物、Gate 决策或 completion receipt 本身的独立要求。

---

# 36. Formal Experiment 前冻结清单

第一次 claim-bearing formal experiment 前，以下不得再为自然语言：

- Gate A/1/2/3/4 predicates；
- OT selection strategy；
- transport mass threshold / Top-k；
- DAS rank grid；
- orthogonality tolerance；
- source matching min concept delta；
- max source reuse；
- Effect normalizer；
- IIC/Cause/Isolation normalizer；
- Wasserstein rho search space；
- environment definition；
- unseen regime rule；
- statistical test / CI / multiple comparison；
- stop/repair rule。

这些必须进入 `GateSpec` / `PairingSpec` / `EffectComputationSpec` / `DASRefinementSpec` / `RobustnessTrainingSpec` 等可 hash 的对象。

---

# 37. Change Control

以下变化必须 CCP + version bump：

- Contract class 新增/删除/修改必需字段；
- Tensor axis 语义；
- Window split；
- data/model/checkpoint identity；
- intervention kind；
- effect formula；
- candidate axis；
- Gate predicate；
- artifact serializer/schema；
- task completion/retry policy；
- sealed authorization model。

兼容性变化可以 minor version：

- 新增可选 metadata 且不影响 identity；
- 新 backend 且数值语义等价；
- 新 monitoring field 且不暴露 scientific data。

---

# 38. 兼容与演进策略

M1/M2 应保持下列标准主路径，扩展不得改变其科学输入语义。

## M1

标准主路径：

```text
DatasetSpec
-> dataset registry
-> allowlisted adapter
-> canonical hash verification
-> validated WindowBatch
```

## M2

标准主路径：

```text
WindowBatch
-> ForecastPredictor
-> ForecastDistribution
```

并保持 prediction-only 与 mechanistic capability separation。

后续 Stage 应通过**新增合法 bridge**扩展：

```text
ForecastPredictor
  + separate MechanisticModelAdapter
  -> InterventionResult
  -> EffectRecord
  -> LocalizationTrace
  -> RobustnessModelState
```

而不是回头修改 M1/M2 的科学输入语义。

---

# 39. 协议最终闭环

全项目的唯一标准科学链路为：

```text
ResearchContractManifest
  -> DatasetSpec / DataManifest / WindowBatch
  -> ForecastPredictor
  -> ForecastDistribution
  -> ConceptSpec / ConceptBatch
  -> PairingSpec / InterventionPair / ResolvedInterventionPairBatch
  -> HighLevelInterventionResult + InterventionResult
  -> EffectRecord / EffectNormalizerState
  -> LocalizationCandidate / OTProblem / OTResult / LocalizationTrace
  -> DAS Refinement / Heldout Evaluation
  -> RobustnessModelState / RobustnessEvaluation
  -> MetricContext / MetricRecord
  -> GateSpec / GateDecision
  -> StatisticalTestResult / EfficiencyRecord
  -> ReleaseManifest
```

执行链路与其正交：

```text
ExperimentSpec
  -> TaskManifest
  -> ResourceRequest
  -> ExecutionPlan
  -> ExecutionContext
  -> TaskResult
  -> verified ArtifactRef
```

服务器只是：

```text
ExecutionPlan 的 backend
```

而不是：

```text
新的 scientific pipeline
```

---

# 40. 最终协议判定

当本协议经过 CCP 冻结后，后续任何 Codex/Agent/人工实现必须遵守：

1. **没有标准输入，不得实现函数。**
2. **没有标准输出，不得返回 ad-hoc dict/object。**
3. **没有 producer/consumer 的类不得新增。**
4. **没有 ArtifactStore typed path，不得伪造 ArtifactRef。**
5. **除 `GATE_0_NOVELTY` 人工新颖性 Gate 外，没有 GateSpec 的 Gate 不得声称 PASS。**`GATE_0_NOVELTY 是人工新颖性 Gate，不要求 GateSpec`，但必须绑定冻结的 novelty claims 与 related-work bundle。
6. **没有 PairRegistry 不得按行号配对。**
7. **没有 Effect Bridge 不得直接把 InterventionResult 传给 localization。**
8. **没有 MetricContext 不得制造 experiment/run/split ID。**
9. **没有 ExecutionPlan 不得把 scientific task 直接拼成服务器 shell pipeline。**
10. **任何 runtime/backend 变化不得修改 scientific identity。**
11. **Stage N 的下游只能读取本协议列出的 Stage N 输出或其 ArtifactRef。**
12. **所有未实现能力必须 fail closed，而不是先返回占位结果让后续继续。**

满足这些规则后，TARCA 的工程结构应具有如下性质：

> 一个 Stage 可以独立开发、测试、替换 backend、迁移到服务器，并通过固定的 typed contract 与下一 Stage 连接；上游科学身份不会因下游实现变化而被静默重定义。

---

# 41. 主要外部依据（检索核对：2026-08-20）

## 因果抽象 / 机制定位

1. Geiger et al. — *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*, arXiv:2301.04709.
2. Geiger et al. — *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*, arXiv:2303.02536.
3. Chang et al. — *PLOT: Progressive Localization via Optimal Transport in Neural Causal Abstraction*, arXiv:2605.06979; official repository `jchang153/causal-abstractions-ot`.
4. Felekis et al. — *Distributionally Robust Causal Abstractions*, arXiv:2510.04842; official repository `yfelekis/DiRoCA`.
5. Stanford NLP — `pyvene`, PyTorch internal-state intervention library.

## OT / 模型

6. PythonOT — `POT`, Sinkhorn/UOT and related optimal transport solvers.
7. Nie et al. — *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*, arXiv:2211.14730; official repository `yuqinie98/PatchTST`.
8. Liu et al. — *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, arXiv:2310.06625; official repository `thuml/iTransformer`.

## 契约 / 数据持久化

9. Python 3.11 `typing.Protocol` official documentation.
10. Pydantic v2 official model/config documentation (`strict`, `frozen`, `extra=forbid`).
11. Apache Arrow / PyArrow `Schema` and Parquet official documentation.
12. PyTorch `torch.distributions` official documentation.
13. AutoGluon `fev` official repository (forecast task/evaluation summary and dataset fingerprint design reference).
14. Salesforce AI Research `GIFT-Eval` official repository.

---

# 42. 来源文件

本协议以项目目录中实际存在的以下 TARCA 文件为核心依据：

- `docs/auth/TARCA_项目计划书.md`
- `docs/auth/TARCA_具体实施计划.md`
- `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`
