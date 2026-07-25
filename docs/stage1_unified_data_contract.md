# Stage 1 统一数据合同

## 1. 目的与非目标

本文件定义 Stage 1A 各模块交换运行时张量、持久化元数据和表格制品时必须遵守的工程合同。
它统一字段名、形状、时间边界、分区身份、序列化模式和安全路径，使独立模块能在不共享内部实现的
前提下互操作。

本阶段只提供合同、校验器、静态协议和模式，不提供真实预测模型、训练流程、数据加载器、模型
hook、SCM、OT、DAS、DRO、定位算法、金融策略或实验运行器，也不创建任何正式运行制品。

## 2. 唯一规范包与导入规则

跨模块代码必须从 `tarca.contracts` 导入稳定名称，例如：

```python
from tarca.contracts import WindowBatch, ForecastDistribution
```

`tarca.contracts` 是唯一稳定的跨模块导入面；`tarca.contracts.data` 等子模块是实现组织细节。
根包只含显式导入、`__all__` 和短文档字符串，不重复定义合同，也不提供兼容别名。
`StrictContractModel` 是 Pydantic 实现基类，不属于根公开 API。

## 3. 模式版本与兼容策略

`CONTRACT_SCHEMA_VERSION` 当前为 `"1.0.0"`。每个持久化 Pydantic 合同都带有同值的
`schema_version` 字面量字段；Arrow Schema 元数据也带有
`contract_schema_version="1.0.0"`。读取方必须按版本校验，不能静默猜测不匹配字段的含义。

版本号描述的是工程数据合同，而不是模型、数据集或实验结论的版本。兼容规则见第 12 节。

## 4. 公开类型、字段与不变量

### 4.1 枚举与 JSON 元数据别名

- `SplitPartition` 的唯一值是 `train`、`validation`、`test`。
- `RegimeRelation` 的唯一值是 `same`、`cross`、`unknown`。
- `InterventionKind` 的唯一值是 `full_swap`、`subspace_swap`。
- `RunStatus` 的唯一值是 `pending`、`running`、`completed`、`failed`。
- `JSONScalar` 是 `None | bool | int | float | str`。
- `JSONValue` 递归允许 JSON scalar、字符串键映射和序列；`JSONMetadata` 是字符串键到
  `JSONValue` 的映射。浮点值必须有限，键必须是字符串，Tensor 不是 JSON 值。校验后的映射和
  序列会变成只读映射与 tuple，但其中若有其他可变对象，类型别名本身不承诺深层物理不可变。

### 4.2 `WindowBatch`

`WindowBatch` 是 `frozen=True, slots=True` 的运行时 dataclass。字段如下：

| 字段 | 含义与约束 |
| --- | --- |
| `x` | 必需的有限浮点 Tensor，形状 `[B, L, D]`，三个维度均为正。 |
| `y` | 可选的有限浮点 Tensor，形状 `[B, H, Dy]`；存在时决定目标数和候选预测长度。 |
| `observed_covariates` | 可选的有限浮点 Tensor，形状 `[B, L, Do]`，历史长度必须与 `x` 相同。 |
| `known_future_covariates` | 可选的有限浮点 Tensor，形状 `[B, H, Dk]`，预测长度必须与其他候选一致。 |
| `x_observed_mask` | 可选 bool Tensor；存在时形状和设备必须与 `x` 完全相同。 |
| `y_observed_mask` | 可选 bool Tensor；存在时要求 `y` 存在，且形状和设备完全相同。 |
| `observed_covariates_mask` | 可选 bool Tensor；存在时要求对应协变量存在，且形状和设备完全相同。 |
| `known_future_covariates_mask` | 可选 bool Tensor；存在时要求对应协变量存在，且形状和设备完全相同。 |
| `regime` | 可选整数 Tensor，形状 `[B]`，不能是 bool，设备必须与 `x` 相同。 |
| `window_id` | 每个样本一个非空且唯一的字符串。 |
| `input_feature_names` | 长度为 `D` 的非空、唯一字符串 tuple。 |
| `target_names` | `y` 存在时长度为 `Dy`；`y` 缺失时必须为空。 |
| `observed_covariate_names` | 对应 Tensor 存在时长度为 `Do`，否则必须为空。 |
| `known_future_covariate_names` | 对应 Tensor 存在时长度为 `Dk`，否则必须为空。 |
| `feature_start` | 每个样本一个已带时区且 UTC offset 为零的 `datetime`。 |
| `feature_end` | 每个样本一个 UTC `datetime`，为历史特征边界末端。 |
| `prediction_start` | 每个样本一个 UTC `datetime`，为预测可用信息的截断点。 |
| `label_end` | 每个样本一个 UTC `datetime`，为标签区间末端。 |
| `forecast_time` | 每个样本一个严格递增的 UTC 时间 tuple；所有样本长度相同且为正。 |
| `metadata` | 经过上述 JSON 兼容性校验的映射。 |

批大小必须在所有对齐 Tensor、ID 和时间字段间一致。由 `y`、`known_future_covariates` 与
`forecast_time` 得出的 `H` 必须唯一且为正。`target_names` 与
`known_future_covariate_names` 必须不相交。所有数值 Tensor 即使在 mask 为 false 的位置也必须
有限；合同不会用 mask 替换或修复数值。

### 4.3 `ForecastDistribution`

- `mean`：必需的有限浮点 Tensor，形状 `[B, H, Dy]`，各维为正。
- `scale`：可选；形状、设备、dtype 与 `mean` 完全相同，且每个值严格大于零。
- `quantiles`：实数且非 bool 的 level 到 Tensor 的映射。level 必须有限且严格位于 `(0, 1)`，
  归一化为 float 后仍唯一；Tensor 与 `mean` 对齐。按 level 递增时预测必须逐元素不下降。
- `logits`：可选，形状 `[B, H, Dy, C]`，`C > 1`，设备和 dtype 与 `mean` 相同。
- `samples`：可选，形状 `[S, B, H, Dy]`，`S > 0`，设备和 dtype 与 `mean` 相同。
- `window_id`：可选；存在时为长度 `B` 的非空、唯一字符串 tuple。
- `target_names`：长度 `Dy` 的非空、唯一字符串 tuple。

所有预测 Tensor 必须已物化且有限。`quantiles` 的外层映射会冻结，但 Tensor 本身仍可变。

### 4.4 `ConceptBatch`

- `values` 是形状 `[B, K]`、维度为正、已物化且有限的浮点 Tensor。
- `valid_mask` 是与 `values` 同形状、同设备的 bool Tensor。
- `names` 和 `window_id` 分别是长度 `K` 和 `B` 的非空、唯一字符串 tuple。
- `computed_from_history_only` 必须是精确的 bool，不能用 `0`、`1` 或其他 truthy 值代替。
- `definition_version` 必须是非空字符串。

该类记录“仅由历史计算”的声明和概念定义版本；bool 字段本身不能证明实现没有泄漏。

### 4.5 干预合同与校验函数

`InterventionSite` 描述一个张量位置：

- `site_name` 非空；`layer` 为 `None` 或非负整数。
- `tensor_rank` 为正整数；`shape_template` 必须是同长度 tuple，每项为 `None` 或正整数。
- `batch_axis`、`feature_axis` 必需，`variable_axis`、`patch_axis` 可选；所有已提供轴都在 rank
  范围内且两两不同。

`InterventionSpec` 描述一次交换请求：

- `site_name` 非空；`layer`、`variable_index`、`patch_index` 为 `None` 或非负整数；
  `lag` 是非 bool 的整数，可以为负。
- `intervention_kind` 必须是实际的 `InterventionKind` 成员，不做字符串强制转换。
- `FULL_SWAP` 禁止携带 `subspace_basis`；`SUBSPACE_SWAP` 必须携带 basis。
- basis 必须是已物化、有限的二维浮点 Tensor，至少一列，列数不大于行数，列向量数值正交归一。

`basis_orthonormality_tolerance(dtype)` 接受浮点 `torch.dtype`，精确返回：

```text
max(1e-7, 8 * torch.finfo(dtype).eps)
```

该值同时作为 `torch.allclose(basis.T @ basis, I)` 的 `atol` 和 `rtol`。它只补偿 dtype
相关的浮点舍入误差，是工程数值容差，不是论文指标、方法成功阈值或经验结论。

`validate_spec_against_site(spec, site)` 做跨对象校验：`site_name` 和 `layer` 必须完全相同；
索引只有在相应轴存在时才允许，并在已知维度下检查上界；当 feature 维度已知且存在 basis 时，
basis 第一维必须与它相同。构造 `InterventionSpec` 不能替代调用该函数。

### 4.6 持久化 Pydantic 合同

以下模型都使用 `ConfigDict(extra="forbid", frozen=True, strict=True)`，并继承固定的
`schema_version`：

- `DataSplitSummary`：`partition`、`split_hash`、`count`。hash 必须为
  `sha256:` 加 64 位小写十六进制；计数为非负整数。
- `WindowContractSummary`：正整数 `history_length`、`horizon`；四组特征名 tuple 可为空，
  但其中每个元素若存在则必须非空且组内唯一；`timezone` 只能是 `"UTC"`；
  `missingness_protocol` 非空。
- `InterventionPair`：`pair_id`、`partition`、不同的 `base_window_id` 与
  `source_window_id`、`concept_name`、`regime_relation`、非负有限
  `matching_distance`、有限 `concept_delta`。`pair_id` 必须为 `sha256:` 加 64 位小写
  十六进制。`InterventionPair.build(...)` 生成规范 ID：ID 只由 base/source ID、概念名和
  regime relation 的规范 JSON 计算 SHA-256，不受分区或数值统计量影响；直接构造时也会复核
  该 ID。
- `DataManifest`：非空 `dataset_name`、`dataset_version`、`source_description`；
  `dataset_hash` 必须为 `sha256:` 加 64 位小写十六进制；`created_at` 为 UTC，另含
  `window_contract` 与 `splits`。`splits` 必须恰好各含一个 train、validation、test 摘要。
- `RunManifest`：非空 `experiment_id`、`run_id`；`config_hash` 和 `data_hash` 各自都
  必须为 `sha256:` 加 64 位小写十六进制；`git_commit` 为 40 位小写十六进制，另含 UTC
  `created_at` 和 `RunStatus`。它记录声明的运行状态，不执行运行。
- `MetricRecord`：非空实验、运行和 metric 名，`split`，有限 `value`；`regime`、
  `concept` 可选但若存在必须非空，`horizon` 可选但若存在必须为正整数。

`validate_disjoint_window_partitions(partitions)` 要求映射使用全部三个 `SplitPartition` 键，
拒绝同一 window ID 跨分区，并只报告排序后的前五条冲突证据。

`validate_intervention_pair_partitions(pairs)` 拒绝 window ID 或 `pair_id` 跨 pair 分区复用，
也提供确定且有界的前五条排序证据。两个函数只验证身份分配，不证明数据生成过程没有其他泄漏。

### 4.7 其他公开面

`ForecastModelAdapter` 的字段和局限见第 8 节；三个 Arrow schema 工厂与
`validate_arrow_schema` 见第 9 节；`ArtifactLayout` 见第 10 节。它们同样属于
`tarca.contracts` 的规范公开 API。

## 5. 时间、mask、known-future、target 与泄漏边界

每个窗口必须满足：

```text
feature_start <= feature_end < prediction_start <= label_end
```

每个 `forecast_time` 都严格递增，并位于闭区间 `[prediction_start, label_end]`。`x` 和
`observed_covariates` 只能描述 `prediction_start` 之前的历史；`known_future_covariates`
只能包含在预测时刻已经可知的外生量，不能通过事后值回填。结构校验无法判断一个业务字段是否
真的提前可知，因此生产者必须记录来源并在数据流水线中审计这一语义。

mask 必须与对应 Tensor 同形状、同设备且为 bool。mask 只表达观测有效性，不能放宽“底层数值
必须有限”的要求。目标名和 known-future 名必须不相交，防止把目标本身冒充已知未来变量。
`computed_from_history_only` 是显式声明，不是自动因果证明。

train、validation、test 的 window ID 必须互斥；干预 pair 的 window ID 和规范 pair ID 也不能
跨分区。调用方还必须确保缩放、特征工程、匹配和概念定义不使用相应分区边界之外的信息。

## 6. 为什么运行时 Tensor 与持久化 manifest 分离

`WindowBatch`、`ForecastDistribution` 和 `ConceptBatch` 保留 Tensor 的 dtype、device、
stride、autograd 状态与对象身份，适合进程内计算，但这些属性不属于稳定 JSON 表示。持久化
manifest 使用严格 Pydantic 字段记录身份、版本、哈希、计数和来源；Arrow Schema 单独约束长表
制品。分离后，运行时数据不会为了序列化而被隐式复制或搬移，持久化记录也不会依赖设备相关对象。

manifest 是可验证的工程描述，不是 Tensor 内容的替身；保存制品时仍需同时校验相应 manifest、
Arrow Schema 和实际数据。

## 7. `frozen=True` 与 Tensor 的物理可变性

dataclass 或 Pydantic 的 `frozen=True` 只禁止给字段重新赋值。合同刻意保留调用方传入 Tensor
的对象身份，不 clone、不 detach，也不改变 layout；因此持有同一 Tensor 的代码仍可原地修改其
存储。冻结的外层对象不等于 Tensor 存储物理不可变，也不提供线程安全、哈希稳定性或自动快照。
需要不可变证据的调用方必须在合同之外明确复制、内容哈希或采用只读存储策略。

## 8. `ForecastModelAdapter` 的静态边界与局限

`ForecastModelAdapter` 是未标注 `runtime_checkable` 的 `typing.Protocol`，公开：

- 只读属性 `adapter_name: str`、`model_hash: str`、`is_frozen: bool`；
- `predict_distribution(batch) -> ForecastDistribution`；
- `list_intervention_sites() -> tuple[InterventionSite, ...]`；
- `capture(batch, sites) -> Mapping[str, Tensor]`；
- `intervene(base, source, spec) -> ForecastDistribution`。

该协议服务于静态结构类型检查；Python 运行时不会自动校验实现，且
`isinstance(obj, ForecastModelAdapter)` 会因协议未启用 runtime checking 而抛出
`TypeError`。它不提供真实 adapter、hook 注册、训练/冻结操作、设备搬移、异常恢复或性能保证。
`model_hash` 和 `is_frozen` 是实现返回的声明，协议不会验证哈希内容或模型存储；输入输出的业务
正确性、干预语义和无泄漏性质仍需具体实现的独立测试。

## 9. JSON Schema 与 Arrow Schema 合同

持久化 Pydantic 模型可通过 `model_json_schema()` 产生 JSON Schema。严格类型、固定
`schema_version`、`extra="forbid"` 和每个模型的字段约束都是合同的一部分；JSON round-trip
必须重新验证，不能绕过模型直接接受未知字段。

Arrow 工厂返回如下精确、有序 Schema；括号内 `nullable` 表示可空：

- `metrics_by_regime_schema()`：`experiment_id:string`、`run_id:string`、
  `split:string`、`metric:string`、`value:float64`，以及可空的 `regime:string`、
  `horizon:int32`、`concept:string`。
- `predictions_schema()`：`window_id:string`、`split:string`、
  `forecast_time:timestamp[us, tz=UTC]`、`horizon:int32`、`target:string`、
  可空 `y_true:float64`、`mean:float64`、可空 `scale:float64`。
- `intervention_pairs_schema()`：`schema_version:string`、`pair_id:string`、
  `partition:string`、`base_window_id:string`、`source_window_id:string`、
  `concept_name:string`、`regime_relation:string`、`matching_distance:float64`、
  `concept_delta:float64`，全部不可空。

每个 Schema 的元数据精确包含 `contract_schema_version` 和相应 `schema_name`。
`validate_arrow_schema(actual, expected, schema_name=...)` 要求两者都是 `pyarrow.Schema`，
要求 `schema_name` 为非空字符串，并严格比较字段数、顺序、名称、类型、nullability、
字段元数据和 Schema 元数据；任何差异都失败。

## 10. `ArtifactLayout` 安全规则

`ArtifactLayout` 是严格、冻结的持久化模型，字段为 `schema_version`、`experiment_id`、
`run_id`。两个 ID 必须是安全的单一路径段：非空，不能是 `.`、`..`，不能含 `/`、`\`、NUL、
冒号或 Windows drive。

逻辑根固定为 `artifacts/<experiment_id>/<run_id>`，规范相对项为：

```text
config.yaml
metrics.json
metrics_by_regime.parquet
predictions.parquet
intervention_pairs.parquet
data_manifest.json
environment.txt
git_state.txt
stdout.log
plots
```

`relative_run_root` 和 `required_relative_paths` 只计算逻辑路径，不创建文件。
`validate_relative_path(path)` 只接受未经转换的 POSIX 相对字符串，要求它位于本 run 根内，
并拒绝绝对路径、drive-qualified 路径、反斜杠、空段、`.`/`..` 段、NUL 和冒号。

`resolve_path(filesystem_root, relative_path)` 先做词法校验，再 fail closed 地检查根及候选路径中
已存在的 symlink、junction 或 reparse point；根若存在必须是目录，最终解析路径必须仍在根内。
它也不创建目录或制品。

## 11. 最小合法 `WindowBatch` 可执行示例

下面的 doctest 使用规范字段名 `x_observed_mask`、`y_observed_mask` 和
`input_feature_names`：

```python
>>> from datetime import UTC, datetime
>>> import torch
>>> from tarca.contracts import WindowBatch
>>> x = torch.tensor([[[1.0], [2.0]]])
>>> y = torch.tensor([[[3.0]]])
>>> feature_start = datetime(2026, 1, 1, 0, tzinfo=UTC)
>>> feature_end = datetime(2026, 1, 1, 1, tzinfo=UTC)
>>> prediction_start = datetime(2026, 1, 1, 2, tzinfo=UTC)
>>> batch = WindowBatch(
...     x=x,
...     y=y,
...     observed_covariates=None,
...     known_future_covariates=None,
...     x_observed_mask=torch.ones_like(x, dtype=torch.bool),
...     y_observed_mask=torch.ones_like(y, dtype=torch.bool),
...     observed_covariates_mask=None,
...     known_future_covariates_mask=None,
...     regime=None,
...     window_id=("window-001",),
...     input_feature_names=("signal",),
...     target_names=("target",),
...     observed_covariate_names=(),
...     known_future_covariate_names=(),
...     feature_start=(feature_start,),
...     feature_end=(feature_end,),
...     prediction_start=(prediction_start,),
...     label_end=(prediction_start,),
...     forecast_time=((prediction_start,),),
...     metadata={"source": "documentation-example"},
... )
>>> batch.x.shape
torch.Size([1, 2, 1])
>>> batch.y_observed_mask.tolist()
[[[True]]]
>>> batch.input_feature_names
('signal',)

```

## 12. 后续版本规则

删除字段、重命名字段、改变字段类型/含义、不变量或时间语义都是 breaking change，必须提升 major
版本。新增可选字段或枚举值只有在读写双方仍可互操作时才可视为向后兼容；尤其必须先审计
Pydantic `extra="forbid"` 的影响，因为旧读取器会拒绝它不认识的新字段。迁移必须显式进行，
不能用兼容别名或静默强制转换掩盖版本差异。

## 13. 工程与结论边界

本统一合同只是 Stage 1A 的工程接口，不是 TARCA 方法创新，不构成论文新颖性或有效性主张。
它没有运行实验、没有产生 benchmark 或实验结果，也不产生任何科学结论、因果结论、性能结论或
金融结论。
