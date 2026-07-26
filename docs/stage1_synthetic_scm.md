# Stage 1B 合成 Regime-Switching SCM 设计与接口映射

## 1. 文档状态与权威顺序

本文定义 TARCA Stage 1B 的工程实现边界。它只覆盖可复现的合成
regime-switching SCM、成对反事实 oracle、缺失机制、连续时间切分、持久化、
校验与 E01 工程烟测，不产生正式科研结论。

实现按以下权威顺序解释要求：

1. `docs/preregistration_v0.md`、`docs/assumption_ledger.md`、
   `docs/novelty_claims.md`、`docs/terminology.md`；
2. `docs/stage0_scope.md`、Stage 0 冻结证据和现有测试；
3. `src/tarca/contracts/`、`tests/contracts/` 与
   `docs/stage1_unified_data_contract.md`；
4. `docs/TARCA_项目计划书.md`；
5. `docs/TARCA_具体实施计划.md`；
6. `D:\TARCA_Stage1_合成_Regime_Switching_SCM.md`。

以下文件也是开始实施前必须完整读取和核验的输入，但不改变上述权威优先级：

- `README.md`；
- `docs/literature_audit_log.md`；
- `artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md`；
- `artifacts/stage1/contracts/STAGE1_CONTRACT_IMPLEMENTATION_REPORT.md`；
- `pyproject.toml` 与完整 `uv.lock`。

用户已明确批准两项最小兼容修复：

- 把 Stage 1A 的范围测试升级为 Stage 1B 范围测试，只放行
  `src/tarca/data/synthetic/`，继续禁止模型、训练、干预、定位、鲁棒性等后续模块；
- 不修改统一契约。规范 `DataManifest` 仍是唯一数据清单契约；合成生成器特有信息
  通过严格、私有的 provenance/真值 sidecar 表达。

## 2. 范围边界

### 2.1 本阶段实现

- 持久一阶 Markov regime 链；
- 相互隔离的 trend 与 scale 潜在概念；
- 稳定、非线性、异方差、带外生输入与稀疏冲击的 VAR 型 SCM；
- 显式 `FutureNoiseBank` 和成对 factual/counterfactual 重放；
- `none`、`mcar`、`block` 三种 future-blind 缺失机制；
- 连续 `60%/20%/10%/10%` 时间切分和 train-only 标准化；
- 现有 `WindowBatch`、`DataManifest` 的消费与私有 Arrow 持久化；
- 可复现 hash、checksum、真值 sidecar、校验和 CPU-only E01 工程烟测；
- 两个薄 CLI：数据构建与 oracle 烟测。

### 2.2 明确不实现

- 预测器训练、模型 adapter 的具体实现或 checkpoint；
- activation cache、内部激活干预、机制定位；
- PLOT、OT、DAS、DRO 或鲁棒性算法；
- 真实数据、金融数据下载、回测；
- 正式 E01 阈值冻结、Gate A 或其他科研 Gate；
- 对一般非线性多步模型的闭式因果效应声明。

不会修改 `src/tarca/stage0/`、`src/tarca/contracts/`、Stage 0 冻结证据、
冻结研究文档、`pyproject.toml` 或 `uv.lock`。

## 3. 统一契约映射

### 3.1 运行时窗口

生成器直接构造 `tarca.contracts.WindowBatch`，不复制或继承该类：

- `x`：标准化后、缺失位置以有限填充值表示的历史窗口 `[B,L,D]`；
- `y`：标准化后的未来目标 `[B,H,D]`；
- `observed_covariates`：历史外生输入 `[B,L,U]`；
- `known_future_covariates`：预测期已知外生输入 `[B,H,U]`；
- 四种 mask：与对应张量同 shape、同 device 的 `torch.bool`；
- `regime`：shape `[B]`，表示每个窗口历史最后一步、即预测原点的 regime；
- `window_id`：四个物理 split 间也全局唯一；
- 时间字段：UTC、严格递增，且窗口完全位于自己的物理 split；
- `metadata`：只放可 JSON 化的轻量标量，不放 Tensor 或完整真值。

完整 regime path 和概念 path 只存在于 `truth.npz`，不会塞入
`WindowBatch.regime` 或 `metadata`。

### 3.2 三分区规范清单与四个物理切分

现有 `DataManifest` 要求恰好包含 `TRAIN`、`VALIDATION`、`TEST`。映射固定为：

| 物理切分 | 规范分区 |
|---|---|
| `train` | `SplitPartition.TRAIN` |
| `validation` | `SplitPartition.VALIDATION` |
| `test_seen_regime` | `SplitPartition.TEST` 的第一部分 |
| `test_unseen_regime` | `SplitPartition.TEST` 的第二部分 |

两个 test 物理切分必须互不相交。规范 TEST 的 `count` 是二者窗口数之和；
规范 TEST 的 `split_hash` 是按固定顺序对
`{"test_seen_regime": <hash>, "test_unseen_regime": <hash>}` 的规范 JSON 做
SHA-256，因而不依赖文件遍历顺序。

### 3.3 复合 `manifest.json`

`manifest.json` 是严格私有包装对象，不是第二个统一契约：

```json
{
  "data_manifest": {
    "schema_version": "1.0.0",
    "dataset_name": "synthetic_easy",
    "dataset_version": "sha256-prefix-version",
    "dataset_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "splits": [
      {
        "schema_version": "1.0.0",
        "partition": "train",
        "split_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "count": 1
      },
      {
        "schema_version": "1.0.0",
        "partition": "validation",
        "split_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "count": 1
      },
      {
        "schema_version": "1.0.0",
        "partition": "test",
        "split_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "count": 2
      }
    ],
    "window_contract": {
      "schema_version": "1.0.0",
      "history_length": 48,
      "horizon": 12,
      "input_feature_names": ["x0", "x1", "x2", "x3"],
      "target_names": ["x0", "x1", "x2", "x3"],
      "observed_covariate_names": ["u0"],
      "known_future_covariate_names": ["u0"],
      "timezone": "UTC",
      "missingness_protocol": "none"
    },
    "source_description": "TARCA Stage 1B deterministic synthetic SCM",
    "created_at": "2026-07-26T00:00:00Z"
  },
  "synthetic_provenance": {
    "config_hash": "sha256:...",
    "root_seed": 20260725,
    "root_entropy": 20260725,
    "random_streams": {},
    "parameter_summary": {},
    "physical_splits": {},
    "train_scaler_fit_interval": {},
    "seen_unseen_parameter_difference": {},
    "software_versions": {},
    "git_commit": "...",
    "generated_at": "...",
    "research_status": "ENGINEERING_ARTIFACT"
  }
}
```

上例的 `data_manifest` 是嵌套 JSON object，不是 JSON 字符串；真实值直接来自
`DataManifest.model_dump(mode="json")`。

读取时先以严格、frozen、extra-forbid 的私有 Pydantic 模型验证包装和
`synthetic_provenance`，再以现有 `DataManifest.model_validate` 验证
`data_manifest`。私有模型不从 `tarca.contracts` 导出，也不改变其 schema。

### 3.4 私有 Arrow schema

每个物理 split 单独保存一个 Arrow IPC 文件。私有 schema 的字段名、顺序、类型和
nullability 固定如下：

| # | 字段 | Arrow 类型 | nullable |
|---:|---|---|---|
| 1 | `window_id` | `string` | 否 |
| 2 | `x` | `large_list<large_list<float64>>` | 否 |
| 3 | `y` | `large_list<large_list<float64>>` | 是 |
| 4 | `observed_covariates` | `large_list<large_list<float64>>` | 是 |
| 5 | `known_future_covariates` | `large_list<large_list<float64>>` | 是 |
| 6 | `x_observed_mask` | `large_list<large_list<bool>>` | 是 |
| 7 | `y_observed_mask` | `large_list<large_list<bool>>` | 是 |
| 8 | `observed_covariates_mask` | `large_list<large_list<bool>>` | 是 |
| 9 | `known_future_covariates_mask` | `large_list<large_list<bool>>` | 是 |
| 10 | `regime` | `int64` | 是 |
| 11 | `input_feature_names` | `large_list<string>` | 否 |
| 12 | `target_names` | `large_list<string>` | 否 |
| 13 | `observed_covariate_names` | `large_list<string>` | 否 |
| 14 | `known_future_covariate_names` | `large_list<string>` | 否 |
| 15 | `feature_start` | `timestamp[us, tz=UTC]` | 否 |
| 16 | `feature_end` | `timestamp[us, tz=UTC]` | 否 |
| 17 | `prediction_start` | `timestamp[us, tz=UTC]` | 否 |
| 18 | `label_end` | `timestamp[us, tz=UTC]` | 否 |
| 19 | `forecast_time` | `large_list<timestamp[us, tz=UTC]>` | 否 |
| 20 | `metadata_json` | `string` | 否 |

`contract_schema_version` 只放在 schema metadata，不重复成数据列。metadata 键按
`contract_schema_version`、`physical_split`、`tensor_dtype`、`x_shape`、`y_shape`
的固定顺序写入；shape 使用规范 JSON。行按预测原点升序写入，`metadata_json` 使用
排序键和紧凑分隔符。Arrow IPC 固定为无压缩、Metadata V5、每个物理 split 一个
record batch，不使用 dictionary encoding；这些规则在同一锁定环境内固定 split 文件字节。

round-trip 后必须重新构造 `WindowBatch` 触发统一契约校验。该 schema 不导出到
`tarca.contracts`，也不冒充预测或干预长表 schema。

## 4. 数学模型与时间语义

### 4.1 Regime 链

\[
r_{t+1}\sim\operatorname{Categorical}(P_{r_t,:}),\qquad P\in[0,1]^{R\times R}.
\]

`P` 必须是有限的 float64 方阵、元素非负、每行在严格容差内和为 1。
初始分布必须显式给出或由平稳分布计算。采样函数只消费预生成的
`uniforms`，不在内部调用 RNG。完整 `regime_sequence` 被保存。

### 4.2 潜在概念

\[
C^{trend}_{t+1}=a_{r_t}C^{trend}_t+\xi_t,\qquad |a_r|<1,
\]

\[
C^{scale}_{t+1}=b_{r_t}C^{scale}_t+\omega_t,\qquad |b_r|<1.
\]

两个状态及其 innovations 独立演化。`C^{scale}` 本身可为任意有限实数；
必须为正的是它经 scale response 后得到的观测噪声尺度。
`concept_overlap` 只改变 trend/scale 对观测维度的载荷重叠，不让两个潜变量互读。

在预测原点 `t` 的概念替换只替换指定概念的当前值。之后 factual 与
counterfactual 使用相同结构方程、regime 和未来 innovations 自然演化；
不会覆盖整条未来概念路径。trend 替换不得改变 scale 状态或 innovations，
scale 替换不得改变 trend 状态或 innovations。source 等于 base 时轨迹必须相同。

### 4.3 观测 SCM

\[
X_{t+1}=
A^{(r_t)}X_t+
\gamma_{r_t}\tanh(W^{(r_t)}X_{t-\delta_x})+
M^{trend}C^{trend}_{t-\delta_c}+
B^{(r_t)}U_t+
s_{r_t}(C^{scale}_t)\epsilon_t+
S_t.
\]

\[
s_r(c)=\operatorname{softplus}(
\operatorname{base\_log\_scale}_r+
\operatorname{scale\_loading}_r c
)+\epsilon_{\mathrm{floor}}.
\]

`U_t` 是保存的可观测外生输入，`S_t` 是按 Bernoulli 发生率和配置幅度生成的
稀疏冲击。二者都不是本阶段可干预概念。

`delta_c` 是 trend 概念的 causal lag，`h` 是从 1 开始的 forecast horizon：

- `delta_c=0` 的最早直接效应在 `h=1`；
- 一般最早直接效应在 `h=delta_c+1`；
- delay recovery 使用 `estimated_delay=h_peak-1`。

实现、manifest、指标和报告必须分别存储 `h` 与 `delta_c`，不得把二者都叫
“timestep”。

### 4.4 参数稳定性

每个 regime 的候选稀疏 VAR 矩阵只在参数生成阶段处理一次：

1. 计算原始谱半径 `rho_raw`；
2. 当 `rho_raw>0.85` 时乘以明确因子 `0.85/rho_raw`；
3. 保存 `rho_raw`、缩放因子和最终 `rho_final`；
4. rollout 中不得再次 clip 或重缩放；
5. rollout 后检查所有输出有限，发现 NaN/Inf 立即失败。

谱半径条件只是 MVP 数值稳定性设计，不被描述为完整非线性系统的全局稳定性证明。

## 5. 随机性与确定性

根种子只通过：

```python
root = numpy.random.SeedSequence(root_seed)
children = root.spawn(10)
```

派生以下十个名称和顺序固定的独立子流：

1. `regime_transitions`；
2. `trend_innovations`；
3. `scale_innovations`；
4. `exogenous_variables`；
5. `observation_innovations`；
6. `sparse_shocks`；
7. `missingness`；
8. `parameter_generation`；
9. `counterfactual_mc_bank`；
10. `random_concept_negative_control`。

manifest 记录根 entropy、每个 child 的 spawn key 和可重建标识。禁止
`root_seed + worker_id`、模块级可变 RNG、`np.random.seed` 和 `RandomState`。

所有随机数组在 rollout 前生成。相同初始状态、参数、regime path/transition
uniforms、概念 innovations、外生路径、观测 innovations、冲击路径必须在同一锁定环境中
得到 bitwise-identical 核心数组。

MC bank 子流再通过 `SeedSequence.spawn(mc_samples)` 为样本派生独立子序列。
每个样本的 factual/counterfactual 共用同一个 `FutureNoiseBank`；不同 MC 样本独立。

## 6. 模块职责

### 6.1 `regimes.py`

- 严格校验 transition matrix 和 initial distribution；
- 计算平稳分布、采样 regime sequence、统计驻留时间；
- 构造与 regime path 等长的参数 schedule；
- 管理十流 registry；
- 生成并记录 seen/unseen 参数调度。

### 6.2 `latent_concepts.py`

- 纯函数生成 trend/scale 状态与真值；
- 校验 regime-specific AR 系数；
- 实现只替换预测原点当前概念值的干预；
- 提供正值、有限的 softplus scale response。

### 6.3 `nonlinear_var.py`

- 生成并校验 regime-specific SCM 参数；
- 生成稀疏稳定矩阵并记录缩放证据；
- 提供无采样 deterministic transition 与 factual rollout；
- 保存 `true_graph`、参数、初始历史和完整噪声真值。

### 6.4 `counterfactual_oracle.py`

显式 immutable `FutureNoiseBank` 至少包含每个 MC 样本的：

- future regime transition uniforms 或固定 future regime path；
- trend、scale 和 observation innovations；
- exogenous path/innovations；
- shock path/innovations。

模块负责 factual/counterfactual 共噪声重放、trend/scale intervention、mean/std/quantile
effect、effect signature、delay recovery、解析 sanity 子例、wrong-delay、
wrong-scale-response 和 random-concept control。oracle 自身不调用 RNG。

### 6.5 `missingness.py`

- 生成 `none`、MCAR、block bool mask；
- 只消费独立随机数组、配置和当前/过去允许状态；
- 不读取未来观测或标签；
- 以有限填充值应用 mask，不修改完整 truth。

### 6.6 `dataset_builder.py`

- 以 strict/frozen/extra-forbid Pydantic 模型解析 YAML；
- 解析 `true_delay` 标量或闭区间 `[min_delay,max_delay]`；
- 生成参数、burn-in、完整事实轨迹与真值；
- 构造连续四切分和 split 内窗口；
- 仅在 train 拟合 scaler，并复用到其余 split；
- 构造现有 `WindowBatch` 和 `DataManifest`；
- 计算 config/data/split/file hash；
- 原子写入所有数据产物并支持严格 round-trip。

仓库没有可替代的 Stage 1 配置契约，因此该 Pydantic 模型是生成器私有解析层，
不是跨模块统一契约。

### 6.7 `validation.py`

- 校验 shape、dtype、有限性、谱半径、scale 正值；
- 校验 concept 隔离、paired-noise identity 和 factual self-replay；
- 审计 split、window、scaler 和 future leakage；
- 校验 checksum、manifest、truth 与 Arrow round-trip；
- 执行 analytic sanity、delay recovery、MC convergence 与 E01 汇总。

## 7. 配置语义

三份 YAML 的以下核心字段和值必须原样保留：

| 配置 | D/L/H/R | true_delay | seed | burn-in/steps | MC/pairs |
|---|---|---|---|---|---|
| easy | 4/48/12/2 | 2 | 20260725 | 256/4096 | 256/16 |
| medium | 8/96/24/3 | `[1,4]` | 20260726 | 384/8192 | 128/8 |
| hard | 16/192/48/4 | `[0,8]` | 20260727 | 512/12288 | 64/4 |

`true_delay` 列表总是闭区间，由 `parameter_generation` 子流为每个 regime
解析具体 delay，并写入 provenance 与 `truth.npz`。它不是“两个 regime 各一个值”。

所有矩阵维度、概率、AR 系数、delay、missing rate、MC/pair 数、burn-in、总长度、
窗口长度和 horizon 都 fail-fast。未知配置字段失败；不会用 `abs`、排序、全局 clip、
自动归一化错误转移矩阵等方式修补错误配置。

最后 10% 的 `test_unseen_regime` 保留可审计 regime 标签，但使用只由配置确定的
parameter shift；前 90% 使用 seen 参数。shift 可作用于 `A_r`、scale response、
resolved delay 或噪声设置，具体差异在运行前由配置固定并完整记录，不能看结果后调整。

## 8. 切分、标准化与缺失

去除 burn-in 后，按时间顺序取：

- `[0,60%)` train；
- `[60%,80%)` validation；
- `[80%,90%)` test seen；
- `[90%,100%]` test unseen。

边界使用确定性的整数索引规则并写入 provenance。每个 split 只在自己的时间块内创建
`L` 历史加 `H` 未来的窗口，任何窗口都不得跨界。`window_id` 包含 dataset identity、
物理 split 和预测原点索引，确保全局唯一。

均值、标准差及零方差处理参数只在 train 的完整允许区间拟合。validation/test
只能读取保存的 train statistics。零方差使用预先固定的 epsilon 规则并记录，不能读取
后续 split 决定规则。

完整 `x_complete` 永远无缺失。mask 的 `True` 表示已观测；缺失位置在对外张量中使用
有限 `0.0` 标准化填充值，统一契约通过 bool mask 区分。改变未来随机数组不能改变已生成
前缀 mask。

## 9. 数据产物与指纹

用户指定安全输出目录内至少生成：

```text
config_resolved.yaml
manifest.json
checksums.json
truth.npz
windows_train.arrow
windows_validation.arrow
windows_test_seen_regime.arrow
windows_test_unseen_regime.arrow
normalization.json
```

`truth.npz` 至少包含：

```text
x_complete
trend
scale
regime_sequence
exogenous
observation_noise
trend_noise
scale_noise
shock_sequence
missing_mask
resolved_true_delay
```

还可包含重放所必需的初始历史、参数数组与 `true_graph`，但不包含 pickle/object array。
加载时使用 `allow_pickle=False`。

- `config_hash`：对解析后配置的规范 JSON 做 SHA-256；
- physical split hash：对确定性 Arrow IPC 内容做 SHA-256；
- canonical TEST hash：按第 3.2 节聚合两个 test hash；
- `dataset_hash`：对 config hash、核心 truth 数组的 dtype/shape/bytes、normalization
  和四个有序 physical split hash 的规范身份记录做 SHA-256；
- `checksums.json`：记录每个落盘文件的 SHA-256，不把自身加入自身造成循环。

生成时间、输出路径和临时目录不进入 dataset identity。相同配置/锁定环境两次生成的
核心 truth、config hash、split hash 和 dataset hash 必须相同；改变 root seed 必须改变数据。

持久化先写输出目录的安全 sibling staging 目录，fsync/关闭文件后再原子替换。
拒绝绝对路径逃逸、`..`、已有祖先 symlink/junction/reparse escape 和越出批准根目录的
resolved path。失败时清理 staging，且不得留下成功状态报告。

生成数据、缓存和烟测输出不跟踪进 Git；仅提交本设计和小型实施报告。

## 10. 测试与验证策略

实现严格按 RED → GREEN → REFACTOR 前进。每一模块先写能因缺失行为而失败的测试，
观察预期失败后只写最小实现，再运行本模块、全部既有 synthetic 测试和合同回归。

关键测试包括：

- regime 转移校验、复现、驻留时间、schedule；
- trend/scale 隔离、source=base identity、非法 AR；
- deterministic transition、手算线性子例、稳定半径和 factual replay；
- no-intervention/source=base 零效应、共享 noise bank、h/delta 恢复；
- scale variance sanity、wrong-delay、wrong-scale、random concept；
- quantile shape/单调性和总体 MC convergence 趋势；
- missingness 统计、future blindness、truth 不变；
- split 不重叠、window 不跨界、train-only scaler、unseen shift 隔离；
- manifest/Arrow/truth/checksum 严格 round-trip；
- CLI help、非零失败码、路径安全、CPU-only 和失败清理；
- Stage 0、统一契约、coverage、Ruff、format、pre-commit 全回归。

解析 sanity 固定为单 regime、`nonlinear_strength=0`、`shock_rate=0`、短 horizon、
已知 trend loading 与 scale response；只验证该子例的解析均值和条件方差。

MC convergence 使用固定高样本参考和预先固定的总体误差趋势，不要求每个相邻样本量
严格单调，避免把采样波动误判为失败。

## 11. E01 工程烟测

默认只运行 easy，CPU-only：

- pairs 不超过 16；
- MC samples 不超过 256；
- 比较 `[32,64,128,256]`；
- 单命令预计附加峰值内存不超过 4 GiB；
- 输出不超过 2 GiB；
- 不执行超过 30 分钟的正式 sweep。

固定比较 correct SCM、wrong-delay SCM、wrong-scale-response SCM、
random-concept control。报告：

- mean effect RMSE；
- scale/std effect RMSE；
- quantile effect RMSE；
- delay absolute error；
- MC estimator variance 与 convergence curve；
- correct 与 wrong/random 的 signature distance；
- runtime、峰值内存估计、输出大小；
- root seed、Git commit、config hash、data hash。

最低工程通过条件在运行前固定：

- paired-noise invariants 和 factual replay 全部通过；
- analytic sanity 与 MC 在预设容差内一致；
- 大样本总体误差低于小样本；
- wrong SCM 与 random concept 在固定 sanity case 上更差；
- easy true delay 按 `delta=h_peak-1` 恢复；
- 无 NaN/Inf、泄漏、契约或 checksum 违规。

状态只能是 `ENGINEERING_SMOKE_PASS` 或 `ENGINEERING_SMOKE_FAIL`。
不得输出 `E01_FORMAL_PASS`、Gate A 通过或 TARCA 科学假设已验证。

## 12. 资源、安全与失败策略

- 仅使用现有锁文件中的 NumPy、SciPy、Pydantic、PyArrow、PyYAML 和 CPU PyTorch；
- 不新增 Tigramite、TimeGraph、statsmodels、hmmlearn、Dynamax、JAX 等运行时依赖；
- 不访问网络、不下载模型/数据、不调用外部 LLM/API、不使用 CUDA；
- 测试全部使用 `tmp_path`，不写正式 `data/processed`；
- 配置、路径、数组、schema、checksum 和报告输入均在边界校验；
- 公共 API 提供 shape、dtype、索引和随机性 docstring；
- 状态、参数、noise bank、结果使用 frozen record/read-only array 语义；
- 不吞异常，错误包含字段名和实际/期望 shape；
- 任一关键校验失败时实施状态为 `PARTIALLY_COMPLETED`，不得伪造成功报告。

## 13. 外部设计来源与许可证边界

只参考公开概念和官方说明，不复制实现：

- Hamilton (1989) 的 Markov-switching 建模思想：
  <https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html>；
- Regime-PCMCI 对持久 regime 下时变因果结构的研究背景：
  <https://arxiv.org/abs/2007.00267>；
- Tigramite structural causal process 的广义加性时序设计说明：
  <https://jakobrunge.github.io/tigramite/_modules/tigramite/toymodels/structural_causal_processes.html>；
- Tigramite 官方仓库标明 GPL-3.0：
  <https://github.com/jakobrunge/tigramite>；
- TimeGraph 的非线性、异方差、趋势与缺失合成基准思路：
  <https://arxiv.org/abs/2506.01361>；
- TimeGraph 仓库代码为 MIT、数据为 CC BY 4.0：
  <https://github.com/hferdous/TimeGraph>；
- NumPy `SeedSequence` 官方文档：
  <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html>。

本项目代码为从数学规格独立实现的 NumPy/SciPy 生成器，不导入、包装或复制上述仓库
源码，也不声称这些工作直接实现了 TARCA paired regime-switching oracle。合成生成器是
后续验证基础设施，不作为 TARCA 方法创新。

### 13.1 文献直接支持的设计背景

- Hamilton 支持持久离散状态切换的时间序列建模背景；
- Regime-PCMCI 支持 regime 下因果结构、滞后和效应可变化的研究背景；
- Tigramite 文档支持用显式滞后和加性噪声描述时序结构过程的设计背景；
- TimeGraph 支持在合成基准中覆盖非线性、异方差、趋势和缺失的设计背景；
- NumPy 文档支持用 `SeedSequence.spawn` 构造可复现、可重建的子流。

这些来源不直接定义 TARCA 的 paired oracle、四切分映射或 E01 判据。

### 13.2 本项目工程决策

- 谱半径上限 `0.85`、softplus scale response 和有限值 fail-fast；
- 十个固定命名随机流及其顺序；
- `h=delta+1` 与 `estimated_delay=h_peak-1` 的工程索引映射；
- `60/20/10/10` 四个物理切分映射到三分区 `DataManifest`；
- 复合 manifest、私有 Arrow schema、hash 和 atomic-write 规则；
- easy/medium/hard 的资源上限与 E01 工程状态词。

这些值和映射来自本项目规格，不描述为外部文献结论。

### 13.3 仍待正式 E01 验证的假设

- correct SCM 的效应 signature 在固定 sanity case 上优于 wrong-delay、
  wrong-scale-response 和 random-concept controls；
- easy 配置可可靠恢复 true delay；
- MC 样本量增加时总体估计误差下降；
- 当前工程默认能在正式冻结的阈值与统计协议下保持相同结论。

本阶段只运行 engineering smoke；正式阈值、重复数、统计检验和失败判据仍须按预注册
在第一次正式实验前冻结。

## 14. 下一阶段入口

Stage 1B 结束后只提供以下入口，不提前实现：

```text
E01 正式协议冻结
→ 正式 synthetic 数据生成
→ 基础预测器和统一 ForecastDistribution
```

在正式协议、数据和基础预测器就绪前，不进入机制定位、内部激活干预或金融实验。
