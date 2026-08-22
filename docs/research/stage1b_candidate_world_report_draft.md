# Stage1B 外部世界资格审查与候选组合报告

> 状态：`IMPLEMENTED_V1_FAILED_WQ13_UNFROZEN`
> 日期：2026-08-22
> 分支：`codex/stage1b-world-implementation-v1`
> 研究边界：仅运行独立资格流程；未运行 E01/E02，未访问正式数据或正式种子。

> 本文件第 2～8、11 节保留实施前的外部研究与候选审查。v1 的实际执行结果以
> `docs/research/stage1b_world_qualification_report_v1.md` 为准。

## 1. 执行结论

用户已批准 v1 资格方案，实施和完整资格运行均已完成。

实际结果：

1. Interfere 固定提交、MIT 许可证、显式噪声重放和三个候选世界均已实现。
2. 线性控制世界中 VAR 按预期获胜。
3. 网络 CML 主世界的最佳神经候选在三个种子上分别比 VAR 差约 4.06、4.05、4.06 CRPS，主要问题来自 unseen coupling。
4. 生态 LV 主世界的最佳神经候选在三个种子上分别比 VAR 差约 0.018、0.027、0.040 CRPS，主要问题来自 unseen ecology regime。
5. 18 次神经训练均完成并通过模型内部捕获/交换操作；失败不是工程失败，而是神经预测优势不存在。
6. 自动 suite Gate 为 `FAIL`；冻结器拒绝创建 v1；活动指针不存在。

结论：v1 失败证据保留，不得在同一个 v1 中按结果调参重跑。若继续，需要用户另行授权 v2。

## 2. 研究方法

### 2.1 研究问题

本轮围绕五个问题检索：

1. 哪些外部来源有图、lag、regime、干预和精确答案？
2. 哪些来源能复用同一未来噪声形成 Level-3 counterfactual？
3. 哪些世界可以承载趋势、尺度、传播、冲击和状态概念？
4. 哪些世界能继续用于 Stage3–9，而不是只用于预测？
5. 哪些来源存在神经预测优势的外部证据，证据与 TARCA E02 有多接近？

### 2.2 来源范围

核验只使用：

- 论文正文或正式 DOI 页面；
- 官方 GitHub 仓库及固定 commit；
- 官方文档；
- Zenodo/Hugging Face 正式数据页面；
- PyPI 官方包元数据。

共审查 13 个来源条目、20 个以上一手网页/仓库/论文入口。精确 commit、许可证和包版本保存在 `docs/research/stage1b_world_sources_draft.yaml`。

### 2.3 执行边界

- 新建隔离 Conda 环境 `tarca-stage1b-py311`，未修改既有环境；
- 安装并锁定 Interfere `1.0.2` / commit `adfa3f7...`；
- 只生成资格轨迹并运行资格训练；
- 未进行结果后调参；
- 未运行 E01 或 E02；
- 未访问 sealed test 或正式数据；
- 未修改冻结的 Stage0/Stage1A 权威文件或工件。

## 3. TARCA 项目有效性要求

候选主世界必须服务于以下研究链：

| 后续研究 | 世界必须提供的功能 |
|---|---|
| TARCA-C1 / Stage4–6 | horizon 与 lag 独立、true/wrong lag、连续概率效应 |
| TARCA-C2 / Stage3–8 | 多变量传播、联合位置真值可植入、变量/lag/horizon/subspace 可区分 |
| TARCA-C3 / Stage9 | 有解释的 regime、sequential unseen、所有解释组件 zero-refit |
| TARCA-C4 / Gate A/1/2 | source=base、随机概念/site/环境、wrong source/lag、支持集诊断 |
| Stage5 source matching | target concept 不同、non-target 相似、same-regime pair、有共同支持 |
| Stage10/11 | 趋势、尺度、传播、冲击和状态语义可以映射到天气、电力、交通和金融 |

详细准入条件见 `docs/research/stage1b_world_qualification_spec.md`。

## 4. 候选能力矩阵

标记：`Y` = 官方来源直接支持；`C` = 有条件，需要 TARCA 薄适配或进一步只读证明；`N` = 不支持；`U` = 官方材料不足。

| 来源 | 同噪声反事实 | 显式 lag | 图/传播 | regime | TARCA 概念 | source pair | 下游连接 | 神经胜 VAR 证据 | 决定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Interfere SDE | Y | C | Y/C | C | C | C | Y | C | 条件主核心 |
| CausalDynamics | N/U | Y | Y | N/C | C | U | Y | N | graph/lag 组件 |
| DoTime Continuous | Y | C | Y | C | C | C | C | N | Level-3 oracle |
| DoTime discrete suites | N（Level-2） | Y | Y | Y | C | C | C | N | regime/识别辅助 |
| dysts | N | C | C | N | C | N | Y | C | 预测压力测试 |
| Causal Chamber | N | C | Y | C | Y/C | C | Y | N | 物理现实验证 |
| EpiCF-Bench | U | N/U | N/U | C | C | N | Y | C（无 VAR） | 许可证待定的后期验证 |
| TimeGraph | N | Y | Y | N/C | C | N | C | U | 图/缺失压力辅助 |
| ODEBench | N | N | C | N | C | N | C | N | 方程 sanity reference |
| CausalTimePrior | Y/C | Y | Y | Y | C | C | C | U | 仅生成器设计参考 |
| causalflow/CAnDOIT | N | Y/C | Y | C | N/C | N | C | N | 干预式结构发现参考 |

## 5. 逐来源审查

### 5.1 Interfere

官方仓库和文档表明，Interfere 面向动态干预响应预测，包含线性、非线性、混沌、连续、离散、随机和确定性模型，并覆盖金融、生态、生物、神经科学和公共卫生。SDE API 接受调用方提供的 Wiener increments `dW`，因此可以让 factual/counterfactual 使用完全相同的未来随机性。

一手来源：

- [GitHub](https://github.com/djpasseyjr/interfere)
- [仿真与模型列表](https://djpasseyjr.github.io/interfere/simulation/)
- [固定 commit 的 SDE `dW` 接口](https://github.com/djpasseyjr/interfere/blob/adfa3f730019f17c3554dd7e0c181248f785bb8b/interfere/dynamics/base.py)
- [JOSS 论文](https://doi.org/10.21105/joss.09792)

适合 TARCA 的部分：

- `VARMADynamics`、`LinearSDE`、`DampedOscillator`：VAR 控制；
- `Kuramoto`、`StochasticCoupledMapLattice`、`WilsonCowan`：网络传播候选；
- `LotkaVolterraSDE`、`MutualisticPopulation`：生态相互作用候选；
- `sigma`/diffusion：尺度概念；
- drift、growth、mean reversion、external drive：趋势/持续性概念候选；
- coupling/interaction matrix：传播概念候选；
- `SignalIntervention`：外生/冲击候选。

缺口：

- 没有统一的原生 regime-switching contract；
- 没有所有模型通用的显式 lag truth；
- Interfere Benchmark 1.1.1 的数据许可证在已检查官方页面中不明确；
- benchmark JSON 是否保存 future noise 未得到证明；
- 论文没有给出 TARCA 小型 PatchTST/iTransformer 稳定胜 VAR 的直接结果。

决定：作为外部动力学核心进入 `CONDITIONAL_PRIMARY_CORE`，不能整包直接冻结。

### 5.2 CausalDynamics

CausalDynamics 提供真实 causal graph、耦合 ODE/SDE、显式 lag、非线性、噪声、混杂和气候动力学。官方配置直接公开 `time_lag`、`time_lag_edge_probability`、node count、node dimension 和非线性激活。

一手来源：

- [GitHub](https://github.com/kausable/CausalDynamics)
- [论文](https://arxiv.org/abs/2505.16620)
- [文档](https://kausable.github.io/CausalDynamics/)
- [固定配置示例](https://github.com/kausable/CausalDynamics/blob/c0b1def6acc4280b87bfb81f5eaff31809d3455a/config.yaml)
- [Hugging Face 数据](https://huggingface.co/datasets/kausable/CausalDynamics)

适合 TARCA 的部分：

- variable/graph/lag 真值；
- source-target path；
- D=8/16 以上可扩展系统；
- 多种驱动、非线性和噪声；
- 连接交通、天气、传感器和金融溢出概念。

缺口：

- 官方 benchmark 目标是从观察性动态恢复图，不是反事实效应；
- 未发现 TARCA 式同噪声 factual/counterfactual API；
- 没有现成 concept pair 和 regime contract；
- PyPI `causaldynamics==1.0.0` 要求 Python `>=3.10,<3.11`，与 TARCA Python 3.11 冲突。

决定：先作为 graph/lag 来源。只有在隔离环境导出固定数据且能够证明 exact replay/干预语义时，才可升级为主世界组件；不得直接安装进冻结 TARCA 环境。

### 5.3 DoTime

DoTime 提供 temporal SCM、图、K=1..3 lag、非线性机制、干预、regime 和四个冻结套件。论文明确区分：

- continuous suite：共享噪声 Level-3 counterfactual；
- discrete suites：重新采样噪声的 Level-2 interventional twins。

一手来源：

- [GitHub](https://github.com/thummd/dotime)
- [论文及完整参考结果](https://arxiv.org/html/2607.27263)
- [RegimeSwitch Zenodo 记录](https://doi.org/10.5281/zenodo.20846073)
- [Continuous Zenodo 记录](https://doi.org/10.5281/zenodo.20845980)

适合 TARCA 的部分：

- 独立图、lag、识别和 regime 答案；
- 连续套件的共享噪声反事实；
- null-effect 和 non-identifiable 负对照；
- in-support intervention 和 positivity guard；
- checksummed frozen suites。

不能作为 E02 主世界的原因：

- 论文 Table 3 中 PFN 的绝对 RMSE 不稳定优于 VAR；
- Regime/Continuous 上 PFN 的方向优势也没有稳定迁移；
- 其主要阳性结果是同容量 interventional-vs-observational PFN 的方向准确率差，不是 TARCA 所要求的小型神经预测器绝对胜线性基线；
- `dot-Generic-100k` v1.0.0 披露 28.7% zeroed episodes，正式使用必须过滤并记录，不能静默接受。

决定：`ORACLE_AUXILIARY`，不承担神经预测胜 VAR 的主门槛。

### 5.4 dysts

dysts 提供 135 个具名连续动力学系统、公开方程和参数，包含部分 delay differential equations，并有 24 种预测方法的大规模比较。

一手来源：

- [GitHub](https://github.com/GilpinLab/dysts)
- [NeurIPS 2021 论文](https://arxiv.org/abs/2110.05266)
- [Physical Review Research 2023 论文](https://arxiv.org/abs/2303.08011)

适合 TARCA 的部分：

- 提供非线性/混沌预测压力；
- 方程和系统属性透明；
- 外部研究表明模型规模、数据量和领域知识之间存在可分析的预测权衡。

缺口：

- 没有 paired counterfactual；
- 没有统一 regime；
- 多数系统维度低；
- 参数变化不等于 TARCA concept source matching；
- `dysts_data` 仓库许可证未明确。

决定：`FORECAST_STRESS`，不能单独进入主证据。

### 5.5 Causal Chamber

Causal Chamber 是真实的风洞和光洞实验平台，提供物理干预、经过经验验证的 causal graph、OOD、change-point 和 impulse-response 数据。

一手来源：

- [Nature Machine Intelligence 论文](https://doi.org/10.1038/s42256-024-00964-x)
- [数据仓库](https://github.com/juangamella/causal-chamber)
- [Python 包与模拟器](https://github.com/juangamella/causal-chamber-package)

优势：真实物理系统、明确操纵量、CC-BY-4.0 数据许可、与传感器传播/状态变化高度相关。

缺口：无法重放完全相同的物理未来噪声，不是 Level-3 SCM truth，也不能提供神经内部植入位置真值。

决定：`EXTERNAL_REALISM`，用于后期外部有效性，不关闭合成 Gate。

### 5.6 EpiCF-Bench

EpiCF-Bench 使用校准的可微 agent-based model，为 158 个美国县提供 168 天的 factual/counterfactual 疫情轨迹和单/多政策干预。

一手来源：

- [论文](https://doi.org/10.1145/3770855.3817522)
- [GitHub](https://github.com/complex-ai-lab/epi-cf-benchmark)
- [Zenodo v1.1](https://doi.org/10.5281/zenodo.20680219)

优势：真实人口、流动性和疫情记录校准；时间变化政策；分布、calibration 和 CATE 评测；有 Transformer/RNN baseline。

缺口：

- 官方 GitHub 和 Zenodo 当前许可证字段为空；
- 没有 VAR baseline；
- 没有 TARCA graph/lag/shared-noise truth manifest；
- ABM 重生成较重，不能在未过硬件可行性门前执行；
- 政策概念与趋势/尺度/传播只存在间接映射。

决定：`REFERENCE_ONLY_PENDING_LICENSE`；许可证明确后也只作为后期现实反事实验证。

### 5.7 TimeGraph、ODEBench、CausalTimePrior、causalflow

- [TimeGraph](https://github.com/hferdous/TimeGraph)：图、lag、趋势、季节性、缺失和混杂很完整，但没有干预/反事实，作为 graph/missingness stress。
- [ODEBench](https://github.com/GPBench/ODEBench)：63 个低维精确 ODE，适合作为方程 sanity reference；无 TARCA counterfactual/regime，许可证未明确。
- [CausalTimePrior](https://github.com/thummd/CausalTimePrior)：功能接近 TARCA 需要，但生成器/先验本身是研究对象，会重新引入“生成数据决定结论”的风险，只作为设计参考。
- [causalflow/CAnDOIT](https://github.com/lcastri/causalflow)：支持观察性+干预式因果发现，但没有 Level-3 paired effect 协议，只作为结构发现参考。

## 6. 候选世界组合 v1-draft

### 6.1 `CONTROL-LINEAR-EXT`

外部核心：Interfere `VARMADynamics`、`LinearSDE`、`DampedOscillator`。

功能：

- 验证 VAR 实现公平；
- 检查概率头和窗口协议不会人为伤害线性模型；
- 允许 VAR 获胜或持平；
- 不计入“神经必须获胜”的主聚合。

状态：`CONDITIONAL_ELIGIBLE`。需要批准具体模型和参数后冻结。

### 6.2 `PRIMARY-NETWORK-EXT`

外部核心候选：Interfere `Kuramoto`、`StochasticCoupledMapLattice`、`WilsonCowan`。

概念功能：

- coupling/interaction → 跨变量传播；
- external drive/drift → 趋势或持续性；
- diffusion `sigma` → 尺度；
- parameter presets → regime；
- network path → variable truth。

薄适配需求：

- 固定外部模型参数和拓扑；
- 定义 model-native concept mapping；
- 加入预声明 delay buffer 或只选择具有可证明 lag 的配置；
- 保存/重放同一 `dW`；
- 构造 same-regime base/source support；
- 装入 Stage1A `DatasetSpec/DataManifest/WindowBatch`。

状态：`CONDITIONAL_PRIMARY`。在 WQ-03/05/07 的只读设计证明完成前不能冻结。

### 6.3 `PRIMARY-ECO-EXT`

外部核心候选：Interfere `LotkaVolterraSDE`、`MutualisticPopulation`。

概念功能：

- growth/drift → 趋势/持续性；
- diffusion → 尺度；
- interaction coefficients → 跨变量传播；
- parameter regimes → 机制状态；
- shock/signal intervention → 稀疏或外生影响。

状态：`CONDITIONAL_PRIMARY`。需要证明概念可以独立变化、不会同时暗改 non-target concept，并且 D/L/H 可映射到预注册配置。

### 6.4 `PRIMARY-LAG-GRAPH-EXT`

外部核心候选：固定的 CausalDynamics lagged coupled-system 配置。

功能：

- 显式 graph；
- 显式 lag；
- D=8/16 变量规模；
- 非线性和传播路径；
- true/wrong lag。

强制条件：

- 不在 TARCA Python 3.11 环境安装不兼容包；
- 使用隔离环境或官方冻结数据完成只读导出；
- 必须先证明可以定义共享噪声 concept intervention；
- 若无法证明，立即降级为 `GRAPH_AND_LAG_AUXILIARY`，不能冒充主世界。

状态：`CONDITIONAL_COMPONENT`。

### 6.5 `ORACLE-DOTIME`

组合：

- `dot-Continuous-v1`：Level-3 same-noise counterfactual；
- `dot-RegimeSwitch-v1`：regime truth；
- `dot-Identifiability-v1`：null/non-identifiable controls。

功能：独立检查答案和失败边界，不承担神经预测胜 VAR 门槛。

状态：`ORACLE_AUXILIARY`。

### 6.6 `STRESS-DYSTS`

选择固定、许可可复核的方程，不下载许可证不明的 `dysts_data`。用于：

- 非线性/混沌压力；
- 长/短时预测差异；
- 检查结论是否只在单一生成器成立。

状态：`FORECAST_STRESS`。

### 6.7 `REALISM-CHAMBER` 与 `REALISM-EPICF`

- Causal Chamber：物理干预和外部 OOD；
- EpiCF：许可证澄清后作为现实 ABM counterfactual。

二者只进入后期验证，不反向关闭 Stage1B/Stage3–9 Gate。

## 7. 神经学习空间证据

### 7.1 能支持的判断

- Interfere 提供非线性、混沌、耦合和随机系统，并集成 LSTM/NHITS/VAR 等预测方法，说明它适合做多方法干预响应预测比较。
- dysts 的 135 系统、24 方法研究说明在数据量、系统复杂度和模型规模不同的条件下，神经/通用模型确实可能获得预测优势。
- EpiCF-Bench 提供 Transformer/RNN 与 KDE 等方法，但不是 TARCA 的 VAR 比较。
- DoTime 的结构匹配实验表明 interventional training 能稳定改善相对 observational twin 的干预方向准确率，但不是绝对预测误差胜 VAR。

### 7.2 不能支持的判断

当前外部证据不能证明：

- TARCA 的小型 PatchTST/iTransformer 一定胜 VAR；
- 上述优势会在三个 seed、NLL/CRPS/MAE 和 unseen regime 上同时成立；
- 一个特定 Interfere 系统族已经满足 Stage3–9 所有资格；
- 通过挑选单个神经友好系统可以替代项目有效性审查。

因此，神经优势证据评级为：

| 候选 | 外部证据等级 | 解释 |
|---|---|---|
| Interfere nonlinear families | 中 | 动力学和预测框架合适，但无 TARCA 直接结果 |
| dysts selected systems | 中 | 有大规模预测研究，但缺 TARCA 干预/图/regime |
| EpiCF-Bench | 弱到中 | 有神经 baseline，但没有 VAR 和 TARCA truth |
| DoTime | 不适合作绝对预测优势证据 | 已公开结果不支持 PFN 稳定绝对胜 VAR |
| CausalDynamics | 未知 | 论文评测 causal discovery，不是 forecasting |

## 8. 防止挑数据的执行规则

1. 先冻结 WQ-01～WQ-11，再看模型成绩；
2. 先冻结 candidate family，再冻结具体系统；
3. 系统级、参数级和 seed 级 TEST 都必须整体保留；
4. 控制、主机制、oracle、stress、realism 分表报告；
5. 不把 VAR 控制世界混入 neural-win 聚合；
6. 不因一个系统失败而删除它；
7. 不因一个系统成功就代表整个 family；
8. 任何替换必须形成 v2、保留 v1、由用户授权；
9. E02 失败后不能用新的数据世界给结果补票；
10. 所有 failed seed、divergence、clipping、zero episode 和 pair-support failure 必须保留。

## 9. 流程执行记录

| 流程 | 本轮完成内容 | 状态 |
|---|---|---|
| 收集外部世界 | 检索论文、仓库、文档、数据存档 | COMPLETED |
| 登记来源 | 固定 commit、版本、许可证和哈希 | COMPLETED |
| 分配职责 | CONTROL 与两个 PRIMARY 世界 | COMPLETED |
| 检查概念 | persistence/growth/scale/propagation/shock/regime | COMPLETED |
| 检查 horizon × lag | 环图最短路径 lag 与三个跨度组 | COMPLETED |
| 检查变量传播 | 图、路径和冲击传播测试 | COMPLETED |
| 检查预测相关性 | 三种子、CRPS/NLL/MAE、整轨迹 bootstrap | COMPLETED_FAIL |
| 检查 pair | 同初值、同未来噪声 factual/counterfactual | COMPLETED |
| 检查负对照 | 结构能力和线性控制 | COMPLETED |
| 检查 regime | QUAL_SEEN 与 QUAL_UNSEEN | COMPLETED_FAIL_WORST_REGIME |
| 检查下游对应 | 网络/生态/金融映射 | COMPLETED |
| 组成套件 | stage1b-worlds-v1 | COMPLETED |
| 检查神经证据 | 18 次训练、3 seeds、2 architectures | COMPLETED_FAIL |
| 形成实施报告 | v1 资格实施报告 | COMPLETED |
| 用户批准 | 用户已授权 v1 实施 | COMPLETED |
| 冻结 | 自动 Gate FAIL，冻结器拒绝 | REJECTED_UNFROZEN |

## 10. 后续需要的新决定

v1 已完成且失败，不再等待 v1 批准。

如果继续，需要用户明确授权创建 v2。v2 必须：

- 保留 v1 配置、收据和失败报告；
- 使用新版本目录和活动指针规则；
- 更换失败世界或外部生成器，而不是在 v1 内按结果微调；
- 继续先过项目有效性门，再运行相同神经余量 Gate；
- 仍不运行 E01/E02，直到 Stage1B 有合法冻结版本。

## 11. 一手来源索引

1. [Interfere GitHub](https://github.com/djpasseyjr/interfere)
2. [Interfere JOSS](https://doi.org/10.21105/joss.09792)
3. [Interfere simulation docs](https://djpasseyjr.github.io/interfere/simulation/)
4. [DoTime GitHub](https://github.com/thummd/dotime)
5. [DoTime paper](https://arxiv.org/abs/2607.27263)
6. [DoTime Continuous suite](https://doi.org/10.5281/zenodo.20845980)
7. [DoTime RegimeSwitch suite](https://doi.org/10.5281/zenodo.20846073)
8. [CausalDynamics GitHub](https://github.com/kausable/CausalDynamics)
9. [CausalDynamics paper](https://arxiv.org/abs/2505.16620)
10. [CausalDynamics dataset](https://huggingface.co/datasets/kausable/CausalDynamics)
11. [dysts GitHub](https://github.com/GilpinLab/dysts)
12. [dysts NeurIPS paper](https://arxiv.org/abs/2110.05266)
13. [dysts forecasting comparison](https://arxiv.org/abs/2303.08011)
14. [Causal Chamber paper](https://doi.org/10.1038/s42256-024-00964-x)
15. [Causal Chamber datasets](https://github.com/juangamella/causal-chamber)
16. [EpiCF-Bench paper](https://doi.org/10.1145/3770855.3817522)
17. [EpiCF-Bench GitHub](https://github.com/complex-ai-lab/epi-cf-benchmark)
18. [EpiCF-Bench Zenodo](https://doi.org/10.5281/zenodo.20680219)
19. [TimeGraph GitHub](https://github.com/hferdous/TimeGraph)
20. [TimeGraph paper](https://doi.org/10.1145/3711896.3737439)
21. [ODEBench GitHub](https://github.com/GPBench/ODEBench)
22. [ODEFormer/ODEBench paper](https://arxiv.org/abs/2310.05573)
23. [CausalTimePrior GitHub](https://github.com/thummd/CausalTimePrior)
24. [causalflow/CAnDOIT GitHub](https://github.com/lcastri/causalflow)
