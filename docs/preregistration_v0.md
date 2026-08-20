# TARCA 预注册 v0

> preregistration_id：`TARCA-PREREG-0.1.0`
> protocol_id：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 冻结日期：2026-08-20
> 适用范围：TARCA 的首轮 claim-bearing 合成实验、机制定位和跨状态验证。金融仅为后期压力测试。

## 1. 研究问题

### RQ1（主要）

冻结的多步概率时序预测器中，高层概念干预能否由内部交换干预近似，并且 forecast horizon 与 causal lag 能否作为独立实验轴被识别？

对应 `TARCA-C1`。

### RQ2（主要）

在具有联合植入真值的预测模型中，能否恢复 variable × causal lag × forecast horizon × constrained subspace，而不只是重复 PLOT 已有的渐进层/时间定位？

对应 `TARCA-C2`。

### RQ3（主要）

predictor、位置、映射、normalizer 和阈值全部冻结时，解释能否在 sequential unseen regimes 上 zero-refit 保持较低最坏抽象误差？

对应 `TARCA-C3`。

### RQ4（强制支撑）

观察到的抽象一致性是否能通过容量限制、随机负对照、held-out pairs、未映射变量 faithfulness 和表示支持检查排除信息注入与空洞映射？

对应 `TARCA-C4`。

### RQ5（探索性验证层）

仅在 Gate A、1、2 和非金融跨域检查通过后，研究机制解释在金融压力测试中的失效边界。RQ5 不提供方法新颖性证据，也不能反向关闭前述 Gate。

## 2. 证据等级

### 主证据

- 冻结 predictor；
- paired high-level/low-level interventions；
- held-out intervention pairs；
- 合成 SCM 与植入位置真值；
- sequential unseen regime zero-refit；
- 完整负对照和 anti-injection 检查。

### 次级证据

- 联合训练或 abstraction-regularized forecasting；
- 真实数据上的弱概念和专家解释；
- 金融预测或回测收益。

次级证据不能关闭 Gate A、1 或 2。

## 3. 数据与切分

### 3.1 首要合成数据

| 配置 | D | L | H | regimes | true_delay | SNR | overlap | missing |
|---|---:|---:|---:|---:|---|---|---|---:|
| synthetic_easy | 4 | 48 | 12 | 2 | 2 | high | low | 0.00 |
| synthetic_medium | 8 | 96 | 24 | 3 | 1–4 | medium | medium | 0.05 |
| synthetic_hard | 16 | 192 | 48 | 4 | 0–8 | low | high | 0.15 |

每个配置必须具有真实 graph、regime、shock sequence、causal delay、高层 intervention oracle 和植入位置 manifest。factual/counterfactual 共用未来外生噪声。

### 3.2 时间切分

- TRAIN：连续时间前 60%；
- VALIDATION：随后 20%；
- TEST_SEEN_REGIME：随后 10%，状态组合在 TRAIN 出现；
- TEST_UNSEEN_REGIME：最后 10%，保留状态、机制参数或状态组合。

不随机打乱。标准化、PCA、状态描述、概念阈值和 effect normalizer 只在 TRAIN 拟合；模型和超参数只用 VALIDATION 选择。

### 3.3 intervention-pair 划分

- pair partition 与数据 partition 分开持久化，但同一窗口不得跨 partition；
- TRAIN pairs 用于映射、normalizer 和定位搜索；
- VALIDATION pairs 用于选择 rank、阈值和超参数；
- TEST pairs 仅用于一次最终评价；
- base/source 必须不同、距离有限非负，并记录 SAME/CROSS/UNKNOWN regime relation；
- source=base 只作为独立 sanity-check 集合，不参与拟合。

### 3.4 后续非金融域

Weather、Electricity、Traffic，以及一个 fev-bench 中具有 known-future covariates 的任务。至少两个非金融域通过跨域机制检查后才允许金融压力测试。

## 4. 模型与基线

### 4.1 预测基线

Last value / Seasonal naive、AR/VAR、DLinear。

### 4.2 主神经预测器

- 小型 PatchTST：只支持 layer × time-patch 主证据；
- 小型 iTransformer：支持 variable-token 和跨变量主证据；
- 自定义 `(variable, patch)` 二维 token 植入模型：支持联合位置真值。

首轮主模型固定：`d_model=64`、`n_layers=3`、`n_heads=4`、`dropout=0.1`。主概率头为对角高斯，必须验证 finite mean/scale/log_prob 且 scale 严格为正。

Chronos 类冻结 TSFM 只作为后续有限预测/表示基线，不作为首轮内部干预主模型。

### 4.3 解释和定位基线

- oracle site、随机 site、穷举/Full DAS；
- DAS、Boundless DAS、PLOT、PLOT-guided DAS；
- Integrated Gradients、occlusion、input counterfactual、concept bottleneck、SAE；
- CAE-style abstraction error 与 unmapped-variable faithfulness。

### 4.4 鲁棒性基线

ERM、balanced environment sampling、Group-DRO、DiRoCA-style、随机重加权；FOIL/COGS 只在输入和训练协议可比时报告。

## 5. 允许的调参与模型选择

正式主结果只允许以下验证期网格；不得在 TEST 上扩展：

- learning rate：`{1e-4, 3e-4, 1e-3}`；
- batch size：`{32, 64, 128}`，受内存限制时从大到小尝试并记录；
- weight decay：`{0, 1e-4, 1e-3}`；
- maximum epochs：100；validation early-stopping patience：10；
- patch length：从能整除/覆盖 `L` 的 `{8, 16, 24}` 中选择；
- constrained-subspace rank：`{1, 2, 4, 8, 16}` 且不得超过表示维度；
- OT/DRO 正则与半径：只可在 Stage 6/9 开始前写入版本化 amendment，使用独立 pilot/VALIDATION 冻结。

如果资源 probe 显示完整网格不可行，应先报告资源需求并选择服务器 backend；不得未经 amendment 缩小科学比较。

## 6. 随机性

- 主结果 seeds：`1729, 2718, 3141, 5772, 8111`；
- 最小开发 smoke seed：`1729`，不计入正式主结果；
- 每个正式结果至少 3 个 seed；最终主表使用全部 5 个 seed；
- 数据生成、模型初始化、pair 抽样、OT/DRO 和 bootstrap 使用由 root seed 派生且持久化的独立子流；
- 失败 seed 不得静默删除；仅可因预注册的非有限数值/基础设施故障重跑，并保留 attempt 记录。

## 7. 主要指标

### 7.1 预测

NLL、CRPS（或固定样本近似）、MAE、MSE、coverage、calibration error、每 regime 与 worst-regime 风险。calibration 只在 `fold × horizon × subgroup/regime` 聚合，不作为测试时定位输入。

### 7.2 抽象与反空洞

TII/效应距离、IIC、Cause、Isolation、Completeness、held-out pair generalization、unmapped-variable faithfulness、随机负对照差距、表示支持/偏离诊断。

### 7.3 定位

layer accuracy、patch IoU、variable F1、lag absolute error、horizon accuracy、subspace projection distance、rank recovery、联合真值 exact/partial recovery。

### 7.4 鲁棒与效率

mean/worst-regime abstraction error、平均—最坏前沿、干预次数、定位时间、训练/定位 accelerator hours、峰值内存/显存、缓存大小和相对 Full DAS 成本。

## 8. 负对照

所有 claim-bearing 实验至少包含：

- source=base；
- wrong lag；
- wrong source window；
- random site / random orthogonal subspace；
- random concept / shuffled concept labels；
- random-initialized predictor；
- parameter-matched unconstrained/high-capacity mapper；
- mapper-only label prediction probe；
- deliberate future-label leakage，仅作应当失败的 sanity check；
- random environment partition；
- intervention representation support 与 hidden-path divergence 检查。

## 9. Gate 与阈值

### Gate 0

基于一手来源逐项判断。只有保留 claim 均有最近邻、实质差异、证伪实验和失败动作，且所有直接覆盖已标为 `NOT_NOVEL`/baseline，才可 `PASS`。

### Gate A、1、2、3、4

计划书中的数值门槛尚无独立先验支持，统一标为：

```text
TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT
```

它们必须在对应首个正式实验前，使用独立 pilot 或 VALIDATION 通过版本化 amendment 冻结。不得观察 TEST 后设定阈值。定性必要条件仍立即生效：oracle 必须优于随机、true lag 必须优于 wrong lag、zero-refit 不得发生任何 fit、泄漏和信息注入会直接否决相应 Gate。

## 10. 统计分析

- 时间序列与预测指标：moving/block bootstrap，block 长度只用 TRAIN/VALIDATION 的依赖诊断选择；
- intervention metrics：以 base window 为 group 的 paired bootstrap；
- 预测差异：DM test 或 paired block bootstrap，按适用假设选择；
- 多数据集/概念/比较：主要 family 使用 Holm 校正；探索性 family 可用 Benjamini–Hochberg，并分表报告；
- 报告点估计、95% confidence interval、效应量和校正后 p 值；
- 不把同一 base window 派生的多个 pair 当独立样本。

## 11. 失败结果与停止规则

- 两轮概念/SCM 修复后 Gate A 仍失败：暂停自动定位并重构高层模型；
- Gate 1 不优于随机或只恢复 PLOT 已有边际轴：删除多轴定位 claim；
- Gate 2 需要 test-time fit 或测试状态标签：删除 zero-refit robust claim；
- 随机/无能力模型接近真实模型或 mapper 单独预测标签：所有受影响机制 claim 无效；
- 只有金融结果有效：停止方法创新路线；
- 所有负结果、失败区域和 protocol deviation 均进入最终报告，不删除失败 seed 或数据域。

## 12. 资源和执行面

本预注册冻结任务和科学身份，不冻结某台机器、GPU 型号、数量或显存上限。每个重负载 Stage 先运行最小代表性 probe，估计 CPU/RAM、GPU/VRAM、存储和时间，再选择本地、单机服务器、多卡或分布式 backend。backend 变化不得改变 seed、split、checkpoint、metric 或 Gate。服务器接入需要用户单独授权。

## 13. 修订规则

冻结后不得静默编辑本文件。任何修改必须：

1. 创建递增版本的新 preregistration/amendment；
2. 说明触发原因、变更字段和受影响 claim；
3. 在读取 TEST 结果前完成；
4. 生成新的 content hash 和 `ResearchContractManifest`；若修改触及新颖性证据，再由人工核验流程签发新的 Gate 0 决策；
5. 默认保留旧版本并标为 `SUPERSEDED`。只有用户显式授权、提供覆盖理由且系统先归档旧 artifact 并生成审计回执时，才可替换活动版本。
