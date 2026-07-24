# TARCA 预注册草案 v0

状态：`STAGE_0_RESEARCH_CONTRACT`
冻结日期：`2026-07-23`

本文件不宣称实验已实施。凡计划书缺少充分依据的数值阈值，统一冻结为：

`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 1. 研究问题

### 主要 RQ

- **RQ1**：对冻结的神经时序预测器，受容量限制的高低层对齐能否在 held-out intervention pairs 上保持多步概率输出的近似交换一致性？
- **RQ2**：在同时给出 variable、causal lag、forecast horizon 与 constrained subspace 真值时，forecast-indexed 渐进定位能否恢复植入机制，并在干预次数/运行成本上优于 Full DAS？
- **RQ3**：同一冻结解释器能否在 sequential unseen regime 上 zero-refit 地保持干预保真度，并优于 ERM、Group-DRO、DiRoCA-style 与随机重加权基线？

### 次要 RQ

- **RQ4（探索性）**：仅在 RQ1–RQ3 通过后，抽象正则是否改善最坏状态预测或校准？解释成立但预测无收益必须如实报告。
- **RQ5（验证层）**：同一方法在至少两个非金融域与一个 point-in-time 金融压力测试中的失败模式是否一致？金融不是方法新颖性来源。

## 2. 假设

- **H1**：真实植入机制的低容量对齐在 held-out pairs 上优于随机模型、随机概念、错误 SCM、错误 source/lag 与参数量匹配无约束映射。
- **H2**：显式区分 horizon 与 lag、并加入 variable truth 的窄 TARCA 变体在联合四轴真值上优于 generic PLOT-style timestep/layer/subspace 定位。
- **H3**：若共享机制存在，zero-refit robust objective 将降低 unseen-regime worst abstraction error，且不依赖测试状态标签。
- **H4（探索性）**：机制约束可能改善 worst-regime forecast/calibration，但不作为前置成功条件。

## 3. 指标

### 主要指标

1. horizon-wise 与聚合的 TII distribution distance；具体主距离：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`
2. IIC、Cause、Isolation、Completeness。
3. planted location recovery：site precision/recall/F1、variable F1、patch/lag IoU、subspace principal-angle、rank error。
4. unseen-regime worst abstraction error 与 zero-refit held-out IIC。
5. 负对照与真实机制之间的效应量。

### 次要指标

- forecasting：NLL、CRPS、pinball loss、coverage、calibration error、MAE/MSE。
- 效率：干预次数、定位/训练时间、峰值内存/显存、激活缓存、相对 Full DAS 加速。
- 稳定性：跨 seed、时间起点、concept、regime 和 pair bucket 的方差/失败率。
- 金融经济指标（若执行）仅为次级结果，不替代预测与机制指标。

所有 unsupported primary/secondary success thresholds：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 4. 合成困难度轴

变量数 \(D\)、历史窗口 \(L\)、预测期 \(H\)、状态数、概念重叠、子空间维度/秩、lag 范围、信噪比、环境偏移半径、高层 SCM 错设、跨变量传播、异方差、缺失率。每条轴的离散水平与具体范围均为 `TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。

## 5. 主要负对照

- random model、random concept、shuffled concept label、wrong high-level SCM。
- wrong source、source=base、wrong lag、random layer/variable/patch/subspace。
- random regime labels、random reweighting。
- parameter-matched unconstrained mapping、high-capacity nonlinear mapping。
- mapping-only future-target probe。
- 明知可见未来标签的 leakage 版本仅作 sanity check，不得参与方法排名。

## 6. 候选数据集

优先顺序：

1. 合成 regime-switching nonlinear VAR/SCM + planted teacher network（必须同时有 intervention truth 与四轴 location truth）。
2. 非金融候选：Weather、Electricity、Traffic；至少再加入一个更新/跨域协议候选（fev-bench 或 GIFT-Eval），最终选择在正式实验前冻结。
3. 金融压力测试候选：FI-2010 LOB；多资产收益/波动率 point-in-time 数据。金融数据许可、可得性和具体标的均须在下载前单独审计。

正式主数据集清单、版本、切分日期、hash：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 7. 候选模型

- 最小合成 teacher/student 与简单线性/MLP/GRU 基线。
- 冻结 forecasting backbone 候选：DLinear、PatchTST、iTransformer。
- Chronos-class TSFM 仅作可选扩展，不是首个正式实验前置。

正式模型、参数规模、checkpoint/hash：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 8. 随机种子策略

- 多个固定 seed，覆盖数据生成、模型初始化、pair sampling、mapping/OT 优化。
- 所有主比较共享 seed 集合；失败 seed 不删除。
- seed 数量与具体整数列表：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 9. intervention pair 拆分

- 先按 base window/时间块分组，再构造 train/validation/test pairs，禁止同一 base window 跨 split。
- train pairs 用于学习位置/映射和距离尺度；validation pairs 仅选超参数；test pairs 仅做一次锁定评估。
- 同状态主结果与跨状态压力测试分层；source overlap/support 不足的 pair 拒绝并计数。
- 金融任务使用 rolling-origin，并按标签重叠采用 purging/embargo。
- split 比例、pair 数、匹配阈值、purge/embargo 长度：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 10. 允许调参范围

只允许在 validation 上选择：alignment rank/sparsity、OT/UOT regularization、candidate top-K、lag window、DRO radius/cost weights、optimizer 和 learning rate。不得调：测试 split、测试状态定义、未来特征、结论方向。

每项数值网格与总调参预算：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 11. Gates

### Gate A：固定位置干预

必须同时满足：oracle site 在 held-out pairs 上优于随机 site；Cause 与 Isolation 均有效；source=base 效应接近零；true lag 优于 wrong lag；随机模型/概念不接近真实机制；train–test gap 可接受。具体阈值：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。两轮概念/SCM 修复仍失败则暂停自动定位。

### Gate 1：合成定位与反空洞

必须同时恢复 intervention truth 与 location truth；narrow TARCA 在联合四轴真值上优于 PLOT、DAS、随机定位，且成本低于 Full DAS；held-out pairs 有效；容量增大不能让随机模型追平。具体阈值：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。失败则停止真实数据定位。

### Gate 2：跨状态解释

unseen-regime worst abstraction error 优于 ERM/Group-DRO/DiRoCA-style/随机重加权，平均性能无不可接受退化，不读取测试状态标签，rho 选择跨 seed 稳定。具体阈值：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。若只能 test-refit，撤销状态鲁棒声明。

### Gate 3：预测收益（探索性）

只有在 Gate A/1/2 通过后评估；若声明预测收益，须跨至少两个非金融域和一个金融任务/或预注册替代域、多个模型与 seeds 一致。具体阈值：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。机制有效但预测无收益时改为纯解释结果，不隐藏失败。

### Gate 4：论文完整性

主表可由脚本生成；结果可追溯到 config/data/code hash；负结果与失败状态完整；主结论不依赖单一数据集/模型/concept/seed；理论假设与实现一致；无真实市场因果越界。任何量化完整性阈值：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`。

### Gate 0：持续新颖性复核

每次正式阶段开始前复查最新一手来源。若差异只剩金融应用，停止该方法声明。

## 12. 统计检验

- 时序与 pair 指标使用按时间块/base window 分组的 block/cluster bootstrap。
- 预测误差使用适当的 paired bootstrap 或 Diebold–Mariano；机制指标按 pair/time block 配对。
- 多 concept/dataset/asset/state 比较采用预注册的 Holm 或 Benjamini–Hochberg 之一。
- 同时报效应量、置信区间、全部 seed 和失败 bucket，不只报 p 值。
- alpha、bootstrap 次数、block length、correction family：`TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`

## 13. 失败结果报告

- 报告所有失败 regime、concept、dataset/model、market 和 seeds。
- 单独报告 negative controls 过强、mapping-only probe 成功、pair overlap/support 不足、distance 不稳定、OT mass 分散、DRO 平均性能下降。
- import/compile/help/single-function smoke 不得升级为论文复现或科学结果。

## 14. 停止规则

- 随机模型/概念/错误 SCM 在同容量下接近真实机制：停止机制主张。
- held-out pairs 崩溃或映射能单独预测未来标签：停止并审计注入。
- Gate A 或 Gate 1 两轮修复仍失败：停止自动定位与真实数据实验。
- robust 收益依赖 test-refit/test regime labels：停止 N5。
- 新颖性只剩“用于金融”：停止方法方向。
- 预计命令超过阶段资源门禁时停止运行，不以伪造/删减结果替代。

## 15. 不可提出的越界结论

- 不得声称模型内部一致性识别了真实市场因果。
- 不得声称 PLOT-guided DAS 是 TARCA 新方法。
- 不得声称 generic multi-axis/progressive localization 是新方法。
- 不得声称 generic Wasserstein distributionally robust causal abstraction 是新方法。
- 不得把金融应用、收益或非平稳性写成方法新颖性。
- 不得把未运行、smoke、单 seed、单数据集或选择性成功写成正式验证。
