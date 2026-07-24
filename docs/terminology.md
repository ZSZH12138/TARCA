# TARCA 术语与证据边界

核查日期：`2026-07-23`

## 不可越过的因果边界

TARCA 首篇研究只能对模型内部计算机制提出因果陈述。

模型内部干预一致性不能自动推出真实金融市场中的因果关系。

允许的表述是：在预先定义的高层模型、干预支持集、base/source 配对规则、冻结低层模型和输出度量下，某个内部表示干预与高层概念干预在 held-out pairs 上近似一致。禁止把这种一致性写成“变量在真实市场中导致结果”“模型发现了真实市场机制”或“交易收益证明真实世界因果”。

## 核心术语

| 术语 | TARCA 中的操作性定义 |
|---|---|
| 高层模型（high-level model） | 由研究者明确规定变量、机制、输入/输出与可允许干预的可解释模型；首篇计划中是动态 SCM/算法假设，不是另一个无限容量黑箱。 |
| 低层模型（low-level model） | 被解释的神经时序预测器及其可观测内部计算。主要证据模式要求预测器参数冻结。 |
| 结构因果模型（SCM） | 由变量、结构方程、外生项及干预算子组成的模型。合成 SCM 的真值只属于生成器；真实数据上的高层 SCM 是待证伪假设。 |
| 高层概念 | 高层模型中具有预注册定义、历史可计算性、支持集和干预语义的变量，如局部趋势、波动状态、稀疏冲击或跨变量传播。 |
| 模型计算因果 | 对模型内部节点/表示进行受控干预后，模型输出发生的可重复变化；因果对象是已知可操作的模型计算图。 |
| 真实世界因果 | 关于现实数据生成过程、市场参与者、政策或资产关系的因果陈述；需要额外识别假设与外部证据，不能由模型内部 patching 自动获得。 |
| 交换干预（interchange intervention） | 在 base 输入的计算中，将指定高层变量或低层内部表示替换为其在 source 输入下取得的值，然后比较高低层干预输出。 |
| 基础输入（base input） | 保留上下文并接收被交换概念/表示的输入窗口；低层干预输出在它的前向计算上产生。 |
| 来源输入（source input） | 提供欲交换概念值或内部表示的输入窗口；必须满足预注册支持集和匹配约束。 |
| 表示对齐 | 把高层变量与低层内部位置/子空间关联的映射。映射必须受线性、低秩或结构稀疏容量限制并接受 held-out 检验。 |
| 因果抽象 | 当高层模型的干预行为可由低层模型经指定映射保持时，高层模型作为低层计算的简化描述；声明只在给定输入、干预和输出支持集上成立。 |
| 近似因果抽象 | 高低层干预结果并非完全相等，而是在预注册度量与误差界内接近的因果抽象。误差界不自动推广到支持集之外。 |
| IIT | Interchange Intervention Training。用高层模型反事实输出监督低层模型的交换干预输出，使神经模型学习预定因果结构；TARCA 的主证据模式与之不同，要求冻结预测器。 |
| IIA / IIC | IIA 通常指离散输出的 interchange intervention accuracy；IIC 是 TARCA 计划中的连续/分布输出 interchange intervention consistency。IIC 必须说明 normalizer、horizon 聚合和距离，不能把二者混用。 |
| DAS | Distributed Alignment Search。通过梯度学习非标准基下的分布式表示子空间，使高低层交换干预对齐。 |
| PLOT | Progressive Localization via Optimal Transport。用高低层输出干预效应签名和 OT/UOT coupling 逐步缩小候选位置，并已明确实现 PLOT-guided DAS。 |
| OT | Optimal Transport。以给定代价在两个质量分布之间求 transport plan；在 TARCA 中只是候选 effect-signature 对应的数值工具，不自带因果语义。 |
| UOT | Unbalanced Optimal Transport。允许总质量不完全匹配的 OT 变体，可容忍无关候选；它不等于 distributional robustness。 |
| 时延（causal lag） | 高层概念变化到影响低层/输出所跨越的历史或传播偏移 \(\delta\)。它与 forecast horizon \(h\) 必须分开定义。 |
| 状态/环境（regime/environment） | 用预测时可获得的信息或合成真值定义的机制/分布条件。真实数据状态模型只能 `fit(train)` 后冻结地 `transform` validation/test。 |
| DRO | Distributionally Robust Optimization。对规定不确定分布集合中的最坏期望损失进行优化。使用 DRO 不等于自动获得真实未见状态保证。 |
| Wasserstein 模糊集合 | 以训练经验分布为中心、按训练尺度定义 cost、半径受验证集选择的 Wasserstein 球/集合。DiRoCA 已将其用于一般 causal abstraction。 |
| Cause | 目标概念的低层干预效应是否接近对应高层干预效应；必须按 concept、horizon 和 held-out pair 报告。 |
| Isolation | 干预目标概念时，非目标概念/输出分量是否避免不必要改变；高 Cause 不能替代 Isolation。 |
| Completeness | 选定位置/概念集合解释 oracle/full 可干预效应的覆盖程度；需要独立 normalizer，避免零效应产生虚高分。 |
| 空洞对齐 | 映射器靠过高容量、记忆 pair、读取标签或对随机模型拟合而得到高一致性，却不反映低层模型原有机制。 |
| 信息注入 | 解释/对齐模块把未来标签、目标输出、测试状态或 source 特有信息编码进干预表示，使结果看似符合高层模型。 |
| held-out intervention pair | 从未参与位置、映射、normalizer、超参数或阈值拟合的 base/source 干预对；测试 pair 必须与训练 pair 按预注册规则分离。 |

## 额外消歧

- **forecast horizon \(h\)**：从预测起点到第几个未来输出；是输出索引。
- **causal lag \(\delta\)**：来源概念/机制到其效应的传播偏移；是机制语义。`h` 与 `\delta` 不得合并为 generic timestep。
- **zero-refit**：解释器、位置、对齐映射、normalizer 和状态变换在 unseen regime 上不重新拟合。
- **位置真值**：合成教师网络中机制被植入的 layer/variable/lag-or-patch/subspace 地址。
- **干预真值**：合成高层 SCM 在规定干预下产生的未来输出效应。

## 证据等级边界

- 文献中已正式发表、预印本、OpenReview 和 workshop 状态必须分开记录。
- 论文许可证不等于代码许可证；PLOT/DiRoCA 代码许可证目前均为 `UNVERIFIED`。
- `NO_EXACT_DIRECT_MATCH_FOUND` 只是本次检索结果，不是新颖性证明。
- 金融数据只可作为 point-in-time 非平稳压力测试，不是方法创新或真实市场因果证据。
