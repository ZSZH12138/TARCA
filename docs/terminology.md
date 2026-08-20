# TARCA 术语表

> 版本：1.0.0
> 冻结范围：Stage 0 研究合同。下游代码、配置、表格和论文不得为同一词另设冲突含义。

## 1. 因果声明边界

### 模型计算因果（model-computational causality）

在一个已指定并冻结的模型中，主动替换、消融或改变内部状态后，模型输出发生的变化。它支持关于“模型怎样计算”的声明，不自动支持关于真实市场、天气、电力系统或人类行为的因果声明。

### 真实世界因果（real-world causality）

关于现实系统中干预如何改变结果的声明，需要额外的识别假设、数据生成过程和外部效度论证。TARCA 的模型内部干预不能单独证明真实世界因果。

### 识别（identification）

在明确假设下，从可观察证据唯一确定目标因果量的条件。高 IIC、faithfulness、activation patching 或预测提升是验证证据，不等同于完成识别。

## 2. 时间序列轴

### 历史窗口长度（history length, `L`）

预测器一次读取的过去时间步数量。

### 预测期（forecast horizon, `H` / `h`）

`H` 是模型一次输出的未来步数；`h` 是其中某一个未来位置。它回答“预测未来第几步”。

### 因果时延（causal lag, `δ`）

一个高层机制或干预经过多少时间步才影响目标。它回答“机制延迟多久生效”，不得与 forecast horizon 混用。

### 变量/通道（variable/channel）

多变量时间序列中的可命名输入或目标维度。通道独立模型中的 channel 不自动具有跨变量机制语义。

### 时间 patch

由连续时间步组成的模型 token。必须记录原始时间覆盖，不能只使用 token 序号表示 causal lag。

## 3. 因果抽象与干预

### 高层模型（high-level model）

使用概念和机制描述目标计算的可解释模型或 SCM。

### 低层模型（low-level model）

被解释的冻结神经时序预测器及其内部计算。

### 因果抽象（causal abstraction）

一个高层模型与低层模型之间的映射，使对应干预的输出行为在规定误差内一致。

### 交换干预（interchange intervention）

用 source 样本在指定位置的值或表示替换 base 样本对应内容，并观察低层模型输出变化。

### base / source

`base` 是接受干预的样本；`source` 提供被替换内容。`source=base` 是应产生近零效应的 sanity check。

### intervention pair

一个明确的 base/source 组合及其 partition、regime、时间距离和概念关系。相同窗口不得跨 train/validation/test pair partition 泄漏。

### 高层效应 / 低层效应

高层效应来自高层概念干预，低层效应来自模型内部干预。二者必须使用同一输出语义、同一 horizon 索引和只在训练 pair 上拟合的 normalizer 才能比较。

## 4. 位置与子空间

### intervention site

模型中可被干预的明确位置描述，例如层、时间 patch、变量 token、坐标组或子空间。位置目录与单次干预请求是不同对象。

### 受限子空间（constrained subspace）

有预注册秩、参数量或群稀疏约束的表示子空间。无限容量非线性映射不属于可接受的 claim-bearing 对齐。

### 联合位置真值

同时指定 variable、causal lag、forecast horizon 和 constrained subspace 的植入机制真值。只恢复其中一个轴不等于完成四轴定位。

### PLOT-guided DAS

使用 PLOT 定位信号缩小 DAS 搜索范围的已有方法。TARCA 仅把它作为基线。

## 5. 预测与抽象指标

### 概率预测分布

对每个 target 和 horizon 输出完整、可验证的概率分布参数或样本，而不只是点预测。

### TII / 干预效应距离

高层和低层干预效应签名之间的连续距离。具体距离可以是 Wasserstein、energy distance、CRPS difference 或预注册的归一化距离。

### IIC

归一化的交换干预一致性分数。它衡量高层和低层效应的接近程度，不应被解释为真实世界因果效应大小。

### Cause

目标概念的内部干预是否产生与高层干预方向和大小相符的效果。

### Isolation

干预目标概念时，非目标概念或输出是否避免不必要变化。

### Completeness

选定概念和位置解释了多少 oracle/full intervention 效应。

### 未映射变量 faithfulness

检查没有被显式映射的变量是否仍破坏高低层干预一致性。忽略这些变量不能被当作有效抽象。

## 6. 鲁棒性和生命周期

### seen regime / unseen regime

seen regime 的机制参数或组合出现在训练数据中；unseen regime 保留了新的机制参数、组合或传播结构，且不能用于拟合 claim-bearing 组件。

### sequential unseen regime

按预注册顺序依次到达的多个未见状态。后一个状态的评估不能利用未来状态信息。

### zero-refit

predictor、位置、映射、解释器、normalizer 和阈值在进入未见状态前全部冻结；评估期间不得更新。它不同于 test-time adaptation。

### test-time adaptation

使用测试期输入或反馈更新模型或表示。任何此类更新都不能被报告为 zero-refit 结果。

### 预测鲁棒性

状态变化下 NLL、CRPS、MAE、coverage 等预测指标的稳定性。

### 抽象/解释鲁棒性

状态变化下高低层干预一致性、位置和概念效应的稳定性。预测鲁棒不自动推出抽象鲁棒。

### fit / freeze / apply

- `fit`：只用允许的 train/validation 范围估计参数；
- `freeze`：固定参数、版本和内容 hash；
- `apply`：在不更新冻结状态的前提下变换或评估新输入。

## 7. 反空洞与执行治理

### 信息注入

映射器、配对器或干预机制从标签、未来信息或高容量自由度中构造答案，而不是揭示冻结模型已有计算。

### 表示偏离（representational divergence）

干预后的内部表示离开正常计算支持。必须区分无害 null-space 变化与通过隐藏路径改变输出的危险偏离。

### failure region

抽象在某类输入、intervention pair 或 regime 上系统失败的区域。全局均值不能替代该诊断。

### ArtifactRef

由 artifact type、内容 SHA-256 和 schema version 定义身份的不可变引用；相对路径不是内容身份。

### ResearchContractManifest

Stage 0 对预注册、新颖性、假设、术语、环境和相关工作证据的统一冻结入口。

### Gate / GateDecision

Gate 是预先定义的继续/停止规则；GateDecision 是绑定相应 GateSpec 和证据的结构化决定。`GATE_0_NOVELTY 是人工新颖性 Gate，不要求 GateSpec`，其决定直接绑定冻结的 novelty claims 与 related-work bundle。它不是计划书中的进度标记。

### Science Plane / Execution Plane

Science Plane 决定 seed、split、metric、checkpoint 和 Gate；Execution Plane 决定本地/服务器、CPU/GPU、worker 和调度。更换执行 backend 不得改变 scientific identity。

### PROVISIONAL

经过最新直接碰撞审计后仍存在可证伪差异的候选 claim。它不等于“已证实创新”或“首次”。
