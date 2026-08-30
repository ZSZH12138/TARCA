# TARCA：面向状态切换的时序因果抽象与分布鲁棒机制定位研究计划

> **计划版本**：v1.3
> **原始检索截点**：2026-07-14
> **契约修订核对日期**：2026-08-20
> **契约优先级**：Stage 0 应冻结 `preregistration_v0.md`、`assumption_ledger.md`、`novelty_claims.md`、`terminology.md`；Stage 1+ 必须服从这些研究契约。没有明确、可追溯的上位修订时，不得静默改写其 Gate、证据等级或创新边界。
> **目标**：形成一个以通用时间序列方法为主体、金融序列为高难度验证场景。

---

## 0. 执行结论

### 0.1 最终判断

本计划建议研究：

> **将神经因果抽象、渐进式最优传输机制定位和分布鲁棒优化统一到非平稳多变量时间序列中，构造可被干预验证、可定位到“层 × 时间片/因果时延 × 变量/通道 × 受限子空间”、并在状态切换下保持稳定的预测解释。**
### 0.2 CCF-A 级别判断

结论为：**条件性可行**。

仅做下列工作，不足以达到 CCF-A 型论文标准：

- 把 PLOT、DAS、IIT 或 DiRoCA 直接套到股票数据；
- 只展示若干可视化解释；
- 只在一个市场、一个数据集和一个预测任务上优于基线；
- 将模型内部干预结果错误地表述为真实市场因果关系；
- 仅提升收益率回测，没有通用方法、理论或机制验证。

要形成有竞争力的论文，必须同时完成：

1. **窄化后的形式化候选**：在冻结神经时序预测器上，定义按 forecast horizon 分解且与 causal lag 独立索引的概率干预抽象；
2. **窄化后的算法候选**：围绕预测期、因果时延、变量/通道与受限子空间联合真值做渐进式机制定位；
3. **窄化后的鲁棒性候选**：在冻结神经时序预测器上，优化 sequential unseen regime 的最坏抽象误差并保持 zero-refit；
4. **理论结果**：至少给出近似抽象误差或最坏环境误差上界；
5. **反空洞性设计**：限制对齐映射容量，设置冻结模型、安慰剂概念和反信息注入实验；
6. **通用实证**：合成真值、非金融时间序列和作为验证层的金融压力测试三层证据；
7. **严格金融协议**：滚动切分、purging/embargo、发布时间对齐、交易成本和统计显著性检验；
8. **开源可复现基准**：公开代码、配置、数据处理脚本和主要实验日志。

若其中第 1、3、4、5 项无法实现，应将项目降级为普通应用论文，而不是继续按 CCF-A 目标投入。

---
## 3. 研究空白

现有时间序列解释方法通常回答：

- 哪些输入时间点重要；
- 哪些变量对预测贡献较大；
- 删除某段输入后预测如何变化；
- 如何生成能改变预测的反事实序列。

这些方法往往不能同时回答：

1. 一个高层概念是否真的由模型内部某个机制实现；
2. 概念在网络的哪一层、哪些变量、哪些时间片和哪个表示子空间中被实现；
3. 替换该内部机制后，输出是否按高层概念模型的干预语义变化；
4. 该解释在牛市、熊市、高波动、低流动性或结构突变环境下是否仍成立；
5. 解释模块是否偷偷向模型注入了新信息，而不是发现原有机制；
6. 在主机制证据成立后，解释约束是否可能改善分布外预测（探索性问题，不是主方法成立的前提）。

TARCA 的研究空白是：

> **针对非平稳、多变量、序列到多步概率分布的预测器，建立具有时间延迟、跨变量传播和状态切换语义的神经因果抽象，并以最坏环境而非平均环境的干预一致性作为优化目标。**

---

## 4. 必须严格区分的两种“因果”

### 4.1 模型计算因果

TARCA 研究的是：

> 对模型内部表示做受控干预，是否会按照给定高层算法的预测改变模型输出。

这是关于**模型计算机制**的因果陈述，可以通过交换干预、路径干预和消融实验验证。

### 4.2 真实市场因果

“利率变化导致股票下跌”“订单不平衡导致未来收益”属于真实世界因果陈述。仅靠神经因果抽象不能识别这些关系，因为还涉及混杂、选择偏差、反馈、政策反应和不可观测因素。

因此论文必须避免：

```text
错误表述：模型发现了市场中 X 对 Y 的真实因果效应。
允许表述：在给定高层机制假设和模型干预定义下，模型的预测计算表现出与 X→Y 一致的内部因果抽象。
```

若要提出真实市场因果结论，需额外引入有效工具变量、自然实验、随机干预或明确的可识别性假设；这不应成为第一篇论文的主线。

---

## 5. 核心研究问题与假设

### RQ1：时序预测器能否被一个高层动态 SCM 忠实抽象？

给定多变量输入窗口：

$$
X_{t-L+1:t}\in \mathbb{R}^{L\times D},
$$

低层预测器输出多步概率预测：

$$
F_\theta(X_{t-L+1:t}, Z_t)
\rightarrow
p_\theta(Y_{t+1:t+H}\mid X,Z),
$$

其中 $Z_t$ 为当时可获得的外生变量。

构造高层动态结构因果模型：

$$
\mathcal{M}^{(r)}_H:
C_t \rightarrow C_{t+1:t+H} \rightarrow \widehat{Y}_{t+1:t+H},
$$

其中 r 表示状态/环境，$C_t$ 是高层时序概念。

**H1**：存在受容量限制的对齐映射，使高层干预和低层内部干预的预测效应近似交换。

### RQ2：概念机制位于模型的什么位置？

候选神经位置定义为：

$$
s=(\ell,p/\delta,d,U),
$$

分别表示层 $\ell$、时间 patch/因果时延 $p/\delta$、变量/通道 $d$、受限表示子空间 $U$。预测期 $h$ 属于输出效应索引，必须与因果时延 $\delta$ 独立定义和验证，不能合并为泛化的 timestep。

**H2**：从粗到细的最优传输定位比穷举 DAS 更高效，并能恢复合成模型中的已植入机制。

### RQ3：解释能否跨状态保持稳定？

**H3**：在 Wasserstein 模糊集合上最小化最坏环境抽象误差，可提高未见状态下的干预保真度。

### RQ4：解释约束能否改善分布外预测？（探索性）

**H4（探索性）**：当高层概念是任务相关、低容量且干预语义正确时，因果抽象正则可能抑制状态特异的伪相关，提高最坏状态预测、校准和稳定性。H4 只在 Gate A/1/2 通过后评估；不成立不否定 RQ1–RQ3 的机制解释结果。

---

## 6. 高层概念系统

### 6.1 通用概念

首篇论文应优先使用跨领域可定义的概念，而不是直接使用金融术语：

- 局部水平与趋势；
- 周期/季节性成分；
- 局部尺度或波动状态；
- 稀疏冲击；
- 跨变量传播；
- 外生变量影响；
- 均值回复与持续性；
- 结构切换。

### 6.2 金融实例化

金融实验中可映射为：

| 通用概念 | 金融实例 |
|---|---|
| 局部趋势/持续性 | 动量 |
| 均值回复 | 短期反转 |
| 局部尺度 | 波动率状态 |
| 稀疏冲击 | 宏观公告、新闻或跳跃 |
| 跨变量传播 | 市场、行业或资产溢出 |
| 外生影响 | 利率、通胀、风险情绪 |
| 状态切换 | 牛/熊、高/低波动、高/低流动性 |
| 局部供需压力 | 订单簿不平衡 |

### 6.3 概念来源的三层设计

1. **解析概念**：用只依赖历史窗口的公式计算，如局部斜率、已实现波动率和滞后相关；
2. **弱监督概念**：由领域规则或事件元数据产生；
3. **可学习但受限的概念**：低维、稀疏、单调或具有最小描述长度约束。

主论文不得只使用完全自由的可学习概念，否则高层模型容易沦为另一个黑箱。

---

## 7. 方法设计

### 7.1 时序因果抽象

定义输入概念映射 $\tau_X$、模型表示映射 $\alpha_s$ 和输出映射 $\tau_Y$。

对基础窗口 x、来源窗口 x' 和概念 $C_k$，理想关系为：

$$
\tau_Y\!\left(
F_\theta^{do(\alpha_s\leftarrow \alpha_s(x'))}(x)
\right)
\approx
\mathcal{M}^{do(C_k\leftarrow C_k(x'))}
\left(\tau_X(x)\right).
$$

与静态任务不同，需要增加：

- 预测期 $h=1,\ldots,H$；
- 干预延迟 $\delta$；
- 跨通道传播路径；
- 不同状态 r；
- 连续概率输出距离。

定义时序交换干预误差：

$$
\mathcal{E}_{\mathrm{TII}}
=
\sum_{h=1}^{H} w_h
D\!\left(
p_{\theta,h}^{do(s\leftarrow x')},
p_{\mathcal{M},h}^{do(C_k\leftarrow x')}
\right),
$$

其中 D 可取 Wasserstein、 energy distance、 CRPS 差异或归一化 $L_1/L_2$。

### 7.2 渐进式多轴 OT 定位

第一阶段在粗粒度位置上寻找概念：

```text
模型族单独审计 → 层 → 时间 patch/因果 lag → 变量/通道 → 受限表示子空间
```

模型家族不是位置轴；四轴定位按上述固定顺序逐级缩小候选集合。为每个高层概念计算干预效应签名：

$$
e_k^{H}
=
\left[
\Delta\mu_{1:H},
\Delta\sigma_{1:H},
\Delta q_{\alpha,1:H}
\right].
$$

校准是预测分布与真实观测的联合、聚合性质，不能作为单个干预 pair 的可加签名分量。只有在合成 oracle 或真实目标 $y_i$ 可用时，才可另报逐样本 $\Delta\mathrm{NLL}_i$、$\Delta\mathrm{CRPS}_i$ 或固定分位数的 $\Delta\mathrm{pinball}_{\alpha,i}$；PIT、coverage、reliability 与 calibration error 必须在 `fold × horizon × subgroup/regime` 层聚合，且不得作为测试时定位输入。

为每个候选神经位置计算低层效应签名 $e_s^L$，构造代价矩阵：

$$
C_{ks}
=
d(e_k^H,e_s^L)
+\lambda_{\text{lag}}d_{\text{lag}}
+\lambda_{\text{iso}}d_{\text{isolation}}
+\lambda_{\text{cap}}\Omega(U_s).
$$

求解熵正则最优传输：

$$
\pi^\star
=
\arg\min_{\pi\in\Pi(a,b)}
\langle C,\pi\rangle
+\varepsilon\operatorname{KL}(\pi\Vert a\otimes b).
$$

高传输质量（mass）的候选区域进入下一层细化，最终用 DAS 优化低维干预子空间。IIT 只属于后述联合训练的次级模式，不是冻结模型主线的定位搜索器。

### 7.3 时延感知交换干预

普通交换干预把一个样本的隐藏表示替换为另一个样本的表示。时序版本需解决来源窗口与基础窗口的相位、尺度和状态差异。

建议设计三类干预：

1. **同状态匹配干预**：只在相同状态的窗口间交换概念；
2. **跨状态压力干预**：在不同状态间交换，评估解释鲁棒性；
3. **时延对齐干预**：允许概念在 $\delta\in[-\Delta,\Delta]$ 内匹配，以识别传播延迟。

来源窗口匹配必须基于干预概念以外的协变量做近邻或最优传输匹配，降低不受控变化。

### 7.4 分布鲁棒抽象目标

环境集合包括：

- 时间阶段；
- 波动率分位组；
- 流动性分位组；
- 结构突变前后；
- 不同市场/资产；
- 合成 SCM 中的已知机制状态。

模式 A 的主目标在冻结预测器 $F_\theta$ 上优化解释参数与传输计划：

$$
\begin{aligned}
\min_{\alpha,\pi}\quad
&\lambda\,\mathcal{L}_{\text{TII}}\\
&+\beta
\sup_{Q:W_c(Q,\widehat{P})\le \rho}
\mathbb{E}_{Q}
[\mathcal{E}_{\mathrm{TII}}]\\
&+\gamma\mathcal{L}_{\text{isolation}}
+\eta\Omega(\alpha)
+\zeta\mathcal{L}_{\text{anti-injection}}.
\end{aligned}
$$
其中：

- $\mathcal{L}_{\text{TII}}$：平均时序交换干预误差；
- DRO 项：最坏分布邻域下的抽象误差；
- isolation：干预非目标概念时目标输出不应错误改变；
- $\Omega(\alpha)$：子空间维度、秩、稀疏度或描述长度约束；
- anti-injection：阻止解释模块编码目标标签或未来信息。

### 7.5 两种训练模式

### 模式 A：冻结模型的事后解释

- 冻结 $F_\theta$；
- 只学习位置、低秩子空间和高层映射；
- 用于验证模型原有机制；
- 作为论文的主要可解释性证据。

### 模式 B：联合训练（探索性/次级）

- 可在独立实验中优化 $\theta$、$\alpha$ 与 $\pi$，并加入预测损失；
- 联合训练只能作为次级证据、消融或性能上界，必须与模式 A 分表、分结论报告；
- 该模式只能回答 abstraction-regularized forecasting 是否有用，不能证明冻结预测器原本已经实现该机制，也不能用于关闭 Gate A/1/2 或支持 zero-refit 主张。

---

## 8. 反空洞性与反信息注入设计

因果抽象存在一个重要风险：若对齐映射足够强，任意模型可能被映射到任意高层算法，导致解释失去信息量。

必须加入以下限制：

1. 对齐映射只允许线性、低秩或结构化稀疏形式；
2. 概念子空间维度预注册并做容量扫描；
3. 优先在冻结模型上定位；
4. 训练来源窗口和测试来源窗口严格分离；
5. 不允许解释模块访问未来标签；
6. 使用随机初始化模型作为负对照；
7. 使用随机概念、打乱概念和错误高层 SCM；
8. 测量解释模块单独预测目标的能力；
9. 报告准确率—容量前沿，而非只报最高一致性；
10. 对每个概念同时报告 cause 与 isolation；
11. 使用 held-out intervention pairs 检查交换泛化；
12. 对映射使用 MDL、秩或参数量惩罚。

**关键否决条件**：若随机模型或随机概念也能得到接近真实模型的高干预一致性，当前方法无效，不能进入真实金融实验。

---

## 9. 理论工作包

理论不应被写成装饰性附录，而应直接支撑方法成立。

### 定理目标 1：近似时序抽象

在干预族 $\mathcal{I}$、预测期 1:H 和受限对齐映射族 $\mathcal{A}$ 上，若训练得到：

$$
\sup_{i\in\mathcal{I}}
\mathcal{E}_{\mathrm{TII}}(i)
\le \epsilon,
$$

则高层动态 SCM 是预测器在该干预族上的 $\epsilon$-近似时序因果抽象。

需明确：

- 结论只对支持集内干预成立；
- 不外推为真实数据生成过程的因果模型；
- 连续输出使用何种概率度量；
- 多步误差是否随 H 累积。

### 定理目标 2：最坏环境误差界

在损失对环境分布满足 Lipschitz 条件时，争取证明：

$$
\mathcal{E}_{\mathrm{worst}}
\le
\widehat{\mathcal{E}}_{\mathrm{emp}}
+
L\rho
+
\epsilon_{\mathrm{loc}}
+
\epsilon_{\mathrm{est}},
$$

其中：

- $L\rho$：Wasserstein 模糊半径带来的项；
- $\epsilon_{\mathrm{loc}}$：渐进定位近似误差；
- $\epsilon_{\mathrm{est}}$：有限样本估计误差。

### 定理目标 3：预测鲁棒性联系

在高层概念对预测目标具有充分性、且状态变化主要作用于非因果/伪相关特征的假设下，分析抽象正则如何限制状态敏感表示，给出最坏环境风险差异界。

若无法得到第三个理论结果，前两个结果仍可构成主理论；只要 Gate A/1/2 闭合，机制解释结果仍可独立成立。预测收益只在 Gate 3 中作为探索性附加证据，不能反向为前述 Gate 补票。

---

## 10. 合成基准

合成部分是论文可信度的核心，不能只用真实金融数据。

### 10.1 数据生成器

设计 regime-switching nonlinear VAR/SCM：

$$
X_{t+1}
=
A^{(r_t)}X_t
+
g^{(r_t)}(X_{t-\delta})
+
B^{(r_t)}U_t
+
\sigma^{(r_t)}(X_t)\epsilon_t,
$$

包含：

- 非线性；
- 异方差；
- 延迟作用；
- 跨变量传播；
- 稀疏冲击；
- 不可观测混杂的可控版本；
- 机制切换；
- 观测噪声和缺失。

### 10.2 两类真值

#### A. SCM 真值

已知高层概念和干预后未来路径，可测抽象效应误差。

#### B. 机制植入网络

构造带已知概念模块的教师网络，将机制植入特定层、变量、时间片和子空间，可直接测定位精确率。

### 10.3 难度轴

- 变量数 $D$；
- 窗口长度 $L$；
- 预测期 $H$；
- 状态数；
- 概念重叠程度；
- 子空间维度；
- 延迟范围；
- 信噪比；
- 环境偏移半径；
- 高层模型错设程度。

---

## 11. 通用时间序列实验

金融不能是唯一场景。建议至少覆盖三种非金融域：

1. 电力/负荷；
2. 天气或环境；
3. 交通或传感器；
4. 可选：医疗生理序列。

推荐同时使用：

- 经典可比基准；
- 至少一个较新的、未明显饱和的基准；
- 具有外生变量的预测任务；
- 一个明显状态切换或跨域测试设置。

传统 ETT、Weather、Electricity、Traffic 可用于与文献对比，但不应成为唯一证据。TSLib 维护者在 2026 年已提醒部分旧基准可能不再适合衡量研究进展，因此应结合 fev-bench、GIFT-Eval 或自行构造的跨环境协议。

---

## 12. 金融实验设计

### 12.1 任务 A：限价订单簿短期方向与分布预测

数据：

- FI-2010 作为公开可复现基准；
- 可选 Binance 公共逐笔成交/订单簿档案；
- 有条件时增加 LOBSTER/TAQ，但不得使核心结论依赖付费数据。

预测目标：

- 中间价方向；
- $H$ 步价格变化分布；
- 短期实现波动率；
- 价格跳跃概率。

高层概念：

- 买卖盘不平衡；
- 深度与流动性；
- 价差；
- 订单流持续性；
- 短期冲击传播。

### 12.2 任务 B：多资产收益与波动率

数据：

- 公开日频/小时级多资产价格；
- FRED/FRED-MD 宏观变量；
- 市场、行业、利率、商品和加密资产的跨资产组合；
- 商业数据仅作为附加验证。

预测目标：

- 收益分位数；
- 实现波动率；
- 尾部风险；
- 跨资产条件分布。

高层概念：

- 动量/反转；
- 市场与行业溢出；
- 波动状态；
- 宏观冲击；
- 风险偏好；
- 相关性结构变化。

### 12.3 状态定义

状态标签只用于训练时可获得的信息：

- 历史波动率分位组；
- 历史流动性分位组；
- 在线变点检测；
- 隐马尔可夫状态的仅训练期拟合；
- 明确日期的外生事件。

不得使用整个测试期数据拟合状态划分。

---

## 13. 基线体系

### 13.1 预测基线

- AR/VAR、GARCH/HAR-RV；
- DLinear；
- PatchTST；
- iTransformer；
- TimeXer；
- TFT；
- DeepLOB（订单簿任务）；
- Chronos-2；
- FinCast 或其他可获得代码的金融 TSFM。

### 13.2 归因与反事实基线

- Integrated Gradients；
- DeepSHAP/SHAP；
- TimeSHAP；
- permutation/occlusion；
- attention rollout；
- ForecastCF；
- ConTex；
- 时间序列 Concept Bottleneck；
- SAE 与 activation patching。

### 13.3 因果抽象基线

- IIT；
- DAS；
- Boundless DAS；
- HyperDAS；
- 原始 PLOT；
- 原始 DiRoCA；
- PLOT-guided DAS。

PLOT-guided DAS 是 `NOT_NOVEL` 基线：PLOT 已明确提出这一组合。TARCA 只能比较其窄化变体在 forecast-indexed variable、causal lag、forecast horizon 与 constrained subspace 联合真值上的恢复，不得把候选缩减加速本身写成贡献。

所有基线必须使用相同预测器、数据切分、概念标签和干预对，以隔离解释算法差异。

---

## 14. 评价指标

### 14.1 预测

- MAE、MSE；
- pinball loss；
- CRPS；
- NLL；
- coverage 与 calibration error；
- 方向准确率、AUC/F1；
- 最坏状态风险；
- 跨市场/跨年份性能下降；
- 经济指标只作为次级结果：净收益、Sharpe、最大回撤、换手率和交易成本敏感性。

逐样本 proper score 用于预测误差或单样本 score-delta 诊断；coverage、PIT、reliability 与 calibration error 只在 `fold × horizon × subgroup/regime` 上聚合解释，不能从单个样本宣称“已校准”。

### 14.2 解释与机制

### 连续版交换干预一致性

$$
\mathrm{IIC}
=
1-
\frac{
D(\widehat{Y}^{L,do},\widehat{Y}^{H,do})
}{
D_{\mathrm{normalizer}}+\epsilon
}.
$$

同时报告：

- Cause：目标概念干预是否产生正确效应；
- Isolation：非目标概念是否保持不变；
- Completeness：选定概念解释了多少干预效应；
- Localization F1/IoU：合成植入模型的位置恢复；
- 子空间稀疏度与秩；
- 跨状态解释稳定性；
- 跨随机种子一致性；
- held-out intervention generalization；
- 每个概念的失败区域。

### 14.3 效率

- 定位搜索时间；
- GPU 小时；
- 干预次数；
- 显存；
- 相对穷举 DAS 的加速比；
- 模型与序列长度扩展曲线。

---

## 15. 金融数据泄漏与统计协议

1. 采用 rolling-origin 或 expanding-window；
2. 对标签或干预后效跨区间定义的任务，按信息区间重叠执行 purging；gap/embargo 至少覆盖最大特征回看泄漏与最大标签/干预后效窗口，具体长度须预注册，不能只机械设为 $H-1$；
3. 标准化器、PCA、状态模型和概念阈值只在训练期拟合；
4. 宏观数据按实际发布日期和修订版本对齐；
5. 处理退市、成分股变更和幸存者偏差；
6. 复权逻辑和公司行动需可审计；
7. 禁止使用未来窗口构造输入概念；
8. 超参数只在验证期选择；
9. 多随机种子和多个时间起点；
10. 使用 block bootstrap 置信区间；
11. 预测差异可使用 Diebold–Mariano 或适当的配对检验；
12. 多任务、多资产比较需控制多重检验；
13. 回测纳入手续费、滑点和换手；
14. 报告所有失败市场和失败状态，不只展示正结果。

---

## 16. 核心消融与负对照

### 16.1 方法消融

- 去掉 OT；
- 去掉渐进式定位；
- 去掉时延对齐；
- 去掉跨变量维度；
- 去掉 DRO；
- 固定状态与动态状态；
- 冻结模型主证据与联合训练次级消融（分表、分结论，后者不得关闭 Gate A/1/2）；
- 线性、低秩和非线性对齐；
- 单一高层 SCM 与状态混合 SCM；
- 不同概念子空间维度。

### 16.2 负对照

- 随机概念；
- 打乱概念标签；
- 错误时间延迟；
- 错误来源窗口；
- 随机初始化预测器；
- 随机子空间；
- 干预与目标无关的层；
- 标签随机化；
- 允许与禁止未来信息的 sanity check；
- 参数量匹配但无因果约束的解释器。

### 16.3 竞争性解释

对同一预测器比较：

- attribution 图；
- counterfactual 序列；
- concept bottleneck；
- SAE feature；
- TARCA 的机制路径。

用户研究不是首篇论文的必要条件，但可增加专家是否能利用解释发现错误状态或模型失效的评估。

---

## 17. 预期论文贡献

论文贡献必须写成以下层级，而不是“首次用于金融”：

1. **Temporal Causal Abstraction（候选贡献）**：在冻结神经时序预测器上，检验按 forecast horizon 分解且与 causal lag 独立索引的概率干预抽象；
2. **Multi-axis Progressive Localization（窄化候选贡献）**：检验 forecast-indexed variable × causal lag × forecast horizon × constrained subspace 联合真值，而不是把通用渐进定位重新命名；
3. **Regime-Robust Abstraction（窄化候选贡献）**：检验 frozen forecaster + sequential unseen regime + zero-refit 下的最坏抽象误差，而不是声称提出一般性的 Wasserstein causal abstraction；
4. **Anti-vacuity Protocol（必要支撑贡献）**：用容量限制、冻结模式、未映射变量 faithfulness、表示支持检查和负对照防止空洞对齐与信息注入；不把通用反空洞诊断本身列为首要方法创新；
5. **Abstraction-Regularized Forecasting（探索性）**：仅在 Gate 3 通过时作为附加贡献；失败不否定机制解释主线；
6. **Forecast-specific Benchmark（窄化候选贡献）**：在 CAE 等通用因果抽象评测之外，提供同时具有多步概率预测、干预真值、变量/causal-lag/forecast-horizon/受限子空间位置真值和状态切换的合成基准；不得声称首次提出通用因果抽象 metric 或模拟 benchmark；
7. **Financial Stress Test（验证层）**：在订单簿和多资产预测中检验方法的强非平稳边界，不作为方法新颖性。

---

## 18. CCF-A 型论文评估矩阵

| 维度 | 当前潜力 | 达标条件 | 否决条件 |
|---|---:|---|---|
| 问题重要性 | 8/10 | 解释保真度与状态鲁棒性被统一 | 只讨论金融可视化 |
| 方法新颖性 | 6/10（受 Gate 0 证据边界约束） | 冻结预测器、horizon/lag、变量联合真值与 unseen-regime zero-refit 形成不可分割的窄化方法 | 仅拼接 PLOT 与 DiRoCA |
| 理论深度 | 7/10（条件性） | 至少两个非平凡结果及清晰假设 | 只有直观描述 |
| 实证强度 | 9/10（工作量高） | 合成真值、通用 TS、金融三层验证 | 单一股票数据集 |
| 可解释性可信度 | 8/10 | cause/isolation、负对照、反注入 | 只报 attribution 图 |
| 通用性 | 8/10 | 方法不依赖金融专有概念 | 标题和算法均为股票特化 |
| 可复现性 | 8/10 | 公共核心数据、开源代码、完整配置 | 依赖私有数据 |
| 竞争风险 | 6/10 | 明确碰撞证据到达时 fail closed，并优先完成快速合成原型 | 新工作直接覆盖窄声明 |

### 总体判断

- **作为“金融应用论文”**：不够；
- **作为“通用时序因果抽象方法 + 金融压力测试”**：具备冲击 KDD/AAAI/IJCAI 等高水平会议的研究形态；
- **若理论、基准和跨域实证均达到高质量**：可按更强机器学习会议的审稿强度准备；
- 这只是研究潜力判断，不构成录用保证。

---

## 19. 阶段性 Go/No-Go 门槛

本节中的“明显”“接近”“可接受”“稳定”等定性词都不是可执行阈值。所有尚无支持的主要/次要成功阈值统一标记为 `TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT`，必须在首次正式实验前写入预注册，且不得观察测试结果后修改。

### Gate 0：新颖性复核

通过条件：

- 使用论文与官方仓库逐项复核第 17 节列出的候选贡献；PLOT-guided DAS、一般 Wasserstein/分布鲁棒因果抽象、通用因果抽象 metric/模拟 benchmark、“首次时间序列机制解释”和金融应用继续按 `NOT_NOVEL` 边界管理；
- 必须纳入 PLOT、DiRoCA、Generalised Transportability via Causal Abstractions、CAE、TimeSAE、Representational Divergence、FOIL、COGS 及其最新官方来源；没有检出同名工作只能把声明保留为 `PROVISIONAL`，不能证明“首个”；
- 仅保留冻结预测器、forecast horizon/causal lag 独立索引、变量轴联合真值、sequential unseen regime zero-refit 等仍有可证伪空间的窄声明。

失败处理：

- 若直接工作覆盖窄声明，更新新颖性表并收缩或终止对应主张；若仅剩金融数据集差异，终止方法创新方向。

### Gate A：固定位置干预

通过条件：

- 在冻结预测器和 held-out intervention pairs 上，oracle site 同时满足 Cause 与 Isolation；
- `source=base` 效应接近零，true lag 优于 wrong lag；
- 随机 site、随机模型和随机概念不能接近真实机制；
- 训练与 held-out pair 的差距在预注册范围内。

失败处理：

- 两轮概念/SCM 修复后仍失败，暂停自动定位并重构高层概念或 SCM。

### Gate 1：合成定位

通过条件：

- 在同一合成生成链上同时恢复 intervention truth 与四轴 location truth；
- 独立辨识 forecast horizon $h$ 与 causal lag $\delta$，并证明变量/通道轴提供独立信息；
- 窄化 TARCA 在联合真值上优于原始 PLOT、DAS 和随机定位，且成本低于 Full DAS；
- held-out pairs 保持有效；容量前沿增大时，随机模型和随机概念仍不能追平真实机制。

失败处理：

- 优先修复干预签名、联合真值和容量限制；两轮后仍失败则停止真实数据定位。

### Gate 2：跨状态解释

通过条件：

- 解释器、位置、normalizer 和映射在 sequential unseen regime 上全部保持 zero-refit；
- DRO 版本的未见状态最坏抽象误差稳定优于 ERM、Group-DRO、DiRoCA-style 和随机重加权；
- 平均性能不出现不可接受退化。

失败处理：

- 检查环境定义是否可识别；若读取测试状态标签或需要测试期 refit，终止鲁棒性主张。按状态单独拟合只能作为 oracle 上界，不能混入主证据。

### Gate 3：预测收益（探索性）

通过条件：

- 只在 Gate A/1/2 已通过后评估，且不参与关闭前述任何 Gate；
- 至少两个非金融域和一个金融任务上，最坏状态预测或校准有一致改善；
- 结果跨模型与随机种子稳定。

金融压力测试属于 RQ5/验证层；其结果只是 Gate 3 探索性预测收益判断的一项输入，不构成方法新颖性证据，也不得反向为 Gate A/1/2 补票。

失败处理：

- 若解释有效但预测无收益，保留并诚实报告纯机制解释结果，不强行声称预测收益；
- 若只有金融收益，不能支撑通用方法主张。

### Gate 4：论文完整性

通过条件：

- 理论、算法、基准、真实实验、负对照、开源材料均完成；
- 主要结论不依赖一个数据集或一个模型。

---

## 20. 九个月执行计划

| 月份 | 工作包 | 交付物 |
|---|---|---|
| 1 | 文献审计、问题形式化、统一数据契约与数据协议 | related-work 矩阵；预注册实验协议；契约测试；只消费 `src/tarca/contracts/` 的最小数据加载器 |
| 2 | 合成 SCM 与机制植入网络 | 可控生成器；真值干预与定位测试 |
| 3 | 冻结模型时序交换干预 | pyvene 适配；IIC/cause/isolation 指标 |
| 4 | 多轴渐进 OT 定位 | 层→时间/lag→变量→受限子空间算法；效率实验 |
| 5 | Regime-DRO 与理论 | 鲁棒目标；初版定理与证明 |
| 6 | 通用时间序列实验 | 至少三域结果；完整消融 |
| 7 | 订单簿与多资产实验 | 严格滚动协议；金融结果 |
| 8 | 负对照、统计检验、扩展性 | 反注入证据；置信区间；效率曲线 |
| 9 | 论文、代码与复现材料 | 投稿稿件；匿名仓库；实验清单 |

### 最小可行原型：前六个实施周（Stage 1 四周 + Stage 2 两周）

> 以下六周时钟横跨 Stage 1A/1B 和 Stage 2 起步：前四周属于 Stage 1，后两周属于
> Stage 2。Stage 1 应建立统一契约、合成 SCM、paired oracle 与工程 smoke；Stage 2
> 应建立基础预测器。任何后续模块都必须消费统一契约，不得自行定义第二套跨模块数据结构。

第 1 周（Stage 1A）：

- 实现统一数据契约、模型适配器静态协议和实验产物 Schema；
- 固定张量形状、特征名称、缺失掩码、UTC 时间边界和 split 泄漏规则；
- 完成契约测试，不下载数据、不训练模型。

第 2–3 周（Stage 1B）：

- 实现合成 regime-switching SCM 与事实 rollout；
- 定义两个概念：趋势、波动；
- 保存状态、未来噪声、真实延迟和生成配置。

第 4 周（Stage 1B）：

- 实现 paired counterfactual oracle；
- 使用相同未来噪声验证事实与反事实效应；
- 完成 synthetic easy 和 E01 工程真值 smoke。

> 实施状态不改写本计划的研究边界；E01 的当前冻结结论为 `v2/PASS`，证据与后续任务入口见
> `TARCA_E01_HANDOFF_SNAPSHOT_2026-08-30.md`。

第 5–6 周（Stage 2）：

- 实现 naive、VAR/DLinear 和小型概率预测器；
- 统一输出 `ForecastDistribution`；
- 只在预测有效后进入机制植入与固定位置交换干预，OT 和 DRO 仍不得提前。

---

## 21. 建议代码结构

```text
tarca/
├── configs/
│   ├── synthetic/
│   ├── generic_ts/
│   └── finance/
├── docs/
├── src/tarca/
│   ├── stage0/                    # Stage 0 研究契约、环境与证据
│   ├── contracts/                 # Stage 0 建立最小治理子集，Stage 1A 在同一位置扩展
│   │   ├── research.py
│   │   ├── governance.py
│   │   ├── data.py
│   │   ├── forecast.py
│   │   ├── concepts.py
│   │   ├── interventions.py
│   │   ├── adapters.py
│   │   ├── artifacts.py
│   │   └── arrow_schemas.py
│   ├── data/
│   │   ├── synthetic_scm.py
│   │   ├── generic_loaders.py
│   │   ├── lob_loader.py
│   │   ├── multi_asset_loader.py
│   │   └── leakage_checks.py
│   ├── models/
│   │   ├── patchtst_adapter.py
│   │   ├── itransformer_adapter.py
│   │   ├── chronos_adapter.py
│   │   └── planted_teacher.py
│   ├── concepts/
│   │   ├── temporal_concepts.py
│   │   ├── finance_concepts.py
│   │   ├── high_level_scm.py
│   │   └── concept_capacity.py
│   ├── interventions/
│   │   ├── temporal_swap.py
│   │   ├── lag_alignment.py
│   │   ├── source_matching.py
│   │   └── pyvene_adapter.py
│   ├── localization/
│   │   ├── effect_signatures.py
│   │   ├── progressive_ot.py
│   │   ├── subspace_search.py
│   │   └── das_refinement.py
│   ├── robustness/
│   │   ├── environments.py
│   │   ├── wasserstein_dro.py
│   │   └── worst_regime.py
│   └── metrics/
│       ├── forecasting.py
│       ├── abstraction.py
│       ├── localization.py
│       └── statistical_tests.py
├── experiments/
│   ├── synthetic/
│   ├── generic_ts/
│   └── finance/
├── artifacts/
├── tests/
└── README.md
```

### 21.1 统一数据契约原则

- Stage 0 应创建 `src/tarca/contracts/` 的最小治理子集，承载协议要求的 `StrictContractModel`、`Sha256Hash`、`ArtifactRef`、`ResearchContractManifest`、`GateStatus`、`GateSpec` 与 `GateDecision`。除 Gate 0 外，GateDecision 与相应 GateSpec hash 绑定；`GATE_0_NOVELTY 是人工新颖性 Gate，不要求 GateSpec`，直接绑定 novelty claims 与 related-work bundle。Stage 1A 在同一权威位置扩展数据、预测输出、概念、干预、模型适配器和 Arrow Schema，不得重建第二套基础类型。
- `data`、后续 `models` 等模块只能从 `src/tarca/contracts/` 导入，不得复制定义；Stage 0 不提前创建 Stage 1 的数据、SCM、模型、干预、OT 或 DRO 占位实现。
- 运行时张量载荷采用冻结 dataclass 与显式 shape/device/dtype 校验；JSON/Parquet 元数据采用严格、禁止额外字段的 Pydantic 模型。
- `WindowBatch` 必须包含特征名称、目标名称、显式缺失掩码和可审计时间边界，才能实际检查 known-future 目标泄漏和切分重叠。
- 可持久化契约必须携带语义版本；Parquet 文件使用 Arrow Schema 固定字段、类型、nullable 行为和契约版本 metadata。
- 模型适配器必须区分“可干预位置描述 `InterventionSite`”与“单次干预请求 `InterventionSpec`”，避免把位置目录和执行参数混为一类。
- Stage 0 研究契约负责约束实验边界、依赖锁、科学证据及最小治理类型；Stage 1A 统一数据/科学契约属于工程基础设施，不作为 TARCA 方法创新。

---

## 22. 算力与工程建议

### 资源发现与执行后端

- 当前本机、单机服务器、多卡服务器和其他等效资源都只是 Execution Plane backend，不构成项目算力上限；
- Stage 0 的强制工作应能在 CPU 上完成；无 GPU 只记录环境能力，不阻断研究契约和 Gate 0；
- 每个包含训练、大规模干预、上游复现或并行搜索的 Stage，在任务图冻结后先运行最小代表性 probe，再估计总运行时间、CPU/RAM、GPU 数量/显存、存储与并行度；
- 根据该次估计选择本地或服务器资源，不预先绑定某个 GPU 型号、数量或固定显存区间；资源不足时报告所需的最低/建议配置，不静默缩减科学任务；
- backend、worker 数和调度变化不得改变 seed、split、checkpoint、metric、Gate 或 scientific identity；服务器接入必须另行获得用户授权并遵守服务器 runbook。

### 工作负载分级

- **Stage 0 最小门禁**：静态检查、import、序列化、微型 PyTorch hook 与 2×2 Sinkhorn，CPU/offline；
- **最小 claim-bearing 实验**：合成 easy/medium、固定位置、基础定位与负对照；使用本地或服务器上的单加速器/等效资源，具体配置由 probe 决定；
- **完整实验**：多随机种子、多预测期、多模型族、Chronos 类冻结基线、跨域与全消融；按冻结任务图选择单机、多卡或分布式服务器。

### 降低成本

- 缓存基础模型表示；
- OT 先粗后细；
- 只对高质量 mass 区域运行 DAS；
- 使用低秩子空间；
- 先在短窗口和少变量上验证；
- 通过早停 Gate 避免在失效方向上继续消耗。

---

## 23. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 高层概念错设 | 抽象不忠实 | 多 SCM 组合；Good-Apples 式失败区域诊断 |
| 对齐映射过强 | 空洞解释 | 线性/低秩/MDL；随机模型负对照 |
| 解释器注入信息 | 虚假保真度 | 冻结模型、held-out 干预、单独预测测试 |
| 交换窗口不自然 | 干预失真 | 支持集匹配、同状态干预、条件生成匹配 |
| 状态标签泄漏 | OOD 结果失真 | 训练期拟合、在线状态检测、时间审计 |
| 金融噪声过大 | 结论不稳定 | 通用 TS 为主、金融为压力测试 |
| 基准饱和 | 改进缺乏意义 | 新 benchmark、跨域和最坏状态评估 |
| 文献快速碰撞 | 新颖性下降 | 每月检索、优先完成合成与理论原型 |
| 理论难以闭合 | CCF-A 强度不足 | 尽早证明简化版本；Gate 2 前完成 |
| 计算成本爆炸 | 无法完成消融 | PLOT 粗到细、表示缓存、低秩搜索 |

---

## 24. 推荐论文叙事

### 一句话问题

现有时间序列解释通常在平均分布上给出输入归因，但无法验证高层概念是否由模型内部机制实现，也无法保证该机制解释在状态切换后仍然成立。

### 一句话方法

TARCA 用高层动态 SCM 定义概念干预，以渐进最优传输按层、时间/lag、变量和受限子空间定位对应机制，并用分布鲁棒目标优化最坏环境的干预一致性。

### 一句话证据

在具有机制真值的合成模型和多个非金融序列域上，TARCA 的主要证据应证明定位准确、解释稳定且计算可扩展；订单簿/多资产任务用于金融压力测试。只有探索性 Gate 3 通过时，才附加声称分布外预测或聚合校准得到改善。

---

## 25. 关键论文与资料

### 25.1 研究主线

1. Geiger et al., **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**
   https://arxiv.org/abs/2301.04709

2. Geiger et al., **Inducing Causal Structure for Interpretable Neural Networks**
   https://arxiv.org/abs/2112.00826

3. Geiger et al., **Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations**
   https://arxiv.org/abs/2303.02536

4. Sun et al., **HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks**
   https://arxiv.org/abs/2503.10894

5. Felekis et al., **Distributionally Robust Causal Abstractions**
   https://arxiv.org/abs/2510.04842

6. Chang et al., **PLOT: Progressive Localization via Optimal Transport in Neural Causal Abstraction**
   https://arxiv.org/abs/2605.06979

7. Pîslar et al., **Combining Causal Models for More Accurate Abstractions of Neural Networks**
   https://arxiv.org/abs/2503.11429

8. Li et al., **Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction**
   https://arxiv.org/abs/2605.02234

9. Sutter et al., **The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?**
   https://arxiv.org/abs/2507.08802

10. Esponera & Cinnà, **Inducing Causal Structure for Interpretable Neural Networks Applied to Glucose Prediction for T1DM Patients**
    https://arxiv.org/abs/2503.14442

11. Gneiting & Raftery, **Strictly Proper Scoring Rules, Prediction, and Estimation**
    https://doi.org/10.1198/016214506000001437

12. Gneiting et al., **Assessing Probabilistic Forecasts of Multivariate Quantities, with an Application to Ensemble Predictions of Surface Winds**
    https://arxiv.org/abs/0806.0813

13. **Generalised Transportability via Causal Abstractions**
    https://arxiv.org/abs/2608.15645

14. **Validating Causal Abstraction Metrics on Simulated Complex Systems**
    https://arxiv.org/abs/2607.00267
    https://github.com/MelouxM/CAE

15. **Addressing divergent representations from causal interventions on neural networks**
    https://arxiv.org/abs/2511.04638
    https://github.com/grantsrb/rep_divergence

16. **Boundless Distributed Alignment Search: Machine Checking Semantic Content in Neural Networks**
    https://arxiv.org/abs/2305.08809

17. **Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning (FOIL)**
    https://arxiv.org/abs/2406.09130
    https://github.com/AdityaLab/FOIL

### 25.2 已排重方向示例

1. **Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models**
   https://arxiv.org/abs/2603.10071

2. **Mechanistic Interpretability for Transformer-based Time Series Classification**
   https://arxiv.org/abs/2511.21514

3. **TimeSAE: Causal Sparse Decoding for Faithful Explanations of Black-Box Time Series Models**
   https://arxiv.org/abs/2601.09776

4. **FinCast: A Foundation Model for Financial Time-Series Forecasting**
   https://arxiv.org/abs/2508.19609

5. **Hierarchical Information-Guided Spatio-Temporal Mamba for Stock Time Series Forecasting**
   https://arxiv.org/abs/2503.11387

6. **Generation of Synthetic Financial Time Series by Diffusion Models**
   https://arxiv.org/abs/2410.18897

7. **Enforcing Interpretability in Time Series Transformers: A Concept Bottleneck Framework**
   https://arxiv.org/abs/2410.06070

8. **ForecastCF: Counterfactual Explanations for Time Series Forecasting**
   https://arxiv.org/abs/2310.08137

9. **Temporal Data Meets LLM — Explainable Financial Time Series Forecasting**
   https://arxiv.org/abs/2306.11025

### 25.3 GitHub 与工程资源

1. **pyvene：PyTorch 内部干预库**
   https://github.com/stanfordnlp/pyvene

2. **POT：Python Optimal Transport**
   https://github.com/PythonOT/POT

3. **Time-Series-Library：通用时间序列模型基线**
   https://github.com/thuml/Time-Series-Library

4. **Chronos Forecasting：预训练时间序列预测模型**
   https://github.com/amazon-science/chronos-forecasting

5. **Captum：PyTorch 归因基线**
   https://github.com/pytorch/captum

6. **Darts：预测与数据管线参考**
   https://github.com/unit8co/darts

7. **PLOT 官方实现：包含 PLOT-DAS 与 Full DAS 比较**
   https://github.com/jchang153/causal-abstractions-ot

8. **DiRoCA 官方实现：Wasserstein 鲁棒因果抽象参考**
   https://github.com/yfelekis/DiRoCA

9. **CAE 官方实现：因果抽象指标与模拟系统评测参考**
   https://github.com/MelouxM/CAE

10. **causalab：因果抽象与机制干预实验框架**
    https://github.com/goodfire-ai/causalab

11. **FOIL：OOD 时间序列预测基线**
    https://github.com/AdityaLab/FOIL

12. **COGS：OOD 时间序列因果表征基线**
    https://github.com/simon-sxx/COGS

---

## 26. 立项后的首个完整科学验证里程碑（不是前六周 MVP）

Stage 1 应完成统一契约、合成 SCM 和 paired counterfactual oracle；Stage 2 随后实现基础预测器。前六周 MVP 不提前执行内部干预、OT 或 DRO。随后按 Gate A/1/2 的顺序完成以下最小科学验证：

1. 一个有状态切换的合成 SCM；
2. 一个带已知机制位置的教师 Transformer；
3. 两个高层概念；
4. 层 × 时间 patch 的两轴 PLOT；
5. 冻结模型交换干预；
6. 随机概念和随机模型负对照；
7. 简单两环境 DRO。

只有 Gate A、Gate 1 和 Gate 2 依次通过后，才进入真实金融数据。完成这些门后再判断该方向是否具有方法学潜力；不得把这一完整科学验证错误压缩为前六周任务。
