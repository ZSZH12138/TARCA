# TARCA：面向状态切换的时序因果抽象与分布鲁棒机制定位研究计划

> **计划版本**：v1.0
> **检索截点**：2026-07-14
> **目标**：形成一个以通用时间序列方法为主体、金融序列为高难度验证场景。

---

## 0. 执行结论

### 0.1 最终判断

本计划建议研究：

> **将神经因果抽象、渐进式最优传输机制定位和分布鲁棒优化统一到非平稳多变量时间序列中，构造可被干预验证、可定位到“层 × 变量 × 时间片 × 子空间”、并在状态切换下保持稳定的预测解释。**
### 0.2 CCF-A 级别判断

结论为：**条件性可行**。

仅做下列工作，不足以达到 CCF-A 型论文标准：

- 把 PLOT、DAS、IIT 或 DiRoCA 直接套到股票数据；
- 只展示若干可视化解释；
- 只在一个市场、一个数据集和一个预测任务上优于基线；
- 将模型内部干预结果错误地表述为真实市场因果关系；
- 仅提升收益率回测，没有通用方法、理论或机制验证。

要形成有竞争力的论文，必须同时完成：

1. **新的通用形式化**：定义面向序列到分布预测器的、随预测期变化的时序因果抽象；
2. **新的算法**：在层、变量、时间片、子空间四个维度上做渐进式机制定位；
3. **新的鲁棒性机制**：优化最坏状态/环境下的抽象误差，而非只优化平均解释保真度；
4. **理论结果**：至少给出近似抽象误差或最坏环境误差上界；
5. **反空洞性设计**：限制对齐映射容量，设置冻结模型、安慰剂概念和反信息注入实验；
6. **通用实证**：合成真值、非金融时间序列、金融时间序列三层证据；
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
6. 解释约束是否能改善分布外预测，而非只产生事后图表。

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
s=(\ell,d,p,U),
$$

分别表示层 $\ell$、变量/通道 d、 patch 或时间位置 p、表示子空间 U。

**H2**：从粗到细的最优传输定位比穷举 DAS 更高效，并能恢复合成模型中的已植入机制。

### RQ3：解释能否跨状态保持稳定？

**H3**：在 Wasserstein 模糊集合上最小化最坏环境抽象误差，可提高未见状态下的干预保真度。

### RQ4：解释约束能否改善分布外预测？(由于风险较大 后续不再深入探究 只当作是一个可以考虑的方向)

**H4**：当高层概念是任务相关、低容量且干预语义正确时，因果抽象正则能抑制状态特异的伪相关，提高最坏状态预测、校准和稳定性。

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
模型族 → 层 → 变量/通道 → 时间 patch → 表示子空间
```

为每个高层概念计算干预效应签名：

$$
e_k^{H}
=
\left[
\Delta\mu_{1:H},
\Delta\sigma_{1:H},
\Delta q_{\alpha,1:H},
\Delta \mathrm{calibration}
\right].
$$

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

高传输质量（mass）的候选区域进入下一层细化，最终再用 DAS/IIT 优化低维干预子空间。

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

联合目标：

$$
\begin{aligned}
\min_{\theta,\alpha,\pi}\quad
&\mathcal{L}_{\text{forecast}}
+\lambda\,\mathcal{L}_{\text{TII}}\\
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

- $\mathcal{L}_{\text{forecast}}$：预测损失；
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

若无法得到第三个理论结果，前两个结果仍可构成主理论；但必须通过强实验证明预测收益。

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
2. 对标签重叠任务使用 purging 与 embargo；
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
- 冻结模型与联合训练；
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

1. **Temporal Causal Abstraction**：首次系统定义适用于多步概率预测的、预测期与时延索引的神经因果抽象；
2. **Multi-axis Progressive Localization**：在层、变量、时间片和子空间上进行渐进 OT 机制定位；
3. **Regime-Robust Abstraction**：在环境/状态变化下最小化最坏抽象误差；
4. **Anti-vacuity Protocol**：用容量限制、冻结模式和负对照防止空洞对齐与信息注入；
5. **Abstraction-Regularized Forecasting**：验证机制一致性约束能否改善分布外预测；
6. **Benchmark**：提供有干预真值、位置真值和状态切换的合成基准；
7. **Financial Stress Test**：在订单簿和多资产预测中验证方法对强非平稳场景的价值。

---

## 18. CCF-A 型论文评估矩阵

| 维度 | 当前潜力 | 达标条件 | 否决条件 |
|---|---:|---|---|
| 问题重要性 | 8/10 | 解释保真度与状态鲁棒性被统一 | 只讨论金融可视化 |
| 方法新颖性 | 8/10 | 时序形式化、多轴定位、DRO 三者形成不可分割方法 | 仅拼接 PLOT 与 DiRoCA |
| 理论深度 | 7/10（条件性） | 至少两个非平凡结果及清晰假设 | 只有直观描述 |
| 实证强度 | 9/10（工作量高） | 合成真值、通用 TS、金融三层验证 | 单一股票数据集 |
| 可解释性可信度 | 8/10 | cause/isolation、负对照、反注入 | 只报 attribution 图 |
| 通用性 | 8/10 | 方法不依赖金融专有概念 | 标题和算法均为股票特化 |
| 可复现性 | 8/10 | 公共核心数据、开源代码、完整配置 | 依赖私有数据 |
| 竞争风险 | 6/10 | 月度文献审计和快速合成原型 | 2026 年新工作抢先覆盖 |

### 总体判断

- **作为“金融应用论文”**：不够；
- **作为“通用时序因果抽象方法 + 金融压力测试”**：具备冲击 KDD/AAAI/IJCAI 等高水平会议的研究形态；
- **若理论、基准和跨域实证均达到高质量**：可按更强机器学习会议的审稿强度准备；
- 这只是研究潜力判断，不构成录用保证。

---

## 19. 阶段性 Go/No-Go 门槛

### Gate 0：新颖性复核

通过条件：

- 没有直接覆盖“时序多轴 PLOT + regime-DRO causal abstraction”的新论文；
- 与最新工作有至少两项实质差异。

失败处理：

- 若仅剩金融数据集差异，终止本方向。

### Gate 1：合成定位

通过条件：

- 在已植入机制上，定位 F1/IIC 明显优于原始 PLOT、DAS 和随机定位；
- 计算成本低于穷举 DAS；
- 随机模型和随机概念不能获得高分。

失败处理：

- 优先修复干预签名和容量限制；两轮后仍失败则终止。

### Gate 2：跨状态解释

通过条件：

- DRO 版本在未见状态的最坏抽象误差上稳定优于 ERM；
- 平均性能不出现不可接受退化。

失败处理：

- 检查环境定义是否可识别；若只能依赖测试状态标签，终止鲁棒性主张。

### Gate 3：预测收益

通过条件：

- 至少两个非金融域和一个金融任务上，最坏状态预测或校准有一致改善；
- 结果跨模型与随机种子稳定。

失败处理：

- 若解释有效但预测无收益，可改投纯解释论文，但降低目标；
- 若只有金融收益，不能支撑通用方法主张。

### Gate 4：论文完整性

通过条件：

- 理论、算法、基准、真实实验、负对照、开源材料均完成；
- 主要结论不依赖一个数据集或一个模型。

---

## 20. 九个月执行计划

| 月份 | 工作包 | 交付物 |
|---|---|---|
| 1 | 文献审计、问题形式化、数据协议 | related-work 矩阵；预注册实验协议；数据加载器 |
| 2 | 合成 SCM 与机制植入网络 | 可控生成器；真值干预与定位测试 |
| 3 | 冻结模型时序交换干预 | pyvene 适配；IIC/cause/isolation 指标 |
| 4 | 多轴渐进 OT 定位 | 层→变量→时间→子空间算法；效率实验 |
| 5 | Regime-DRO 与理论 | 鲁棒目标；初版定理与证明 |
| 6 | 通用时间序列实验 | 至少三域结果；完整消融 |
| 7 | 订单簿与多资产实验 | 严格滚动协议；金融结果 |
| 8 | 负对照、统计检验、扩展性 | 反注入证据；置信区间；效率曲线 |
| 9 | 论文、代码与复现材料 | 投稿稿件；匿名仓库；实验清单 |

### 最小可行原型：前六周

第 1–2 周：

- PatchTST/iTransformer 基线；
- 合成 regime-switching 数据；
- 定义两个概念：趋势、波动；
- 在一个固定层做交换干预。

第 3–4 周：

- 用 OT 在层和时间 patch 上定位；
- 与穷举搜索比较；
- 加入随机概念负对照。

第 5–6 周：

- 加入两个环境；
- 比较 ERM 与简单 Wasserstein-DRO；
- 根据 Gate 1 决定是否继续。

---

## 21. 建议代码结构

```text
tarca/
├── configs/
│   ├── synthetic/
│   ├── generic_ts/
│   └── finance/
├── data/
│   ├── synthetic_scm.py
│   ├── generic_loaders.py
│   ├── lob_loader.py
│   ├── multi_asset_loader.py
│   └── leakage_checks.py
├── models/
│   ├── patchtst_adapter.py
│   ├── itransformer_adapter.py
│   ├── chronos_adapter.py
│   └── planted_teacher.py
├── concepts/
│   ├── temporal_concepts.py
│   ├── finance_concepts.py
│   ├── high_level_scm.py
│   └── concept_capacity.py
├── interventions/
│   ├── temporal_swap.py
│   ├── lag_alignment.py
│   ├── source_matching.py
│   └── pyvene_adapter.py
├── localization/
│   ├── effect_signatures.py
│   ├── progressive_ot.py
│   ├── subspace_search.py
│   └── das_refinement.py
├── robustness/
│   ├── environments.py
│   ├── wasserstein_dro.py
│   └── worst_regime.py
├── metrics/
│   ├── forecasting.py
│   ├── abstraction.py
│   ├── localization.py
│   └── statistical_tests.py
├── experiments/
│   ├── synthetic/
│   ├── generic_ts/
│   └── finance/
├── tests/
└── README.md
```

---

## 22. 算力与工程建议

### 原型

- 1–2 张 24–48 GB GPU；
- PatchTST、iTransformer 或中小型 Transformer；
- 合成数据和固定模型事后定位；
- 不在第一阶段训练大型基础模型。

### 完整实验

- 多随机种子、多预测期、多模型族；
- Chronos-2 作为冻结基础模型基线；
- 需要并行干预和缓存隐藏状态；
- 预算允许时使用 4 张 80 GB 级 GPU 或等效资源。

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

TARCA 用高层动态 SCM 定义概念干预，以渐进最优传输在层、变量、时间和子空间中定位对应机制，并用分布鲁棒目标优化最坏环境的干预一致性。

### 一句话证据

在具有机制真值的合成模型、多个非金融序列域及订单簿/多资产金融任务上，TARCA 应同时证明定位准确、解释稳定、计算可扩展，并改善分布外预测或校准。

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

### 25.2 已排重方向示例

1. **Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models**
   https://arxiv.org/abs/2603.10071

2. **Mechanistic Interpretability for Transformer-based Time Series Classification**
   https://arxiv.org/abs/2511.21514

3. **TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models**
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

---

## 26. 立项后的首个决策

第一阶段不应立即抓取大量股票数据，而应先完成以下最小科学验证：

1. 一个有状态切换的合成 SCM；
2. 一个带已知机制位置的教师 Transformer；
3. 两个高层概念；
4. 层 × 时间 patch 的两轴 PLOT；
5. 冻结模型交换干预；
6. 随机概念和随机模型负对照；
7. 简单两环境 DRO。

只有在合成定位和反空洞性检验通过后，才进入真实金融数据。这样可以在六周内判断该方向是具有方法学潜力，还是仅能产生表面可视化。
