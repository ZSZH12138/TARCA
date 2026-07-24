# TARCA Stage 0 文献审计日志

核查日期：`2026-07-23`
核查范围：截至核查日可访问的正式 proceedings/期刊页、arXiv、OpenReview、机构接收记录、官方文档和作者官方仓库。
证据规则：博客、论坛和营销页面不作为核心证据；无法由一手/官方来源核实的字段一律写 `UNVERIFIED`；没有核实到官方代码时写 `NO_OFFICIAL_CODE_FOUND`。

## 1. 强制研究主线（14/14）

| # | 研究主线/检索式 | 核查的一手或官方来源 | 结果 | 未解决字段 | 对 TARCA 的含义 |
|---:|---|---|---|---|---|
| 1 | `Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability` | JMLR 正式页与论文：https://jmlr.org/papers/v26/23-0058.html；arXiv：https://arxiv.org/abs/2301.04709 | `VERIFIED`：JMLR 26 (2025)，正式发表；建立一般因果抽象、近似抽象与交换干预理论 | paper-specific 官方代码为 `NO_OFFICIAL_CODE_FOUND` | 是理论基础，不包含多步概率预测、预测期/因果时延或 sequential regime 协议 |
| 2 | `Inducing Causal Structure for Interpretable Neural Networks` | PMLR/ICML 正式页：https://proceedings.mlr.press/v162/geiger22a.html | `VERIFIED`：ICML 2022；提出 IIT，用 base/source 交换训练神经模型实现预定高层因果结构 | paper-specific 官方代码为 `NO_OFFICIAL_CODE_FOUND` | TARCA 的主线是冻结预测器的事后解释，不能把 IIT 本身当创新 |
| 3 | `Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations` | PMLR/CLeaR 正式页：https://proceedings.mlr.press/v236/geiger24a.html；arXiv：https://arxiv.org/abs/2303.02536 | `VERIFIED`：CLeaR 2024；DAS 用梯度搜索分布式子空间对齐 | paper-specific 官方代码为 `NO_OFFICIAL_CODE_FOUND` | 分布式子空间对齐已存在；TARCA 必须增加 forecast-indexed variable/lag/horizon truth |
| 4 | `HyperDAS` | OpenReview/ICLR：https://openreview.net/forum?id=6fDjUoEQvm；arXiv：https://arxiv.org/abs/2503.10894；作者仓库：https://github.com/jiudingsun01/HyperDAS | `VERIFIED`：ICLR 2025 Poster；自动定位 residual-stream token position 并学习 concept feature | 代码许可证 `UNVERIFIED`；仓库 HEAD `ad89825abfdc6875f0c8529e64ca7c84f75e3946` | 自动位置搜索与信息注入风险讨论均已有最近邻；TARCA 的低容量与负对照必须更严格 |
| 5 | `PLOT: Progressive Localization via Optimal Transport` | OpenReview：https://openreview.net/forum?id=SYFm456yUL；arXiv：https://arxiv.org/abs/2605.06979；作者仓库：https://github.com/jchang153/causal-abstractions-ot | `VERIFIED`：ICML 2026 Mechanistic Interpretability Workshop；明确实现 output-effect signatures、OT/UOT 渐进定位和 PLOT-guided DAS | 代码许可证 `UNVERIFIED`；HEAD `96dbec5f04bc03aea6e55c430eeafd5c9be27fb2` | **直接碰撞**：PLOT-guided DAS 是 `NOT_NOVEL`；generic 四轴/渐进定位是 `COLLISION_RISK` |
| 6 | `Distributionally Robust Causal Abstractions / DiRoCA` | arXiv：https://arxiv.org/abs/2510.04842；Warwick 机构记录：https://wrap.warwick.ac.uk/id/eprint/200392/；作者仓库：https://github.com/yfelekis/DiRoCA | `VERIFIED`：ICML 2026 accepted/in press；Wasserstein ambiguity set 下的 constrained min-max causal abstraction | 代码许可证 `UNVERIFIED`；HEAD `7002947b4954abea1f3d11fcb6f36e7f3c43e8bd` | **直接碰撞**：一般 Wasserstein 分布鲁棒因果抽象已存在；仅可保留 frozen neural forecaster + sequential unseen regime + zero-refit + horizon/lag 的窄声明 |
| 7 | `Bucketing the Good Apples` | arXiv：https://arxiv.org/abs/2605.02234 | `VERIFIED`：2026 预印本；用 pairwise interchange consistency 划分解释成功/失败区域 | 官方代码 `NO_OFFICIAL_CODE_FOUND`；venue `UNVERIFIED` | TARCA 应预注册 pair-level 失败区域，不能只报全局 IIC |
| 8 | `The Non-Linear Representation Dilemma` | NeurIPS 正式页：https://papers.nips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html；arXiv：https://arxiv.org/abs/2507.08802；作者仓库：https://github.com/densutter/non-linear-representation-dilemma | `VERIFIED`：NeurIPS 2025；证明/展示不受限映射可使因果抽象空洞 | 代码许可证 `UNVERIFIED`；HEAD `7a8d538bd3015dd6e7dc6fecb8b6bd1c488dd304` | 容量约束、冻结、随机模型/概念、mapping-only probe 和 held-out pair 是必要证据，不是可选装饰 |
| 9 | `pyvene` | ACL Anthology：https://aclanthology.org/2024.naacl-demo.16/；官方文档：https://stanfordnlp.github.io/pyvene/；官方仓库：https://github.com/stanfordnlp/pyvene | `VERIFIED`：NAACL-HLT 2024 System Demonstration；Apache-2.0；HEAD `9e333904dcf9e597ca76170010d17f4d4580de8d` | 无 | 是干预工程底座，不是 TARCA 方法新颖性 |
| 10 | `POT: Python Optimal Transport` | JMLR：https://jmlr.org/papers/v22/20-451.html；官方文档：https://pythonot.github.io/；官方仓库：https://github.com/PythonOT/POT | `VERIFIED`：JMLR 2021；MIT；HEAD `a12794ea4cc5eba5f6d215e4d86757344c392788` | 论文精确日级发表日期 `UNVERIFIED` | 是 OT/UOT 数值库；不能为因果或 regime 结论背书 |
| 11 | 时间序列概念瓶颈方法 | arXiv：https://arxiv.org/abs/2410.06070；PMLR：https://proceedings.mlr.press/v182/wu22a.html；作者仓库：https://github.com/dtak/optimal-summaries-public | `VERIFIED/PARTIAL`：arXiv 2410.06070 当前标题为 `Interpretability for Time Series Transformers using A Concept Bottleneck Framework`，作者为 Angela van Sprang、Erman Acar、Willem Zuidema；forecasting CBM 和 clinical time-series CBM 均已存在 | forecasting CBM 官方代码 `NO_OFFICIAL_CODE_FOUND`；clinical repo license `UNVERIFIED` | “把概念瓶颈用于时序预测”不是 TARCA 新颖性 |
| 12 | 时间序列反事实解释方法 | arXiv/作者仓库：ForecastCF https://arxiv.org/abs/2310.08137、Native Guide https://arxiv.org/abs/2009.13211、CoMTE https://arxiv.org/abs/2008.10781、CounTS https://arxiv.org/abs/2306.06024 | `VERIFIED/PARTIAL`：ForecastCF 作者为 Zhendong Wang、Ioanna Miliou、Isak Samsten、Panagiotis Papapetrou；输入空间、实例替换、潜变量 counterfactual 和 forecasting-specific CF 均已存在 | 个别其他条目的作者、license、venue 仍为 `UNVERIFIED`，详见 CSV | TARCA 必须与输入/潜变量反事实区分，主张冻结预测器内部的高低层交换一致性 |
| 13 | 时间序列 SAE 或机制解释工作 | arXiv：TimeSAE https://arxiv.org/abs/2601.09776；TimeSAE-Lib landing：https://oublalkhalid.github.io/TimeSAE/；TS Transformer MI https://arxiv.org/abs/2511.21514；Dissecting Chronos https://arxiv.org/abs/2603.10071 | `VERIFIED`：`TimeSAE: Causal Sparse Decoding for Faithful Explanations of Black-Box Time Series Models` 已 accepted at ICML 2026；SAE、activation patching、head/timestep probing、forecast-feature ablation 已进入时间序列与 TSFM | TimeSAE landing 的 code link 当前解析到 anonymous 4open，commit/license `UNVERIFIED`；其他条目的正式 venue/repo/license 仍按 CSV 谨慎记录 | generic “time-series mechanistic interpretability/SAE” 已碰撞；TARCA 只能主张更窄的 causal-abstraction contract |
| 14 | 与 TARCA 最接近的最新时序因果抽象工作 | 上述 PLOT、DiRoCA、TimeSAE、TS Transformer MI、Dissecting Chronos；同时检索 arXiv/OpenReview/PMLR 的术语组合 | `NO_EXACT_DIRECT_MATCH_FOUND`：未核实到把 interchange abstraction、多步概率预测、独立 horizon/lag、variable/subspace truth 与 unseen-regime zero-refit 全部合并的工作 | 同义词覆盖不可能证明穷尽；结论不是“新颖性证明” | 仍有窄开口，但必须持续月度复核，并把 PLOT/DiRoCA/TimeSAE 作为强最近邻 |

## 2. 七条指定碰撞查询（逐字记录）

检索日期均为 `2026-07-23`。检索范围为 arXiv、OpenReview、PMLR、JMLR、NeurIPS proceedings 与作者官方 GitHub；无直接结果时，记录最接近的一手来源而不将“未搜到”写成新颖性证明。

| exact_query | sources_inspected | result | unresolved_fields | conclusion |
|---|---|---|---|---|
| `temporal causal abstraction` | arXiv、OpenReview、PMLR、JMLR；最近邻为 JMLR causal-abstraction foundation、PLOT、TimeSAE | `NO_EXACT_DIRECT_MATCH_FOUND` for a neural forecasting framework | 未使用该字面短语的同义工作可能存在 | 仅支持继续窄化检索；不支持“首个”的确定陈述 |
| `time-series causal abstraction` | arXiv、OpenReview、PMLR；并交叉检查 TS CBM/CF/SAE/MI | `NO_EXACT_DIRECT_MATCH_FOUND` as an established method label | 同义词与未来更新 | 术语尚非稳定标签；nearest works 必须仍纳入比较 |
| `interchange intervention time series` | arXiv、OpenReview、PMLR；检查 IIT/DAS/PLOT 与 TS CBM/MI | `NO_EXACT_DIRECT_MATCH_FOUND` for multi-step neural forecasting | 可能存在未使用 exact phrase 的 activation-patching 工作 | 当前最有希望的开口是 internal interchange + forecast distribution，而不是 generic patching |
| `progressive optimal transport localization time series` | arXiv、OpenReview、PMLR；最近邻 PLOT https://arxiv.org/abs/2605.06979 | `NO_EXACT_TIME_SERIES_MATCH_FOUND`; generic progressive OT localization is directly covered by PLOT | 时间序列同义表述可能存在 | TARCA 不能声称 progressive OT localization 本身新颖；必须检验 variable/causal-lag/horizon/subspace 联合真值 |
| `regime robust causal abstraction` | arXiv、OpenReview、PMLR；最近邻 DiRoCA https://arxiv.org/abs/2510.04842 | exact phrase 无独立时序方法；generic robust causal abstraction 已被 DiRoCA 覆盖 | sequential-regime 近义词可能存在 | 只保留 frozen forecaster、unseen sequential regime、zero-refit 的限定声明 |
| `distributionally robust mechanistic interpretability` | arXiv、OpenReview、PMLR/JMLR；最近邻 DiRoCA 与 TimeSAE | exact phrase 无核实到完整方法；DiRoCA 已覆盖 DRO causal abstraction，TimeSAE 已讨论 shift-robust TS explanation | mechanistic-interpretability 同义工作可能存在 | 不得把 Wasserstein-DRO + interpretability 的拼接写成新颖性 |
| `multi-axis neural mechanism localization` | arXiv、OpenReview、PMLR；最近邻 PLOT、HyperDAS、TS Transformer MI | `NO_EXACT_DIRECT_MATCH_FOUND`; component axes are already covered across nearest works | “axis/site/position/channel”同义词覆盖不完全 | generic multi-axis localization 为 `COLLISION_RISK`; TARCA 必须使用 forecast-indexed truth 定义 |

## 3. 补充排重查询

| query | sources_inspected | result |
|---|---|---|
| `forecasting concept bottleneck` | arXiv 2410.06070 | 已有 Angela van Sprang、Erman Acar、Willem Zuidema 的时序预测 concept-bottleneck 与 intervention proof of concept |
| `time series forecasting counterfactual` | arXiv 2310.08137、2306.06024 及作者仓库 | 已有 Zhendong Wang、Ioanna Miliou、Isak Samsten、Panagiotis Papapetrou 的 ForecastCF 与 CounTS；主要不是高低层 causal abstraction |
| `time series sparse autoencoder mechanistic interpretability` | arXiv 2601.09776、2603.10071、2511.21514；TimeSAE landing | TimeSAE 已 accepted at ICML 2026；另有 Chronos SAE ablation 与 TS Transformer activation patching |
| `PLOT-guided DAS` | PLOT OpenReview/arXiv/官方仓库 | 明确已实现；TARCA 该项为 `NOT_NOVEL` |
| `Distributionally Robust Causal Abstractions` | arXiv、Warwick 机构记录、官方仓库 | ICML 2026 accepted/in press；generic 声明直接碰撞 |

## 4. 截止本次审计的硬结论

1. `PLOT-guided DAS` 为 `NOT_NOVEL`，只能作为基线/复现对象。
2. 泛化的“层 × 时间/位置 × 子空间渐进定位”已与 PLOT/HyperDAS 高度碰撞；TARCA 必须改为 **forecast-indexed variable × causal lag × forecast horizon × constrained subspace truth**。
3. 泛化的 Wasserstein 分布鲁棒因果抽象已与 DiRoCA 直接碰撞；TARCA 只能研究 **frozen neural time-series forecasters + sequential unseen regimes + zero-refit + horizon/lag fidelity**。
4. 金融应用不是方法新颖性，只能是 point-in-time、强非平稳压力测试。
5. PLOT 与 DiRoCA 代码许可证截至 `2026-07-23` 均为 `UNVERIFIED`；不得假定可复制或改编。
6. 任一 `NO_EXACT_DIRECT_MATCH_FOUND` 都只是检索结果，不是数学或法律意义上的新颖性证明。
