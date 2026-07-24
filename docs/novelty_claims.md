# TARCA 新颖性声明（Stage 0）

核查日期：`2026-07-23`

允许的证据状态仅为：`SUPPORTED`、`PARTIALLY_SUPPORTED`、`UNRESOLVED`、`COLLISION_RISK`、`NOT_NOVEL`。
这里的“支持”仅表示截至核查日的一手来源比较保留了可检验空间，不等于证明新颖。金融应用从不构成实质方法差异。

| claim_id | TARCA 声明 | 当前证据等级 | 最近邻工作 | 最近邻已实现内容 | TARCA 必须保留的实质差异 | 可证伪实验 | 风险 |
|---|---|---|---|---|---|---|---|
| N1 | 多步概率输出的时序交换干预误差 | `PARTIALLY_SUPPORTED` | Causal Abstraction foundation；IIT；DAS；PLOT；ForecastCF；时序 CBM | 已定义交换干预/近似抽象、输出 effect signature、forecast counterfactual 与 forecasting concept intervention | 在**冻结神经时序预测器**上，把联合/边际预测分布的高低层干预差异明确分解到 forecast horizon，并与输入空间 CF 区分 | 在同一冻结概率预测器和 held-out pairs 上比较 TARCA internal interchange、ForecastCF/input edits、IIT/DAS/PLOT 基线；独立报告各 horizon 的 Wasserstein/energy/CRPS 差异 | 若只用点预测、单步分类或输入编辑，该声明退化为已有工作 |
| N2 | 预测期索引与因果时延索引的因果抽象 | `COLLISION_RISK` | PLOT；TS Transformer mechanistic interpretability；时序 CBM | PLOT 已定位 timestep；时序 MI 已分析 timestep/attention head；时序模型已使用预测期 | 必须将 **forecast horizon \(h\)** 与 **causal lag \(\delta\)** 定义为两种不同语义，并提供独立真值和干预支持集，而非把两者统称 timestep | 在合成真值中正交改变 \(h\) 与 \(\delta\)，比较 two-index TARCA 与单一 timestep PLOT-style 表述的恢复率和 held-out IIC | 若 horizon 与 lag 在实现或数据中不可辨识，该项只是 PLOT timestep 的改名 |
| N3 | 层 × 变量 × 时间片 × 子空间的渐进定位 | `COLLISION_RISK` | PLOT；HyperDAS；DAS；TS Transformer MI | 已有 token/timestep/layer→coordinate/PCA 渐进定位、token-position 自动定位和分布式子空间搜索 | 收缩为 **forecast-indexed variable × causal lag × forecast horizon × constrained subspace truth**；必须含显式变量/通道真值、时序效应几何和四轴联合植入真值 | 在同时植入变量、lag、horizon、低秩子空间真值的教师网络上，与原始 PLOT、HyperDAS、DAS 和随机定位比较 F1/IoU/principal-angle/IIC/成本 | 若只是 layer→patch→PCA，或没有联合位置真值，则与 PLOT 基本撞车 |
| N4 | PLOT 引导的 DAS | `NOT_NOVEL` | PLOT | PLOT 论文明确提出并实现 PLOT-guided DAS，用候选缩减加速 DAS | 无可保留的方法新颖性；只能作为强基线/复现对象，TARCA 的新增部分必须位于 forecast-indexed truth 与约束协议 | 仅复现/比较 PLOT-guided DAS 与 Full DAS、TARCA narrow variant；不得把胜负改写成“首次提出 PLOT-guided DAS” | 若继续列为贡献，会构成直接不实新颖性陈述 |
| N5 | 状态切换下的分布鲁棒抽象 | `COLLISION_RISK` | DiRoCA；TimeSAE | DiRoCA 已实现 Wasserstein ambiguity-set 的分布鲁棒 causal abstraction；TimeSAE 已研究 shift 下的时序解释鲁棒性 | 仅保留 **frozen neural time-series forecaster + sequential regimes + unseen-regime zero-refit + horizon/lag intervention fidelity**；不能把测试 regime 标签输入解释器 | 在合成 sequential regimes 上比较 ERM、Group-DRO、DiRoCA-style objective、随机重加权和 TARCA；所有解释器训练后冻结，在 unseen regime 不重拟合 | 若声明仍是 generic Wasserstein causal abstraction，或改善依赖测试状态标签，则已被 DiRoCA 覆盖/实验无效 |
| N6 | 容量限制与反信息注入协议 | `PARTIALLY_SUPPORTED` | Non-Linear Representation Dilemma；HyperDAS；PLOT | 已证明/展示高容量映射可造成空洞抽象；HyperDAS/PLOT 已讨论或部分缓解信息注入与校准 | 把线性/低秩/结构稀疏容量、冻结预测器、标签访问审计、mapping-only probe、随机模型/概念/SCM、source=base、wrong lag、held-out pairs 和容量前沿统一为强制协议 | 对真实植入机制与随机模型/随机概念做容量扫描；要求真实机制在 held-out pairs 上保留优势，而 mapping-only probe 与泄漏版本暴露注入 | 该项更可能是严格评估协议贡献；若随机模型或高容量映射同样高分，全部机制解释主张应停止 |
| N7 | 合成位置真值与干预真值基准 | `COLLISION_RISK` | PLOT；DiRoCA；TimeSAE；Good Apples | PLOT 有已知抽象/定位结构，DiRoCA 有成对 synthetic SCM，TimeSAE 有 synthetic explanation，Good Apples 有 pair-level 失败诊断 | 必须**同时**提供 temporal intervention truth、四轴 planted location truth、regime-switch truth、delay/cross-variable propagation 和预注册难度轴 | 构造同一生成链中的 SCM oracle 与 planted teacher network；分别测 effect fidelity、site recovery、failure buckets、跨难度退化和负对照 | 若只提供 SCM truth 或只提供 planted site，或只是现有 toy benchmark 的时间序列换皮，差异不足 |
| N8 | 金融序列作为强非平稳压力测试 | `NOT_NOVEL` | 金融预测/解释文献；DiRoCA 的鲁棒性目标 | 金融时序、状态漂移、滚动评估和非平稳压力测试本身已有广泛先例 | 没有方法新颖性；研究价值仅来自严格 point-in-time、rolling-origin、purging/embargo、失败状态报告，并验证同一方法先在非金融域成立 | 同一冻结协议至少覆盖两个非金融域和一个金融任务；比较跨状态失败而非只选正收益 | 若“未用于金融”被写为实质差异，或内部一致性外推为市场因果，该声明无效 |

## 当前决策

- 删除 N4 的贡献地位；后续只作为基线。
- N2、N3、N5、N7 在完成窄化前均按 `COLLISION_RISK` 管理。
- N1 与 N6 仅是 `PARTIALLY_SUPPORTED` 的可检验方向。
- N8 作为方法新颖性是 `NOT_NOVEL`，但可保留为验证层压力测试。
- 任何后续检索若发现直接覆盖，必须更新本表并触发 Gate 0；不得用“金融应用”回避碰撞。
