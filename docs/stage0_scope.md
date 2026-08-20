# TARCA Stage 0 范围合同

## 1. 能力目标

Stage 0 将研究方向、证据边界、实验预注册、环境锁和第三方版本冻结为机器可验证的 `ResearchContractManifest`，并由 `GateDecision(gate_id="GATE_0_NOVELTY")` 决定是否允许交接 Stage 1A。

## 2. 权威输入

1. `docs/auth/TARCA_项目计划书.md` v1.3；
2. `docs/auth/TARCA_具体实施计划.md` v1.3；
3. `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md` v2.0；
4. Stage 0 检索得到的一手论文、作者项目页和官方仓库；
5. 本地环境事实；服务器事实仅在另行授权后纳入。

发生冲突时，研究问题和 Gate 服从项目计划书，类型与接口服从协议，执行顺序服从具体实施计划。计划与协议不记录实施状态。

## 3. 范围内

- 初始化 Git 与最小 Python 工程；
- 记录查询式、纳排规则和证据截点；
- 创建 related-work matrix、novelty claims、terminology、assumption ledger 和 preregistration；
- 实现 Stage 0 所需的最小治理、Artifact 和 Gate 契约；
- 冻结依赖、第三方仓库 commit、许可证状态和允许动作；
- 执行 CPU/offline 最小 doctor、合同校验和测试，并验证经人工核验签发的 Gate 0 决策。

## 4. 范围外

- 正式数据下载、数据清洗、SCM 与反事实 oracle；
- 预测模型训练、checkpoint 选择或正式模型下载；
- 内部干预、PLOT、DAS、DRO 或正式上游复现；
- 金融回测或真实世界因果声明；
- 未获单独授权的服务器连接；
- 把当前本机、某台服务器或某种 GPU 写成项目算力上限。

## 5. 检索协议

### 5.1 证据截点

- 核验日期：2026-08-20；
- 快速变化的 2025–2026 工作按预印本、会议页和官方仓库分别记录版本；
- 经人工核验签发 GateDecision 后，Stage 1A 前不例行追加检索或自动重跑 Gate 0；只有用户要求复核或明确的直接碰撞证据进入项目时才替换决策。

### 5.2 来源等级

1. 论文原文、出版方页面、作者项目页；
2. 作者或组织维护的官方 GitHub 仓库；
3. 二手资料只能发现来源，不得单独支持或否决 claim。

### 5.3 实际查询族

```text
"causal abstraction" IIT DAS "distributed alignment search"
"progressive localization" "optimal transport" "causal abstraction"
"distributionally robust causal abstraction" Wasserstein transportability
"causal abstraction metrics" simulated systems "unmapped variables"
"mechanistic interpretability" "time series" forecasting intervention
"sparse autoencoder" "time series" causal explanation
"concept bottleneck" "time series forecasting"
"out-of-distribution" "time series forecasting" invariant learning regime
"sequential unseen regime" zero-refit forecasting
"non-linear representation dilemma" causal abstraction information injection
"representational divergence" model interventions
```

查询站点包括 arXiv、会议/作者项目页和 GitHub。精确组合词未检出直接结果只能支持 `PROVISIONAL`，不能证明“首次”。

### 5.4 纳入规则

- 直接定义或学习 causal abstraction；
- 执行内部干预、DAS、渐进定位或反空洞诊断；
- 处理 causal abstraction 的 distribution shift、transportability 或 metric；
- 对时间序列模型做机制干预、SAE、概念瓶颈或反事实解释；
- 为 OOD 时间序列预测提供直接最近邻基线；
- 提供 Stage 0 可能消费的官方软件实现。

### 5.5 排除规则

- 没有一手来源支撑的二次总结；
- 与模型内部机制、时间序列或鲁棒性均无关系的宽泛因果论文；
- 仅因使用金融数据而宣称方法创新的工作；
- 无法确认来源身份且不能作为反例复核的仓库镜像。

## 6. 固定不变量

- PLOT-guided DAS 是基线，不是 TARCA 新颖性；
- 一般 Wasserstein causal abstraction 不是 TARCA 新颖性；
- 通用 causal abstraction metric/模拟 benchmark 不是 TARCA 新颖性；
- 金融只作为压力测试；
- 所有保留 claim 在 Gate 0 后仍只标为 `PROVISIONAL`，直到正式证伪实验完成；
- 若明确的直接碰撞证据进入项目，先更新 novelty claim，再继续对应下游实现。

## 7. 交付与退出

Stage 0 必须生成协议规定的研究 ArtifactRef、`ResearchContractManifest` 和 Gate 0 决策。只有 Gate 0 `PASS` 才允许交接 Stage 1A；`FAIL` 时收窄/删除 claim 或停止对应方法路线。
