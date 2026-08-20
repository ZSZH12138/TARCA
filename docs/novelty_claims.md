# TARCA Stage 0 新颖性声明

> 核验截点：2026-08-20
> 状态含义：`PROVISIONAL` 仅表示尚未发现直接覆盖且存在可证伪差异，不表示已证明创新。任何后续直接碰撞都必须先更新本文件，再继续正式实现。

<!-- TARCA_NOVELTY_CLAIMS_YAML_BEGIN -->
```yaml
schema_version: "1.0.0"
verification_date: "2026-08-20"
claims:
  - claim_id: TARCA-C1
    status: PROVISIONAL
    nearest_work: [ca-2301.04709, iit-2112.00826, das-2303.02536, plot-2605.06979, timesae-2601.09776]
    falsification: orthogonal horizon-by-lag held-out intervention experiment
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C2
    status: PROVISIONAL
    nearest_work: [plot-2605.06979, hyperdas-2503.10894, das-2303.02536, bdas-2305.08809]
    falsification: joint variable-lag-horizon-subspace truth recovery against PLOT and Full DAS
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C3
    status: PROVISIONAL
    nearest_work: [diroca-2510.04842, transport-2608.15645, foil-2406.09130, cogs-aaai-2026, timesae-2601.09776]
    falsification: sequential unseen-regime worst abstraction error with every explanatory component frozen
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C4
    status: REQUIRED_SUPPORTING_CONTRIBUTION
    nearest_work: [hyperdas-2503.10894, nonlinear-dilemma-2507.08802, good-apples-2605.02234, cae-2607.00267, rep-divergence-2511.04638, id-assumptions-2605.08012]
    falsification: capacity-matched random controls plus unmapped-variable and representation-support checks
    failure_action: REPAIR
excluded_claims:
  - claim_id: TARCA-N1
    status: NOT_NOVEL
    nearest_work: [plot-2605.06979]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N2
    status: NOT_NOVEL
    nearest_work: [plot-2605.06979]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N3
    status: NOT_NOVEL
    nearest_work: [diroca-2510.04842, transport-2608.15645]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N4
    status: NOT_NOVEL
    nearest_work: [cae-2607.00267]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N5
    status: NOT_NOVEL
    nearest_work: [ts-mech-2511.21514, timesae-2601.09776, chronos-sae-2603.10071, ts-cbm-2410.06070]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N6
    status: DROP_CLAIM
    nearest_work: []
    failure_action: STOP
```
<!-- TARCA_NOVELTY_CLAIMS_YAML_END -->

## 1. Gate 0 判断规则

- `PROVISIONAL`：保留为候选贡献，必须通过预注册的正式证伪实验；
- `REQUIRED_SUPPORTING_CONTRIBUTION`：科学可信度所必需，但不单列为首要方法创新；
- `KEEP_AS_BASELINE`：已有方法，仅作为比较或组件；
- `NOT_NOVEL`：已有工作直接覆盖，不得换名恢复；
- `DROP_CLAIM`：删除对应主张；若只剩应用场景差异，停止方法创新路线。

## 2. 保留的窄声明

### TARCA-C1：冻结多步概率预测器上的 horizon/lag 独立因果抽象

- **状态**：`PROVISIONAL`
- **声明**：在冻结神经时序预测器上，以概率预测输出为结果，分别索引 forecast horizon 与 causal lag，并在 held-out intervention pairs 上检验高层干预与内部干预的一致性。
- **最近邻**：Causal Abstraction、IIT、DAS、PLOT、时间序列概念瓶颈、TimeSAE。
- **实质差异**：最近邻没有同时固定预测器、采用多步概率输出，并把“预测未来第几步”和“机制经过多长因果时延生效”作为两个正交实验轴。
- **支持证据**：`ca-2301.04709`、`iit-2112.00826`、`das-2303.02536`、`plot-2605.06979`、`timesae-2601.09776`。
- **可证伪实验**：在预注册的 horizon × lag 正交网格上，固定其他因素；比较 true-lag、wrong-lag、source=base、随机 site，并在未参与拟合的 intervention pairs 上报告连续分布效应误差。
- **否决条件**：horizon 与 lag 的效果不能被实验独立识别；或 wrong-lag 与 true-lag 无可重复差异；或最新工作直接覆盖同一设置。
- **失败动作**：`DROP_CLAIM`，退化为一般时序 IIT/DAS 应用，不作为主贡献。

### TARCA-C2：forecast-specific 四轴联合真值定位

- **状态**：`PROVISIONAL`
- **声明**：在具有已知植入机制的时序预测器上，恢复 variable × causal lag × forecast horizon × constrained subspace 的联合位置真值。
- **最近邻**：PLOT、HyperDAS、DAS、Boundless DAS。
- **实质差异**：PLOT 已覆盖从 timestep/layer 到 coordinate/PCA span 的渐进 OT；TARCA 只保留具有独立 forecast horizon/causal lag 和变量轴联合真值的预测专用问题。
- **支持证据**：`plot-2605.06979`、`hyperdas-2503.10894`、`das-2303.02536`、`bdas-2305.08809`。
- **可证伪实验**：在联合植入真值上同时计算 layer/time/variable/subspace recovery、lag error、held-out abstraction error、干预次数和运行成本，并与 PLOT、Full DAS、随机/穷举基线比较。
- **否决条件**：只能恢复 PLOT 已有的 layer/timestep 轴；联合轴不能优于随机或可分解基线；受限子空间不能在 held-out pairs 上保持 Cause/Isolation。
- **失败动作**：删除“四轴方法”声明，仅保留 PLOT 复现和时序 benchmark 结果。

### TARCA-C3：sequential unseen regime 上的 zero-refit 抽象鲁棒性

- **状态**：`PROVISIONAL`
- **声明**：冻结 predictor、位置、映射、normalizer 和解释器，在顺序到达的未见状态上不重拟合，评估最坏因果抽象误差。
- **最近邻**：DiRoCA、Generalised Transportability、FOIL、COGS、TimeSAE。
- **实质差异**：DiRoCA 和 transportability 已覆盖一般分布鲁棒 causal abstraction；FOIL/COGS 关注预测模型的 OOD 学习。TARCA 只保留冻结时序预测器内部机制解释在 sequential unseen regime 上的 zero-refit 验证。
- **支持证据**：`diroca-2510.04842`、`transport-2608.15645`、`foil-2406.09130`、`cogs-aaai-2026`、`timesae-2601.09776`。
- **可证伪实验**：按预注册顺序暴露 unseen regimes，禁止任何 test-time fit；报告每状态与最坏状态的抽象误差、预测误差和覆盖率，并比较 ERM、Group-DRO、DiRoCA-style、FOIL/COGS 可比设置和随机重加权。
- **否决条件**：任一解释组件在未见状态上重拟合；改进只来自重新训练 predictor；或最坏抽象误差没有稳定改善。
- **失败动作**：删除 robust-abstraction 主张，转为 failure-region 诊断研究。

### TARCA-C4：反空洞和反信息注入协议

- **状态**：`REQUIRED_SUPPORTING_CONTRIBUTION`
- **声明**：所有 claim-bearing 映射与定位必须限制容量、冻结标签访问路径，并通过随机模型/概念/site、held-out pair、未映射变量 faithfulness 和干预后表示支持检查。
- **最近邻**：HyperDAS、Non-Linear Representation Dilemma、Good Apples、CAE、Representational Divergence、Identification Assumptions position paper。
- **实质差异**：不是声称首次提出这些诊断，而是把它们组合成 TARCA 所有 Gate 的强制反空洞条件。
- **支持证据**：`hyperdas-2503.10894`、`nonlinear-dilemma-2507.08802`、`good-apples-2605.02234`、`cae-2607.00267`、`rep-divergence-2511.04638`、`id-assumptions-2605.08012`。
- **可证伪实验**：比较线性/低秩/高容量映射；对随机初始化预测器、随机概念、随机 site、错 lag、source=base 和故意泄漏版本执行同一指标；检查干预表示的支持距离与未映射变量 faithfulness。
- **否决条件**：随机或无能力模型接近真实模型；映射器单独预测标签；效果只存在于表示分布外；未映射变量被静默忽略。
- **失败动作**：Gate A/1/2 不得通过，重构映射和干预协议。

## 3. 明确排除或降级的声明

| claim_id | 状态 | 原因 | 允许用途 |
|---|---|---|---|
| TARCA-N1 通用渐进 OT 定位 | `NOT_NOVEL` | PLOT 已直接覆盖 coarse-to-fine OT localization | `KEEP_AS_BASELINE` |
| TARCA-N2 PLOT-guided DAS | `NOT_NOVEL` | PLOT 官方方法已经包含 | `KEEP_AS_BASELINE` |
| TARCA-N3 一般 Wasserstein/分布鲁棒 causal abstraction | `NOT_NOVEL` | DiRoCA 与 Generalised Transportability 直接覆盖 | `KEEP_AS_BASELINE` |
| TARCA-N4 通用 causal-abstraction metric 或模拟 benchmark | `NOT_NOVEL` | CAE 已系统验证 30+ 指标并提供十个系统与显式 faithfulness | 使用 CAE 作为评测最近邻；只保留 forecast-specific 联合真值 benchmark |
| TARCA-N5 首次时间序列机制解释 | `NOT_NOVEL` | 已有时间序列 activation patching、TimeSAE、Chronos SAE 与 concept bottleneck | 作为相关工作与基线 |
| TARCA-N6 金融因果抽象应用即方法创新 | `DROP_CLAIM` | 应用域差异不能构成方法新颖性 | 金融只作末期压力测试 |

## 4. 检索局限与 Gate 0 解释

本次是针对直接碰撞、最近邻和官方实现的定向一手来源审计，不是覆盖所有数据库的系统综述。精确查询未发现同时满足“冻结概率时序预测器 + horizon/lag 独立 + 联合位置真值 + sequential unseen zero-refit”的直接工作，因此 TARCA-C1～C3 可暂时保留，但这一结论只支持 `PROVISIONAL`。经人工核验签发 GateDecision 后，Stage 1A 前不要求例行追加文献或自动重跑 Gate 0；只有用户要求复核，或明确的直接碰撞证据进入项目时，才替换该决策。
