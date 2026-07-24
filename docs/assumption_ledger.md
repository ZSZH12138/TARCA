# TARCA 假设台账

核查日期：`2026-07-23`
说明：所有标为“Stage 1 计划测试”的检查只冻结未来测试契约，本阶段未实现测试或算法。

| assumption_id | 假设内容 | 层级 | 为什么需要 | 如何检查 | 破坏的结论 | 可否放宽 | 对应代码或测试 | 当前状态 |
|---|---|---|---|---|---|---|---|---|
| A01 | 输入窗口只使用预测时可获得的信息 | 数据/实验 | 防止 look-ahead leakage 把未来信息伪装成预测或机制 | 对每个输入字段做 availability timestamp 审计；故意未来泄漏版本只作 sanity check | 所有预测、IIC、状态鲁棒与金融结论 | 否 | Stage 1 计划：`test_point_in_time_inputs`、rolling-origin/purging/embargo 审计 | `FROZEN_PENDING_STAGE1_TEST` |
| A02 | 高层概念只使用历史窗口 | 数据/理论 | 高层干预必须在预测时可定义，否则高层 oracle 已泄漏 | 检查 concept 函数的最大时间索引；对 validation/test 禁止 fit | RQ1、RQ3、任何概念机制声明 | 否 | Stage 1 计划：`test_concept_uses_history_only` | `FROZEN_PENDING_STAGE1_TEST` |
| A03 | 每个高层干预具有明确支持集 | 理论/实验 | 支持集外 swap 可能不自然或没有因果语义 | 为 concept 干预声明允许值域、状态/重叠条件与拒绝原因；报告支持外比例 | 近似抽象误差的可解释性与外推范围 | 部分；可扩大支持集但须重新验证 | Stage 1 计划：`test_intervention_support_validation` | `FROZEN_PENDING_STAGE1_TEST` |
| A04 | 来源窗口与基础窗口可进行有意义匹配 | 数据/实验 | 未控制的非目标差异会污染交换效应 | 报告 matching distance、overlap、same/cross-regime 分层；比较错误 source 和 source=base | Cause、Isolation、lag 效应与 regime 比较 | 部分；低重叠 pair 可拒绝并报告 | Stage 1 计划：`test_source_base_matching` | `FROZEN_PENDING_STAGE1_TEST` |
| A05 | 干预表示不会修改预测器参数 | 算法 | 主证据要解释冻结模型的原有计算，而不是把机制训练进去 | 干预前后比较参数 hash、`requires_grad`/optimizer 参数集合和 checkpoint | “模型原有机制”与 zero-refit 结论 | 否（联合训练只能作为次要模式另报） | Stage 1 计划：`test_intervention_preserves_model_parameters` | `FROZEN_PENDING_STAGE1_TEST` |
| A06 | 对齐映射受线性、低秩或结构稀疏容量约束 | 理论/算法 | 不受限映射可使任意模型产生空洞高 IIC | 容量扫描、rank/sparsity/参数量记录、mapping-only probe、随机模型对比 | 所有机制定位与反空洞性结论 | 可在预注册范围内改变容量，但不能无界 | Stage 1 计划：`test_alignment_capacity_contract`、capacity frontier | `FROZEN_PENDING_STAGE1_TEST` |
| A07 | 测试 intervention pairs 与训练 pairs 严格分离 | 数据/实验 | 防止记忆 base/source 对或用测试 pair 调参 | 按 base window 分组切分；检查 pair IDs 无交集；normalizer 只用 train pairs | held-out 交换泛化与所有 IIC 主结果 | 否 | Stage 1 计划：`test_intervention_pair_split_no_overlap` | `FROZEN_PENDING_STAGE1_TEST` |
| A08 | 随机模型和随机概念必须作为负对照失败 | 实验 | 若二者也高分，映射可能在注入/记忆信息 | 同容量、同 pair、同调参预算比较随机初始化模型、随机/打乱概念、错误 SCM | 因果抽象、定位与反信息注入主张 | 否 | Stage 1 计划：`test_random_controls_fail` | `FROZEN_PENDING_STAGE1_TEST` |
| A09 | 输出概率距离可以稳定估计 | 统计/实验 | 多步概率交换误差依赖可重复且有限的分布距离 | 在已知分布/重复采样上检查 finite、对称性（适用时）、方差、样本数敏感性和零效应 normalizer | N1、IIC、worst-regime error | 可更换预注册距离，不可看测试结果后选择 | Stage 1 计划：`test_distribution_distance_stability` | `FROZEN_PENDING_STAGE1_TEST` |
| A10 | 状态定义不能读取测试未来 | 数据/实验 | 否则 unseen-regime 与 zero-refit 结论由未来标签构造 | 强制 `fit(train)`、`transform(train/validation/test)`；审计状态特征时间戳 | RQ3、N5、金融压力测试 | 否 | Stage 1 计划：`test_regime_definition_is_point_in_time` | `FROZEN_PENDING_STAGE1_TEST` |
| A11 | Wasserstein cost 的所有尺度只用训练集拟合 | 算法/统计 | 测试尺度会泄漏并改变 ambiguity geometry | 保存 train scaler；在 validation/test 只 transform；检查 cost 对角为零、非负及预注册对称性 | DiRoCA-style/TARCA robust 比较和 worst-regime 结论 | 否 | Stage 1 计划：`test_wasserstein_cost_train_only` | `FROZEN_PENDING_STAGE1_TEST` |
| A12 | 研究结论不得外推为真实市场因果识别 | 理论/叙事 | 模型内部可操纵性不提供现实数据生成过程的识别假设 | 文稿自动/人工查禁语；所有结论绑定“模型内部计算” | 科学有效性与证据边界 | 否 | Stage 0 文档合同；Stage 1 计划：paper-claim audit | `FROZEN` |
| A13 | forecast horizon 与 causal lag 是可独立定义和识别的索引 | 理论/数据 | 这是 TARCA 避免退化为 PLOT generic timestep 的核心限定 | 合成数据正交改变 horizon/lag；检查单索引模型不能同等恢复真值 | N2 与 forecast-indexed abstraction | 可合并，但合并后必须撤销 N2 | Stage 1 计划：`test_horizon_lag_identifiability` | `FROZEN_PENDING_STAGE1_TEST` |
| A14 | 显式变量/通道轴对机制定位有独立信息 | 算法/实验 | 否则 N3 只是 layer/time/subspace 的 PLOT 变体 | 植入跨变量传播与 variable truth；比较去掉 variable axis 的消融 | N3 的实质差异与位置恢复 | 可放宽，但需将 N3 降级为非新颖基线 | Stage 1 计划：`test_variable_axis_recovery` | `FROZEN_PENDING_STAGE1_TEST` |
| A15 | unseen sequential regime 上解释器、位置、normalizer 和映射保持 zero-refit | 算法/实验 | 每个状态重训不能证明共享机制稳健 | 冻结所有解释参数/hash；测试期不更新；与 per-regime oracle 分开标注 | N5、RQ3 | 否；per-regime fit 只能作为上界 | Stage 1 计划：`test_zero_refit_unseen_regime` | `FROZEN_PENDING_STAGE1_TEST` |
| A16 | 合成基准同时包含干预真值和四轴位置真值 | 基准/实验 | 单一真值无法同时验证 effect fidelity 与 localization | 验证 SCM oracle、teacher implantation、regime/lag/cross-variable propagation 元数据一致 | N7、Gate A、Gate 1 | 否，若缺一种真值则撤销联合 benchmark 声明 | Stage 1 计划：`test_synthetic_truth_contract` | `FROZEN_PENDING_STAGE1_TEST` |
| A17 | 高层 SCM 的错设可被负对照和失败区域暴露 | 理论/实验 | 错误高层模型也高分说明抽象评价缺乏区分力 | wrong SCM、缺失中间变量、Good-Apples-style pair buckets；报告所有失败区域 | RQ1 与“忠实高层解释” | 部分；可修订 SCM，但必须在新测试集上重验 | Stage 1 计划：`test_wrong_scm_control` | `FROZEN_PENDING_STAGE1_TEST` |
| A18 | Stage 0 仅允许 Stage 1+ 的空边界，不含实现占位代码 | 工程/范围 | 防止用未来代码填充目录并误报完成 | 检查 `configs/`、`data/*/`、`experiments/` 只有 README/空目录；扫描禁止算法模块 | Stage 0 范围合规性 | 否 | Stage 0 scope/manual checks | `FROZEN` |

## 解释规则

- `FROZEN`：Stage 0 已冻结的不可越界约束。
- `FROZEN_PENDING_STAGE1_TEST`：测试名称和预期行为已冻结，但实现与运行明确属于 Stage 1；本阶段没有创建这些测试。
- 任一 A01、A02、A05、A07、A08、A10、A12 或 A15 失败，都足以停止对应的正式机制/鲁棒/金融结论。
