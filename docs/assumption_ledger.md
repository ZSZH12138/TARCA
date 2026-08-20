# TARCA Stage 0 假设台账

> 版本：1.0.0
> 原则：假设不是事实。每项必须有可观察代理、验证阶段和失败动作；不能通过改写术语绕过 Gate。

| assumption_id | 支撑 claim | 待验证假设 | 可观察代理/检查 | 验证 Stage | 失败动作 | 影响 Gate |
|---|---|---|---|---|---|---|
| A-SCOPE-001 | 全部 | 模型内部干预只能支持模型计算因果，不自动支持真实世界因果 | 术语、论文声明和图表逐项审计 | Stage 0 / 13 | 删除越界因果表述 | Gate 0 / 4 |
| A-ID-001 | C1–C4 | 每个因果性表述都有明确识别策略和可列举假设 | identification disclosure 清单 | Stage 0 / 5 / 13 | 降级为关联或验证性表述 | Gate A / 4 |
| A-CONCEPT-001 | C1–C4 | 高层概念可在不使用未来目标的情况下定义 | 概念标签审计、future-access 测试 | Stage 1B / 4 | 重构或删除概念 | Gate A |
| A-PAIR-001 | C1–C4 | base/source pair 具有足够重叠且不会跨 partition 泄漏 | pair overlap、距离、partition separation | Stage 1A / 4 | 收紧 pairing；无法修复则删除 claim | Gate A |
| A-EFFECT-001 | C1–C4 | 高层与低层效应可在同一输出语义和固定 normalizer 下比较 | shape/horizon/schema 检查；train-only normalizer | Stage 4 / 5 | 重定义 effect record 或指标 | Gate A |
| A-TIME-001 | C1/C2 | forecast horizon 与 causal lag 能被实验正交改变和识别 | horizon × lag factorial design；true/wrong lag | Stage 1B / 4 / 5 | `DROP_CLAIM` C1；C2 降级 | Gate A / 1 |
| A-TIME-002 | C1/C2 | patch/token 到原始时间范围的映射准确且可审计 | patch coverage round-trip | Stage 2 / 3 | 修复 adapter；禁止定位 | Gate A / 1 |
| A-PRED-001 | C1–C3 | 至少一个小型神经 predictor 在主证据数据上优于 naive/linear baseline | validation NLL/CRPS/MAE，多 seed | Stage 2 | 修复模型/SCM；失败则停止机制路线 | Stage 2 Exit |
| A-PROB-001 | C1/C3 | 概率头数值稳定且能生成有效分布 | scale > 0、finite log_prob、coverage/CRPS | Stage 2 | 修复概率头 | Stage 2 Exit |
| A-PLANT-001 | C2 | 植入机制真实控制预测且联合位置真值唯一可审计 | oracle ablation/intervention 与 manifest | Stage 3 | 重构植入模型 | Gate A / 1 |
| A-LOC-001 | C2 | variable × lag × horizon × subspace 联合真值可恢复而非仅恢复边际轴 | joint recovery、IoU/F1、subspace distance | Stage 6–8 | 删除四轴主张；保留边际结果 | Gate 1 |
| A-CAP-001 | C2/C4 | 容量受限映射不能靠参数量记忆标签或 pair | rank/parameter frontier、mapper-only probe | Stage 5 / 8 | 收紧容量；两轮失败则停止 | Gate A / 1 |
| A-NEG-001 | C1–C4 | 随机模型、概念、site、wrong lag 和 source=base 不会接近真实机制 | 全套负对照与置信区间 | Stage 4–8 | Gate 不通过；重构概念/干预 | Gate A / 1 |
| A-UNMAPPED-001 | C4 | 未映射变量不会被静默忽略并制造虚假 faithfulness | CAE-style unmapped-variable test | Stage 5 | 扩展高层模型或删除抽象 | Gate A |
| A-SUPPORT-001 | C4 | 低层干预表示留在可解释的支持范围，或偏离可被显式诊断 | representation distance、null/hidden-path probe | Stage 4 / 5 | 限制干预或将结果标为 OOD 失败 | Gate A |
| A-REGIME-001 | C3 | 环境划分对应可解释的机制变化而非任意标签 | SCM truth、train-only regime descriptors、随机划分对照 | Stage 1B / 9 | 重定义环境；无法识别则删除 robust claim | Gate 2 |
| A-ZERO-001 | C3 | unseen regime 评估期间 predictor、site、map、normalizer、threshold 全部不更新 | artifact hash before/after；fit-call audit | Stage 9–11 | 结果改标 TTA；不能关闭 Gate 2 | Gate 2 |
| A-DRO-001 | C3 | 最坏状态改进不是平均性能崩溃或测试标签泄漏造成 | average/worst frontier、label-access audit | Stage 9 | 收缩 DRO 目标或删除 claim | Gate 2 |
| A-DATA-001 | 全部 | 标准化、PCA、状态模型和概念阈值只在训练期拟合 | fit-time ranges、sealed test、hash audit | Stage 1A / 10 / 11 | 修复并废弃受污染结果 | Gate A–4 |
| A-STATS-001 | 全部 | intervention pairs 与时间点的依赖结构被统计方法正确处理 | grouped/block bootstrap；pair grouping | Stage 12 | 更换估计量并重算，禁止伪独立检验 | Gate 4 |
| A-EXT-001 | 全部 | 结论不依赖单一数据集、模型、概念或 seed | 多域、多模型、5 seeds、失败区域 | Stage 10–12 | 降级外部效度和通用性声明 | Gate 3 / 4 |
| A-RESOURCE-001 | 全部 | 更换本地/服务器 backend 不改变科学身份 | config/data/model/seed hash 一致 | 所有 Stage | 废弃不一致运行并重新执行 | 对应 Stage Gate |
| A-THIRDPARTY-001 | 全部 | 第三方来源身份、commit 和许可证状态可复核 | sources.yaml、git ls-remote、license 文件 | Stage 0 / 每次升级 | 未知许可证仅可 reference/static，禁止复制 | Gate 0 / 4 |

## 失败处理优先级

1. 发现泄漏、信息注入或 hash 不一致：立即废弃受影响结果；
2. 发现核心可识别性失败：删除或收窄 claim，不用新增数据集掩盖；
3. 发现工程适配失败：只修复 adapter/contract，不改变科学 Gate；
4. 发现资源不足：报告最低和建议资源，选择服务器 backend；不静默缩小冻结任务。
