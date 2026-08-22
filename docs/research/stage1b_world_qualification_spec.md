# Stage1B 世界资格规范（审批草案）

> 状态：`DRAFT_AWAITING_USER_APPROVAL`
> 核验日期：2026-08-22
> 工作分支：`codex/stage1b-world-qualification`
> 协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 边界：本文件不是冻结研究契约、Gate decision 或正式实验结果。

## 1. 功能目标

Stage1B 需要提供一组“答案已知、适合预测、可以干预、能继续用于 Stage3–9”的时序世界。世界不能仅因为神经模型可能胜过 VAR 就入选；它必须先支持 TARCA-C1～C4 的证伪链，再讨论预测学习空间。

本轮只完成来源登记、资格规则、候选审核、组合设计和冻结草案。明确不执行：

- E01；
- E02；
- 预测器训练或调参；
- 正式数据下载或生成；
- 正式 SCM truth、split、seed 或测试结果发布；
- 对冻结 Stage0/Stage1A 文件的修改。

## 2. 权威约束

本规范服从以下文件：

1. `docs/auth/TARCA_项目计划书.md`；
2. `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`；
3. `docs/auth/TARCA_具体实施计划.md`；
4. `docs/preregistration_v0.md`；
5. `docs/assumption_ledger.md`；
6. `docs/novelty_claims.md`；
7. `docs/terminology.md`；
8. `docs/auth/TARCA_STAGE1A_HANDOFF_SNAPSHOT_2026-08-22.md`。

若本草案与上述文件冲突，以上述文件为准。

## 3. 世界生命周期

```text
DISCOVERED
→ REGISTERED
→ ROLE_ASSIGNED
→ PROJECT_RELEVANCE_REVIEWED
→ CONDITIONAL_ELIGIBLE / AUXILIARY / REFERENCE_ONLY / REJECTED
→ USER_APPROVED
→ FROZEN_vN
→ 后续另行授权实施
```

功能解释：

1. `DISCOVERED`：发现论文、数据集或生成器；
2. `REGISTERED`：记录版本、commit、许可证和答案能力；
3. `ROLE_ASSIGNED`：分配控制、主机制、oracle、压力测试或现实验证职责；
4. `PROJECT_RELEVANCE_REVIEWED`：对照 WQ-01～WQ-12；
5. `CONDITIONAL_ELIGIBLE`：方向合适，但仍需要后续 adapter/API 验证；
6. `AUXILIARY`：只能承担部分职责；
7. `REFERENCE_ONLY`：许可证或能力不足，只能引用；
8. `REJECTED`：不进入当前 Stage1B；
9. `USER_APPROVED`：用户明确批准来源、职责和薄适配边界；
10. `FROZEN_vN`：版本、参数、split、seed、hash 和测试保留规则全部冻结。

在用户批准之前，不得把 `DRAFT` 或 `CONDITIONAL_ELIGIBLE` 写成已入选世界。

## 4. 项目有效性硬门槛

### WQ-01：来源、版本和许可证

系统必须记录：

- 论文或正式说明；
- 官方仓库；
- default branch 和精确 commit；
- 代码许可证；
- 数据许可证；
- 数据 DOI、版本和 checksum（若存在）；
- Python/运行时兼容性；
- 是否允许复制、生成、修改或只能引用。

失败动作：许可证未知时标为 `REFERENCE_ONLY`；不得复制代码或下载为正式数据依赖。

### WQ-02：外部核心而非 TARCA 自创动力学

主世界的核心方程、图族或物理模拟器必须来自已登记的外部来源。TARCA 只能增加明确、可审计的薄适配层，不能为了结果重新发明大部分动力学。

允许：

- 固定外部模型及参数；
- 把外部输出转换为 Stage1A 契约；
- 保存并重放外部噪声；
- 为外部模型的现有参数建立概念接口；
- 在批准后增加显式 delay buffer；
- 在批准后按预声明参数组切换 regime；
- 记录 provenance、truth 和 checksum。

禁止：

- 根据 TARCA 测试成绩搜索或改写方程；
- 为让神经模型获胜而调整噪声、SNR、维度或 horizon；
- 测试后重新采样图或删除失败系统；
- 隐藏 clipping、zero replacement 或发散样本；
- 用学习型生成器替代需要检验的真实机制。

### WQ-03：概念具有结构语义

每个主世界至少要支持趋势/持续性、尺度/波动、传播中的两个；主世界组合必须覆盖三者，并为冲击、外生影响和状态切换保留明确接口。

每个概念必须记录：

```text
概念名称
外部方程中的状态/参数/输入
只使用当前或历史信息的观测定义
高层干预如何改变它
理论上影响均值、尺度、分位数或传播路径中的哪一项
哪些变量必须保持不变
与 Stage10/11 哪个真实概念对应
```

失败动作：只能由未来目标或事后结果定义的概念不得进入主证据。

### WQ-04：Level-3 paired counterfactual

主证据必须满足：

- factual/counterfactual 起点相同；
- 未来外生噪声相同；
- 无干预时逐位一致；
- 差异只来自批准的概念干预；
- future noise 作为 truth artifact 保存，但不进入普通 `WindowBatch`。

仅有独立噪声的 Level-2 interventional twin 可以作为辅助世界，不能单独关闭 TARCA 的 paired oracle 要求。

### WQ-05：forecast horizon 与 causal lag 可正交变化

主世界必须具有显式 lag 真值，并允许：

- 固定 lag 改变 horizon；
- 固定 horizon 改变 lag；
- true-lag/wrong-lag 对照；
- 将 lag 映射回原始时间步；
- 预先指定 lag，而不是从结果峰值反推真值。

失败动作：不能正交变化时不得支持 TARCA-C1/C2。

### WQ-06：变量图和传播路径可审计

主世界必须提供：

- 多变量命名；
- source-target 关系；
- graph/weight/Jacobian 或等价传播真值；
- 单变量、变量对和小型变量组；
- 干预后跨变量效应路径。

单变量或低维无图世界只能作为预测/数值压力测试。

### WQ-07：可组成 base/source intervention pairs

主世界必须能产生足够的共同支持：

- target concept 有明显差异；
- non-target concepts 尽量相似；
- 历史协变量相似；
- primary pair 为 same-regime；
- source/base 来自不同时间段；
- pair partition 不共享窗口；
- source 复用率可限制。

失败动作：所有概念总是共同变化、无法隔离时不得用于 Cause/Isolation 主证据。

### WQ-08：regime 是可解释机制变化

每个 regime 必须对应明确的参数或结构变化，例如：

- 漂移/增长/均值回复；
- diffusion/noise tail；
- coupling/propagation；
- causal lag；
- shock process；
- 外部驱动。

状态描述必须能由训练期或历史信息构造。测试期不能读取未来目标或真实状态标签来更新 predictor、site、map、normalizer、threshold 或解释器。

失败动作：任意标签或未来可见状态不能支持 TARCA-C3。

### WQ-09：概率效应丰富且可分离

世界组合必须同时覆盖：

- 主要改变均值的干预；
- 主要改变尺度的干预；
- 改变分位数/尾部的干预；
- 延迟后出现的效应；
- 跨变量传播效应。

这样 Stage5 的 `Δmean/Δscale/Δquantile` 高层效应签名才有实际内容。

### WQ-10：完整负对照

主世界必须允许：

- source=base；
- wrong lag；
- wrong source；
- random concept；
- shuffled concept labels；
- random variable/edge/site/subspace；
- random environment；
- wrong high-level SCM；
- 支持集外干预诊断。

失败动作：无法构造负对照时，不得支持 Gate A/1/2。

### WQ-11：连接 Stage10/11

Stage1B 不需要模拟真实天气或金融市场，但其概念操作必须能连接后续领域：

| Stage1B 机制 | Weather/Electricity/Traffic | 金融压力测试 |
|---|---|---|
| drift/growth/persistence | 局部趋势、负荷水平、拥堵持续性 | 动量、反转、均值回复 |
| diffusion/scale | 传感器或负荷波动状态 | 实现波动率、尾部风险 |
| coupling/propagation | 温湿度关系、传感器传播、共享冲击 | 市场/行业溢出、相关结构变化 |
| sparse/exogenous shock | 日历、事故、天气或公共事件 | 宏观、流动性、订单流冲击 |
| regime parameter change | 时间阶段、变点前后、波动分组 | 高低流动性、波动状态、市场阶段 |

没有任何下游对应关系的世界只能作为额外压力测试。

### WQ-12：预测学习空间与防挑数据

只有 WQ-01～WQ-11 先通过，才允许考察神经学习空间。

当前审批前阶段只允许使用：

- 外部论文的已发表预测结果；
- 外部官方 benchmark；
- 方程的结构性非线性、状态依赖和长时传播分析。

当前禁止运行 TARCA E02 或使用 TARCA sealed test 选择世界。后续如获授权：

- candidate pool、系统族、指标和预算先冻结；
- TRAIN/VALIDATION 只用于模型选择；
- TEST 保留整个系统、参数状态或随机种子，而不是只保留窗口；
- 失败系统和 seed 不得删除；
- E02 失败优先诊断 predictor/训练或 DGP headroom，不能静默换世界。

## 5. 世界职责

### CONTROL_LINEAR

功能：验证 VAR 公平性。允许 VAR 获胜或持平，不用于判定神经路线失败。

### PRIMARY_MECHANISTIC

功能：支撑 Stage2 预测、Stage3 机制植入、Stage4/5 干预与指标、Stage6–8 四轴定位、Stage9 regime robustness。

准入：必须满足 WQ-01～WQ-11；WQ-12 只决定后续预测验证优先级。

### ORACLE_AUXILIARY

功能：提供独立的图、lag、regime、识别或 Level-3 counterfactual 答案。允许只覆盖部分 TARCA 概念。

### FORECAST_STRESS

功能：提供复杂非线性、混沌、长时依赖或数值难度。不能替代概念和干预真值。

### EXTERNAL_REALISM

功能：以后验证外部效度和失败区域。不能反向关闭合成 Gate。

### REFERENCE_ONLY

功能：作为论文/设计参考。不得复制未知许可证代码或数据。

## 6. 主世界最小组合

审批建议为至少三个不同科学系统族，而不是同一随机生成器的三个配置：

1. 非线性随机网络/耦合系统；
2. 非线性生态、生物或群体动力系统；
3. 具有显式 graph 和 lag 的模块化动力系统。

每个系统族必须单独报告，不得只给混合平均数。组合应同时提供：

- 一个 VAR 明确占优或持平的控制族；
- 两个以上具有外部非线性预测证据的主候选族；
- 一个独立 Level-3 oracle 来源；
- 一个纯外部或物理现实验证来源。

## 7. 冻结与修改

正常流程：

```text
DRAFT
→ 资格报告
→ 用户批准来源、职责和薄适配边界
→ 冻结 config/source commit/root seed/split/checksum
→ FROZEN v1
```

允许修改，但必须：

```text
提交理由和影响分析
→ 用户授权
→ 归档旧版本
→ 创建递增版本
→ 重新执行受影响的验证
→ 冻结新版本
```

禁止覆盖旧版本或观察 TEST 后静默修改。

## 8. 本草案的审批边界

当前可以批准：

- WQ-01～WQ-12 作为 Stage1B 世界准入规范；
- 外部核心 + 薄适配层的架构；
- 世界职责和候选优先级；
- 在正式实施前继续做只读 API/许可证核验。

当前不能批准为已经完成：

- 某个候选已经通过 E01/E02；
- 某个神经模型已经稳定胜过 VAR；
- candidate pool 已成为正式 `FROZEN v1`；
- Stage1B 已产生正式 SCM truth 或数据分区。
