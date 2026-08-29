# Stage1B 世界资格规范 v2

> 状态：`FROZEN_V2`
> 构建日期：2026-08-25
> 协议身份：`TARCA-E2E-STAGE-PROTOCOL-2.0`
> 完成日期：2026-08-29
> 边界：v2 首次完整资格作为 pilot 失败证据保留；双尺度短期独立确认已通过并冻结为唯一活动 v2；E01/E02 未执行。

```text
Implementation status: FROZEN_V2
Active scientific series: v2
Freeze default: immutable after qualification
Authorized override: exact prior-manifest hash + atomic replacement, still named v2
Executed: full Stage1B pilot, independent short-horizon confirmation, freeze, unit/integration/E2E/probes
Not executed: E01, E02
```

## 1. 功能目标

Stage1B v2 提供一组来自公开研究的“已知答案世界”，用于判断 TARCA 后续研究是否具备：

1. 可学习的非线性预测空间；
2. 可审计的图、符号、传播路径和 regime；
3. 事实/反事实共享噪声的 paired oracle；
4. 可供 Stage3–8 捕获和交换的神经内部位置；
5. 可连接天气、电力、交通和金融机制的功能语义。

世界不能因为“容易让神经网络赢”而入选。它必须先满足项目有效性和数值健康，再进入
独立资格比较。

## 2. 权威边界

本规范服从 `docs/auth` 内的计划书、实施书、协议书和变更控制文件。若冲突，以权威文件
为准。

本版本只允许使用：`QUAL_TRAIN`、`QUAL_TUNE`、`QUAL_SEEN`、`QUAL_UNSEEN`。
禁止读取或生成 E01/E02 的正式 seed、split、sealed test 和结果标识。

## 3. 活动世界

### 3.1 线性 VAR 控制

验证公平 VAR 基线、数据切分和结构负对照。VAR 可以获胜；该世界不承担神经优势门槛。

### 3.2 单尺度 Lorenz-96 F=10

20 维主世界。提供公开非线性方程、方向图、状态依赖符号、传播路径、强迫项、节点冲击
和公开噪声状态。它是 v2 的第一主资格候选。

### 3.3 单尺度 Lorenz-96 F=40

20 维混沌压力世界。用于检查更强非线性下的失败区域，不决定主套件是否通过。

### 3.4 双尺度 Lorenz-96

8 个观测慢变量、256 个潜在快变量的主世界。固定公开参数：
`K=8, J=32, h=1, b=10, c=10, F=20, dt=0.001`，每 0.2 时间单位观测。
它检验部分可观测、多时间尺度和潜在机制。

### 3.5 GVAR 捕食者—猎物

20 维辅助 oracle：10 个猎物、10 个捕食者、每个节点两个跨类父节点，固定
`alpha=1.1, beta=0.2, gamma=1.1, delta=0.2, sigma=0.1`。官方生成器明确把负值截为
0；v2 在配置中声明此边界行为，并在每条轨迹中报告次数，不静默隐藏。

### 3.6 修正 CML

辅助图传播和混沌世界。按公开公式计算：自项不除以度数，只对邻居和求平均。它修复了
历史版本的决定性方程语义错误。

## 4. 来源锁定

活动来源清单是 `third_party_manifest/stage1b_sources_v2.yaml`，详细证据在
`configs/stage1b/worlds_v2.yaml`。每个来源记录论文、官方仓库、精确 commit、证据文件
URL、SHA-256、许可证状态和代码使用方式。

服务器会把六个官方仓库按固定 commit 物化为只读来源，并先运行官方复现通道。世界数据
由各世界适配器调用相应的官方生成入口或经过逐步等价验证的官方方程入口生成；真值由生成
器在生成同一条轨迹时一并输出，不能由预测模型或神经网络反推。用户已明确授权本项目在
资格阶段直接使用这些固定来源；许可证状态仍完整记录在收据中，但不再作为本地执行阻断。

## 5. WQ 功能门槛

### WQ-01 来源可审计

论文、仓库 commit、证据 SHA 和许可证/引用限制完整。

### WQ-02 外部核心

核心方程和参数来自公开研究；TARCA 只提供数值积分与适配。

### WQ-03 概念有结构语义

概念能映射到方程状态、参数、尺度或传播，而不是由未来标签定义。

### WQ-04 paired counterfactual

事实与反事实共享初态和未来噪声；身份干预必须逐位一致。

### WQ-05 图与符号

提供 target-by-source 邻接、固定符号或明确的状态依赖符号。

### WQ-06 lag 与路径

提供预先登记的最短机制路径长度；不得从结果峰值反推真值。

### WQ-07 source/base pairs

能够改变授权概念，同时保持共同支持、方向和噪声一致。

### WQ-08 regime 可解释

seen/unseen 只改变已声明的一个机制参数；来源必须可追踪。

### WQ-09 负对照

支持 identity、wrong lag、wrong source、random site/variable/subspace 等控制。

### WQ-10 数值健康

轨迹必须有限、非塌缩、非伪二周期、可重放。公开声明的边界裁剪必须计数。

### WQ-11 下游有效性

至少映射到天气、电力、交通、传感网络、生态或金融机制中的一类。

### WQ-12 资格隔离

完整轨迹不能跨分区；归一化只使用 `QUAL_TRAIN`；lineage 必须含世界、来源、图、噪声、
seed、regime 和配置哈希。

### WQ-13 高重复神经优势

只对 `PRIMARY_MECHANISTIC` 执行。控制、辅助和压力世界在 WQ-01～12 通过后豁免。

## 6. 公平预测器

### VAR

使用与神经模型相同的数据、history、horizon 和评分。lag 与 ridge 只在 `QUAL_TUNE`
选择。

### PatchTSTReference

保留通道独立、重叠 patch、RevIN、3 层 encoder、flatten head，并输出输入相关概率尺度。
正式配置参考官方 weather 脚本：patch 16、stride 8、`d_model=128`、16 heads、
`d_ff=256`、最多 100 epochs、patience 20、学习率 `1e-4`。

### ITransformerReference

保留变量 token、逐窗口归一化/反归一化、3 层 encoder 和输入相关概率尺度。正式配置使用
`d_model=512`、8 heads、`d_ff=512`、最多 100 epochs、patience 20、学习率 `1e-4`。

两者必须冻结权重后仍能列出、捕获和交换批准的位置，且交换不能改写模型权重。

## 7. v2 资格判定

### 7.1 首次完整运行结果

首次 74/74 任务运行是探索性筛选：套件总门禁失败。双尺度世界的 iTransformer 总体胜率
为 98.6111%、CRPS skill 为 +4.067%；h1–6 为 72/72 胜、skill +13.2423%，h7–12
为 64/72 胜、skill +2.0937%，h13–24 为 19/72 胜、skill -1.0055%。原相对校准
护栏仍使该世界失败。单尺度世界未表现出跨时距优势。完整运行被保留，但不能直接作为确认
结论或活动冻结版本。

### 7.2 双尺度短期独立确认

依据 `docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0002.md`，本次确认只运行
`lorenz96_twoscale_v2 + ITransformerReference + tuned VAR`。h1–6 是主要门禁；h7–12
只报告次要趋势；h13–24 只报告已知能力边界。新种子为 `1649910005`、`2058661680`、
`723243092`，必须重新生成轨迹、训练和预测。校准护栏为名义 90% 区间平均绝对覆盖误差
不超过 0.05。其余既有比较单元、CRPS、种子、seen/unseen、NLL、MAE、最坏状态、尺度和
可操作性门槛保持不变。

每个主世界分别从未见资格轨迹形成完整轨迹比较单元。一个神经候选通过需要：

- 至少 40 个比较单元；
- CRPS 胜率至少 65%；
- 总体 CRPS skill 严格大于 0；
- trajectory-level paired bootstrap 上界大于 0，即证据不能支持“整体稳定劣于 VAR”；
- seen 与 unseen 胜率都严格大于 50%；
- NLL、MAE、校准误差和 worst-regime CRPS 不超过 5% 相对退化容差；
- 概率尺度有限且严格为正；
- 模型通过内部位置可操作性检查。

65% 是 v2 预注册的项目准入线，不宣称是学界统一阈值。单个 seed 或 horizon 允许失败，
失败单元必须保留。套件至少需要一个独立主世界家族通过；另一个主世界失败会被完整报告，
但不会自动否决已存在的有效主世界。

确认运行已经按新种子重新生成、重新训练和重新预测：h1–6 为 72/72 胜，CRPS skill
+13.5274%，seen/unseen 胜率均为 100%，平均绝对校准误差 0.01101、最大 0.03015，
paired bootstrap 95% CI 为 [0.05267, 0.05525]。世界与套件门禁均为 `PASS`。
h7–12 胜率 84.7222%、skill +1.8070%，仅作次要诊断；h13–24 胜率 34.7222%、skill
-1.0366%，作为预先声明的能力边界。

## 8. 构建、资格与冻结状态

```text
BUILT_NOT_QUALIFIED（历史）
→ 硬件探针通过
→ 首次完整 pilot 与独立确认
→ WQ-01～13 自动决策
→ 用户审阅
→ FROZEN_V2（当前）
```

当前独立确认已通过并冻结。唯一活动目录为 `artifacts/stage1b/frozen/v2/`，活动指针为
`artifacts/stage1b/active.json`。正常情况下完整资格收据和 manifest 不可修改。用户可以授权
修改或覆盖，但必须记录授权人、原因和当前 manifest 的精确 SHA-256，并在验证新旧内容后
原子替换；活动身份仍为 `v2`，不创建递增修订。失败或不完整运行不会被冻结，也不会登记成
历史活动版本。具体规则以 `docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0003.md` 为准。

冻结前会验证：官方来源、官方复现、服务器环境、精度策略、硬件探针、运行图、完整
`TaskManifest`、`ExecutionPlan` 和全部任务计数。全部比较行与失败行保存在修订内的完整
`qualification_receipt.json`，不得只保留通过摘要。资格 seed 与 E01/E02 保留 seed 严格
分离，任何来源漂移、科学身份漂移、部分完成或正式实验标识都会拒绝冻结。

## 9. 活动入口

- 世界配置：`configs/stage1b/worlds_v2.yaml`
- 资格配置：`configs/stage1b/qualification_v2.yaml`
- 当前冻结：`artifacts/stage1b/frozen/v2/`
- 确认运行的历史配置：`configs/stage1b/qualification_v2_confirmation_r2.yaml`
- 确认运行设计（历史命名已被 CCP-0003 取代）：`docs/superpowers/specs/2026-08-29-stage1b-v2-confirmation-design.md`
- 确认运行计划（历史命名已被 CCP-0003 取代）：`docs/superpowers/plans/2026-08-29-stage1b-v2-confirmation.md`
- 来源清单：`third_party_manifest/stage1b_sources_v2.yaml`
- 构建设计：`docs/superpowers/specs/2026-08-25-stage1b-v2-design.md`
- 实施计划：`docs/superpowers/plans/2026-08-25-stage1b-v2-implementation.md`
- 服务器运行入口：`scripts/run_stage1b_runtime.py`
- 只读监控前端：`frontend/stage1b-monitor/`
- 历史失败快照：`docs/research/stage1b_world_qualification_report_v1.md`
- 当前权威交接：`docs/auth/TARCA_STAGE1B_HANDOFF_SNAPSHOT_2026-08-29.md`
