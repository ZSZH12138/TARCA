# TARCA E02 完成情况与任务交接快照

> 快照日期：2026-09-01
> 实验身份：`e02_predictor_validity_v1`
> 最终状态：`COMPLETED / PASS`
> 权威范围：E02 正式结论、身份绑定、证据回收边界和 Stage 3 交接；研究契约与 Gate 定义仍以计划书、实施计划和预注册文件为准。

## 1. 功能层结论

E02 检验的不是 TARCA 的机制解释是否已经成立，而是后续所有机制实验将要使用的基础预测器是否可靠：
在冻结的 Stage 1B 合成世界、固定数据切分、固定模型身份和未改写的正式 Gate 下，概率预测器是否能在
保留测试轨迹上稳定优于简单基线，并给出数值有效、校准可接受的概率输出。

最终结果为 `PASS`。这意味着冻结的 primary iTransformer 已满足进入 Stage 3 / E03 的预测器前提，
可以作为后续机制植入和固定位置干预的预测骨干；它**不**证明 TARCA 的因果解释、自动定位、OT、DAS、DRO
或真实金融有效性已经成立。

## 2. 正式判定与核心结果

| 项目 | 正式结果 | 判定含义 |
|---|---:|---|
| h1–6 primary CRPS skill | `0.4728435253`（47.2844%） | 高于最低 `0.02` 要求 |
| 90% bootstrap CI（5,000 次） | `[0.4704311978, 0.4751662237]` | 下界大于 0 |
| 5 个 primary seeds | `5/5` 为正 | 高于至少 `3/5` 的一致性要求 |
| 3 个正初始化 | `3` | 满足至少 `2/3` 要求 |
| 相对 NLL / MAE | `-0.200356` / `-0.134500` | 分别改善约 20.04% / 13.45% |
| calibration coverage error | `0.008583`（0.8583 个百分点） | 低于 `0.05` 护栏 |
| seen / unseen coverage error | `0.006603` / `0.010563` | 均低于 `0.10` 分状态护栏 |
| h7–12 / h13–24 skill | `0.112605` / `0.020821` | 均高于 `-0.10` 次要时距下限 |
| 基线与数值 Gate | 优于 last-value、seasonal-naive；概率有限、scale 为正、quantile 不交叉 | 全部通过 |

主评分中，primary iTransformer 的平均 CRPS 为 `0.4184898429`，固定 strongest-linear（VAR）为
`0.4844567172`；该差异在预注册的 primary h1–6 比较口径中形成上述正 skill。E02 的正式决策不是以
单一分数替代 Gate，而是以下所有条件同时成立：主时距 skill、置信区间、跨 seed/初始化一致性、基线胜出、
seen/unseen 稳定性、概率质量和次要时距护栏均通过。

## 3. 执行与完整性

| 完整性项目 | 结果 |
|---|---:|
| 正式轨迹 | `120/120 COMPLETED` |
| 失败轨迹 | `0` |
| 完整性违规 | `0` |
| 执行账本 attempt | `14/14 COMPLETED` |
| Gate | `24/24 PASSED`，无 failed gate |
| 正式 final receipt | 已验证、可重算 |

因此，本快照中的 `PASS` 是冻结科学配置下的形式化 Gate 判定，不是展示性或营销性标签。

## 4. 身份与证据链

| 对象 | 固定身份 / SHA-256 |
|---|---|
| Stage 2 freeze receipt | `37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166` |
| E02 scientific config | `9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c` |
| E02 final receipt | `69ff94fab71059361b0eb9feb5a4c44c500bc04691ece91a12cb083dcf91704f` |
| 完整 Stage 2 archive | `7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a` |
| 最近确定性 server bundle | `0aa7086261322fe57642649a5abfe65be394f362ab4e12704b76963642e50e29` |

受控只读复核已经验证远端的 `e02_evidence.json`、`e02_decision.json`、`e02_receipt.json` 与执行账本所述
结论相互一致，并重算了最终 decision。该复核并不等于完整服务器证据已落地本机：数据库、日志、checkpoint、
完整 E02 结果归档及其文件哈希仍必须在服务器重置或关闭前回收、校验和登记。

## 5. 已解除与仍保留的边界

**已解除：** Stage 2 基础预测器有效性的阻塞。Stage 3 可以消费已冻结的模型、checkpoint、split、
概率输出接口及本快照所绑定的身份。

**仍保留：**

1. E02 仅在冻结合成世界上验证预测有效性，不能外推为真实金融结论或跨域结论；
2. E02 不验证机制真值恢复、内部干预、因果隔离、层×时间×变量定位、OT 或 DRO；
3. 不得因 E02 通过跳过 Stage 3 / E03，或直接进入 Stage 4、DAS、OT、DRO；
4. 不得替换 E02 已冻结的世界、切分、模型、seed、checkpoint、metric 或 Gate，并将新运行称为本次正式结果。

## 6. 后续任务交接

1. **优先回收服务器完整证据。** 下载完整 E02 归档与运行账本，逐项核验 SHA-256，并把本快照的“已远端只读验证”升级为“本地完整归档已验证”。在此之前不得因资源回收而丢弃服务器上的正式结果。
2. **固定 E02 结果指针。** 保留本快照、Stage 2 快照和历史恢复记录；任何补跑只能作为新身份的受控补充，不可覆盖本次 receipt。
3. **进入 Stage 3 / E03。** 以本快照绑定的 Stage 2 模型和 checkpoint 为输入，实施机制植入并由 E03 对已知 planted mechanism 做真值对照。
4. **遵守串行 Gate。** 只有 E03 通过其独立 Gate，才可讨论 Stage 4 的固定位置交换；其后的 OT、DAS、DRO 仍遵循原计划的依赖顺序。

## 7. 相关权威与实施文件

- `TARCA_项目计划书.md`：研究目标、科学边界和总 Gate 顺序；
- `TARCA_具体实施计划.md`：E02 定义、阈值与 Stage 3 入口；
- `TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`：冻结接口契约；
- `TARCA_STAGE2_HANDOFF_SNAPSHOT_2026-09-01.md`：上游 Stage 2 冻结身份；
- `TARCA_SERVER_ACCESS_RUNBOOK.md`：任何后续服务器接入、回收与清理的唯一操作规范。
