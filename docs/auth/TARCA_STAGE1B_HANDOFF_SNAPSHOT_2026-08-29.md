# TARCA Stage1B 完成情况与任务交接快照

> 快照日期：2026-08-29
> 阶段状态：`FROZEN_V2`
> 活动科学身份：`v2`
> 正式 E01/E02：未运行
> 权威变更控制：`TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0003.md`

## 1. 功能层结论

Stage1B 已完成它在 TARCA 中应承担的工作：找到并冻结一个既能支持神经预测、又有生成器真值
和后续干预结构的“已知答案世界”。当前目标不是证明神经网络在所有世界、所有指标和所有
时距都胜过 VAR，而是为后续解释性研究固定一个已确认有效的组合：

```text
世界：lorenz96_twoscale_v2
预测器：ITransformerReference
比较基线：调优 VAR
主要解释时距：h1–6
活动版本：v2
```

这个组合适合后续研究，因为双尺度 Lorenz-96 同时提供观测慢变量、潜在快变量、已知方程、
机制图、regime、共享噪声的事实/反事实轨迹和可操作的神经内部位置。后续可以解释“模型为什么
能做好短期预测”，而不需要虚构它在中长期也同样有效。

## 2. 历史过程

### 2.1 v1 历史

v1 的具体运行、失败世界与根因集中保存在
`docs/research/stage1b_world_qualification_report_v1.md`。v1 不再是活动实现，也没有被冻成历史
活动版本；保留该报告是为了避免将失败经验抹去。

### 2.2 首次完整 v2 pilot

首次服务器 pilot 完成 74/74 个任务，流程执行本身无失败、来源证据通过、未发现来源或身份
漂移，但总科学门禁为 `FAIL`：

| 世界 | 结果 | 解释 |
|---|---:|---|
| `lorenz96_f10_v2` | 胜率 5.5556%，CRPS skill -1.3654% | 短期线性可预测性强，VAR 占优，不适合作为活动解释目标 |
| `lorenz96_twoscale_v2` | 胜率 98.6111%，CRPS skill +4.0675% | 预测能力强，但旧门禁要求校准还必须相对 VAR 更好，因此被整体判失败 |

双尺度 pilot 的分时距结果是：h1–6 为 72/72 胜、skill +13.2423%；h7–12 为 64/72 胜、
skill +2.0937%；h13–24 为 19/72 胜、skill -1.0055%。这说明“短期有效、中长期逐步失效”
是真实能力边界。根据 CCP-0002，项目在新种子上重新生成数据、重新训练和重新预测，未把
观察过的 pilot 结果直接当作确认性结论。

旧 pilot 原始压缩包 SHA-256 为
`193137ad0c6596472cbd0782576fb9adfe3691cf5f8ed6714c290f16797e419e`。其大型本地副本已在
本次交接记录历史事实后列入清理，不会上传 GitHub。

## 3. 最终 v2 资格结果

最终确认运行完成 33/33 个计划任务，失败任务为 0，123 个内部清单文件全部校验通过；官方
来源证据通过，`source_drift_detected=false`，`identity_drift_detected=false`。

### 3.1 主要门禁 h1–6

| 指标 | 结果 |
|---|---:|
| 完整轨迹比较单元 | 72 |
| 神经模型胜出 | 72/72 |
| 胜率 | 100% |
| 神经平均 CRPS | 0.3448407484 |
| VAR 平均 CRPS | 0.3987860038 |
| CRPS skill | +13.52736928% |
| Seen 胜率 | 100% |
| Unseen 胜率 | 100% |
| 平均绝对校准误差 | 0.01100558 |
| 最大校准误差 | 0.03014706（门槛 0.05） |
| paired bootstrap 95% CI | [0.0526677872, 0.0552531778] |

三个独立确认种子 `1649910005`、`2058661680`、`723243092` 均产生正向结果。主要门禁和
世界门禁均为 `PASS`，没有失败检查。

### 3.2 次要时距与能力边界

- h7–12：61/72 胜，胜率 84.7222%，CRPS skill +1.8070%；
- h13–24：25/72 胜，胜率 34.7222%，CRPS skill -1.0366%。

因此，后续解释对象严格限定为 h1–6。h7–12 可以作为泛化诊断；h13–24 用于展示能力边界和
负面对照，不能改变主要资格结论，也不能被写成神经模型中长期胜过 VAR。

## 4. 冻结身份与证据

唯一活动身份为 `v2`，不区分活动 r1/r2：

| 证据 | SHA-256 / 状态 |
|---|---|
| 活动 manifest | `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25` |
| 资格收据 | `d17a6523dbe4e1e82d6ed36c2a27982f0b54d64146d0cdcbcaa61b6407d2aef0` |
| 最终服务器压缩包 | `0e6107cb77138de4363b90a22ddc563fd45809db89c5089a807f9f738381be23` |
| 最终压缩包大小 | 339,009,390 bytes |
| 官方来源胶囊 | `cb508aaaa7fa8aaccc380b09bdbf167c4e04b9b38ac0d01abbcf4862c53880ea` |
| 来源 manifest | `9b233b370832f1cf1dffc9bb6c2fc034cc3e1f0732ddf0431219d2cb7849a79c` |

原始资格收据中的 `qualification_id` 保留了服务器确认批次名称，只是不可篡改的来源字段，
不构成第二个活动版本。当前目录和活动指针都只有 `frozen/v2`。

## 5. 已完成的工程能力

Stage1B 当前实现已经提供：

1. 六个官方来源的固定 commit、关键文件和完整树哈希核验；
2. 离线来源胶囊导入，服务器不依赖运行时 GitHub 状态；
3. 生成器在生成轨迹时同步输出图、机制、regime 与 paired counterfactual 真值；
4. VAR、PatchTST、iTransformer 的公平数据边界和概率预测比较；
5. 不可变任务图、资源感知调度、双 GPU/CPU 并行和资源释放后补位；
6. SQLite 状态、原子模型 checkpoint、已验证制品复用和手动 `resume`；
7. 只读中文监控前端，显示进程、期望/实际 CPU 核数与占用、内存、GPU/显存、遥测新鲜度、
   checkpoint 和有证据的 ETA；
8. 来源、环境、精度、硬件、任务清单、执行计划、结果与冻结 manifest 的哈希绑定；
9. 单一 v2 冻结：默认不可覆盖，只有绑定旧 manifest 哈希的用户明确授权才能原子替换。

## 6. 本地与 GitHub 制品边界

GitHub 只保存源代码、测试、配置、文档和以下小型冻结证据：

```text
artifacts/stage1b/active.json
artifacts/stage1b/frozen/v2/qualification_receipt.json
artifacts/stage1b/frozen/v2/manifest.json
artifacts/stage1b/frozen/v2/manifest.sha256
```

以下证据只保存在本地，不上传 GitHub：

```text
artifacts/stage1b/server-archives/tarca-stage1b-confirmation-r2-fd27aa4.tar.gz
artifacts/stage1b/server-archives/tarca-stage1b-confirmation-r2-fd27aa4.tar.gz.sha256
artifacts/stage1b/source-capsules/stage1b-v2-official-sources.tar.gz
artifacts/stage1b/source-capsules/stage1b-v2-official-sources.tar.gz.receipt.json
third_party/stage1b/
```

本次白名单清理的精确文件数与字节数将在清理完成后回填到本节，清理不会触碰 Stage0、
Stage1A、最终压缩包、来源胶囊、官方来源缓存或用户已有的未跟踪 Stage0 快照。

## 7. 已知限制与风险

- 当前资格只证明双尺度世界上 iTransformer 的 h1–6 概率预测能力，不证明所有模型、世界、
  时距或指标上的一般优势；
- 生成器真值是合成世界内部的结构真值，不自动等同于真实世界因果；
- h13–24 已实测不具备相同优势，后续不得扩大声明；
- 资格阶段的模型择优是进入解释研究的工具选择，不等于对同一正式测试集无限试错；后续 E02
  必须绑定当前冻结模型、位置、数据和 seed 边界；
- FastAPI/Starlette 测试客户端未来需要迁移到 `httpx2` 参数方式；现有行为仍通过测试；
- PyTorch 的旧 `torch.cuda.amp.GradScaler` 入口存在弃用提示，后续可迁移到新 API，但不得在
  未重验精度身份时直接更改正式训练行为；
- 正式服务器仍以 `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04` 和服务器锁文件为准；
  本地 Python 3.11 环境只用于代码、文档和冻结验证。

## 8. 中断、恢复与核验

服务器 24 小时断电前应停止新任务并保留运行目录。恢复时不得重新 `launch`，应使用
runbook 的 `resume`：它会核对 SQLite、制品和 checkpoint，只重做缺失或未完成工作。Codex
点火并确认独立进程、心跳和只读前端稳定后必须返回，不得代替用户长期盯守。

本地核验唯一活动冻结：

```powershell
python scripts/check_stage1b.py --artifact-root artifacts/stage1b --json
```

预期返回 `status=PASS`、`active_series=v2` 和上述 manifest SHA-256。

## 9. 下一任务的正确起点

1. 读取本快照、CCP-0003 和冻结 manifest，确认活动身份仍为 v2；
2. 把后续实验限定到 `lorenz96_twoscale_v2 + ITransformerReference + h1–6`；
3. 在运行任何正式 E01/E02 前，单独冻结模型 checkpoint、数据 split、normalizer、内部位置、
   干预集合、映射方法、seed 和统计门禁；
4. 先做最小可逆的内部位置捕获/交换 smoke test，再进入解释映射与负对照；
5. E01/E02 只有在用户另行授权后才能运行；本次收尾没有提前使用任何正式保留结果。

这份快照是 Stage1B 到下一任务的权威交接入口；历史构建报告只用于解释当时状态，不能覆盖
本快照记录的 `FROZEN_V2` 事实。
