# Stage1B 世界资格实施报告 v1

> 状态：`QUALIFIED_FAIL_UNFROZEN`
> 日期：2026-08-22
> 实施分支：`codex/stage1b-world-implementation-v1`
> 可回退文档分支：`codex/stage1b-world-qualification`（`b4616d2`）
> 正式实验边界：未运行 E01/E02；未访问正式分区或正式种子。

## 1. 功能层结论

Stage1B v1 已按批准流程完整执行，但没有通过冻结门槛。

- 外部世界可以重复生成，并能保存同一未来噪声。
- 图、路径 lag、机制、配对轨迹和负对照合同均已实现。
- VAR、PatchTST 和 iTransformer 接受相同训练/调优数据。
- 18 次神经训练全部完成，全部通过内部位置捕获和 source/base 交换测试。
- 线性控制世界中 VAR 按预期获胜。
- 两个主科学世界中，没有一个神经模型能在三个资格种子上稳定超过 VAR。
- 自动套件 Gate 为 `FAIL`；冻结器拒绝创建 v1；`active.json` 不存在。

因此，v1 是一份完整的失败证据，不是可用于后续正式研究的冻结世界。

## 2. 执行边界与可回退性

| 项目 | 实际状态 |
|---|---|
| 实施分支 | `codex/stage1b-world-implementation-v1` |
| 未实施回退点 | `codex/stage1b-world-qualification` / `b4616d2` |
| E01 | 未运行 |
| E02 | 未运行 |
| 正式实验标识 | 收据中为空 |
| 资格分区 | `QUAL_TRAIN / QUAL_TUNE / QUAL_SEEN / QUAL_UNSEEN` |
| 资格种子 | `104729 / 130363 / 155921` |
| 正式保留种子 | 只在配置中声明隔离，未用于生成、训练或评分 |
| v1 冻结 | 未创建 |
| 活动指针 | 不存在 |

冻结命令对实际失败收据返回：`FreezeRejected: suite gate did not pass`，退出码为 1。

## 3. 外部来源与重放

外部动力学使用 Interfere：

- 仓库：`https://github.com/djpasseyjr/interfere.git`
- 固定提交：`adfa3f730019f17c3554dd7e0c181248f785bb8b`
- 包版本：`1.0.2`
- 许可证：MIT
- 许可证 SHA-256：`44af4f82e0f356f4a0e48887e48ab0fbb4e586f83d7618b35b496a14f7ffd8f0`

项目检出位于忽略目录 `data/third_party/interfere`，提交号和许可证哈希均在运行前重新核对。生态 SDE 使用显式 Wiener increments；确定性 CML 使用显式零噪声记录；配对 factual/counterfactual 使用相同初值和未来噪声。

## 4. 候选世界

### 4.1 `control_varma_v1`

功能：线性环形传播与随机创新；检查 VAR 实现、窗口、概率尺度和归一化是否公平。

角色：`CONTROL_LINEAR`。允许 VAR 获胜，不进入神经胜出聚合。

结果：VAR 在三个种子上都优于两个神经候选，控制按设计工作。

### 4.2 `network_cml_v1`

功能：非线性耦合映射、环图、传播路径、冲击、已见/未见耦合机制。

角色：`PRIMARY_MECHANISTIC`。

结果：失败。已见机制中两类模型误差都很低；未见耦合机制中发生明显断裂，神经模型比 VAR 更差，无法支持后续 unseen-regime 研究。

### 4.3 `ecology_lv_sde_v1`

功能：随机 Lotka–Volterra、生长、扩散尺度、定向传播、冲击、已见/未见生态机制。

角色：`PRIMARY_MECHANISTIC`。

结果：失败。已见机制接近持平，但未见机制中 iTransformer 在所有跨度上持续落后 VAR，不能提供稳定神经余量。

## 5. 数据与模型预算

每个世界、每个资格种子使用整轨迹隔离：

| 分区 | 轨迹数 | 每条轨迹窗口数 | 总窗口数 |
|---|---:|---:|---:|
| QUAL_TRAIN | 24 | 217 | 5208 |
| QUAL_TUNE | 8 | 217 | 1736 |
| QUAL_SEEN | 12 | 217 | 2604 |
| QUAL_UNSEEN | 12 | 217 | 2604 |

固定预测设置：历史 32、预测 8、跨度组 `1–2 / 3–5 / 6–8`。归一化统计只从 `QUAL_TRAIN` 计算。

模型预算：

- VAR：lag `1 / 2 / 4 / 8`，ridge `1e-6 / 1e-3 / 1e-1`，只用 `QUAL_TUNE` 选择。
- SmallPatchTST：`d_model=64`、3 层、4 头、dropout 0.1。
- SmallITransformer：`d_model=64`、3 层、4 头、dropout 0.1。
- 神经模型：最多 30 epoch、patience 5、batch 64、learning rate 0.001。
- Gate：整轨迹配对 bootstrap 2000 次，95% 区间。

共生成 18 个神经检查点；运行失败台账为空。生态模型按预声明 early stopping 在 10–17 epoch 停止，其余多数运行到 30 epoch。

## 6. 硬件闸门

本机：Intel i5-10500，6 物理核/12 逻辑线程，约 16 GiB RAM，无 CUDA GPU。

| 指标 | 结果 |
|---|---:|
| 代表探针时间 | 2.068 秒 |
| 探针工作单元 | 2 |
| 完整固定工作单元 | 59,400 |
| 外推完整时间 | 17.063 小时 |
| 外推额外峰值内存 | 1,160,773,632 bytes |
| 探针时可用内存 | 3,566,358,528 bytes |
| 硬件 Gate | PASS |

实际完整运行约 15 分钟；原因是预声明 early stopping 和实际单元速度优于保守线性外推。完整配置没有被缩减。

## 7. 每模型、每种子 CRPS 结果

`改进 = VAR CRPS - 神经 CRPS`；正值表示神经优于 VAR。

| 世界 | 模型 | 种子 | VAR CRPS | 神经 CRPS | 改进 |
|---|---|---:|---:|---:|---:|
| control | iTransformer | 104729 | 0.53 | 0.57 | -0.04 |
| control | iTransformer | 130363 | 0.53 | 0.57 | -0.04 |
| control | iTransformer | 155921 | 0.53 | 0.56 | -0.04 |
| control | PatchTST | 104729 | 0.53 | 0.62 | -0.09 |
| control | PatchTST | 130363 | 0.53 | 0.61 | -0.08 |
| control | PatchTST | 155921 | 0.53 | 0.61 | -0.08 |
| network CML | iTransformer | 104729 | 4.04 | 8.14 | -4.11 |
| network CML | iTransformer | 130363 | 4.04 | 8.18 | -4.15 |
| network CML | iTransformer | 155921 | 4.04 | 8.21 | -4.17 |
| network CML | PatchTST | 104729 | 4.04 | 8.10 | -4.06 |
| network CML | PatchTST | 130363 | 4.04 | 8.08 | -4.05 |
| network CML | PatchTST | 155921 | 4.04 | 8.09 | -4.06 |
| ecology LV | iTransformer | 104729 | 0.24 | 0.26 | -0.02 |
| ecology LV | iTransformer | 130363 | 0.23 | 0.26 | -0.03 |
| ecology LV | iTransformer | 155921 | 0.24 | 0.28 | -0.04 |
| ecology LV | PatchTST | 104729 | 0.24 | 0.36 | -0.12 |
| ecology LV | PatchTST | 130363 | 0.23 | 0.37 | -0.14 |
| ecology LV | PatchTST | 155921 | 0.24 | 0.38 | -0.14 |

## 8. 自动 Gate 结果

### 8.1 网络世界

用于失败判定的较优候选为 PatchTST：

- 三种子改进：`-4.0605 / -4.0485 / -4.0569`
- 整轨迹 bootstrap 均值：`-4.0553`
- 95% 区间：`[-4.9560, -3.1549]`
- 失败项：bootstrap 下界、跨度一致性、MAE、三种子方向、最差机制。

机制分解表明，`coupling_low/mid` 中差距约 `-0.01`，主要失败来自 `coupling_unseen`：

| 未见机制跨度 | VAR CRPS | PatchTST CRPS | 改进 |
|---|---:|---:|---:|
| 1–2 | 6.90 | 16.06 | -9.17 |
| 3–5 | 8.06 | 16.18 | -8.12 |
| 6–8 | 9.26 | 16.28 | -7.02 |

### 8.2 生态世界

用于失败判定的较优候选为 iTransformer：

- 三种子改进：`-0.0180 / -0.0270 / -0.0401`
- 整轨迹 bootstrap 均值：`-0.02835`
- 95% 区间：`[-0.03581, -0.02119]`
- 失败项：bootstrap 下界、跨度一致性、NLL、MAE、三种子方向、最差机制。

已见 `ecology_low/mid` 接近持平；主要失败来自 `ecology_unseen`：

| 未见机制跨度 | VAR CRPS | iTransformer CRPS | 改进 |
|---|---:|---:|---:|
| 1–2 | 0.22 | 0.27 | -0.05 |
| 3–5 | 0.34 | 0.39 | -0.05 |
| 6–8 | 0.43 | 0.49 | -0.07 |

## 9. 对后续研究的影响

v1 不能进入后续正式实验，原因不是缺少 TARCA 操作能力，而是缺少项目需要的稳定神经预测余量：

- 内部位置、捕获、source/base 交换和冻结推理均通过。
- 图、lag、机制、配对噪声和下游语义合同均存在。
- 但网络世界的 unseen 机制发生严重泛化断裂。
- 生态世界在 seen 上接近可用，却在 unseen 上稳定由 VAR 获胜。

如果强行冻结，后续 E02 很可能再次得到“神经模型不稳定胜 VAR”的结论，并且无法区分这是 TARCA 假设失败还是世界本身没有神经余量。因此自动拒绝冻结是正确结果。

## 10. 下一步

不得在看过 v1 结果后修改同一个 v1 参数并重跑。下一步必须由用户另行授权创建 v2，并保留本报告和 v1 失败收据。

v2 应更换世界或外部生成器，而不是只微调当前参数。建议优先：

1. 替换 CML 世界，避免已见机制近退化、未见机制尺度断裂的组合。
2. 保留生态世界作为接近边界的辅助/压力世界，不再把它单独视为“保证神经获胜”的主世界。
3. 从已有公开神经时序基准生成器或冻结数据中，先筛选具有非线性长记忆、交叉变量交互和稳定 OOD 预测余量的外部世界。
4. v2 仍先冻结 WQ-01～WQ-11，再运行同一 WQ-13，不复用 v1 资格结果作为调参目标。

## 11. 证据位置

- 资格配置：`configs/stage1b/qualification_v1.yaml`
- 世界配置：`configs/stage1b/worlds_v1.yaml`
- 完整资格收据：`artifacts/stage1b/qualification_v1_summary.json`
- 忽略的硬件与检查点：`artifacts/stage1b/runtime/`
- 实施规范：`docs/research/stage1b_world_qualification_spec.md`
- 实施计划：`docs/superpowers/plans/2026-08-22-stage1b-world-qualification-implementation.md`
