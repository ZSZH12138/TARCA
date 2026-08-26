# Stage1B v2 构建报告

> 状态：`BUILT_NOT_QUALIFIED`
> 日期：2026-08-25
> 正式边界：未执行完整 Stage1B 资格、E01 或 E02；未冻结。

## 1. 功能层结论

Stage1B 的活动实现已从历史 v1 替换为 v2：

- 世界不再依赖一个 Interfere 运行副本，而是按公开论文方程在 TARCA 内重新实现；
- 活动套件包含线性控制、单尺度 Lorenz-96、双尺度 Lorenz-96、GVAR 生态辅助世界和
  修正 CML；
- 每条轨迹在训练前检查有限值、塌缩、伪二周期、动态范围和共享噪声；
- truth 增加有符号图、状态依赖符号、潜变量维度、路径和来源哈希；
- PatchTST 与 iTransformer 已替换为保留官方关键结构的参考适配器；
- 神经资格从“每个 seed/horizon 都必须赢”改为 40 个比较单元、65% 胜率、正 skill、
  seen/unseen 多数和概率护栏；
- 控制、辅助和压力世界不再被错误要求神经胜出；
- v2 收据记录全部来源 commit 和来源清单哈希，正常情况下冻结，用户可授权创建后续版本。

这些功能已经构建并通过测试，但还没有生成“神经相对 VAR 的正式资格结果”。

## 2. 来源证据

| 来源 | 固定 commit | 用途 |
|---|---|---|
| Neural-GC | `c3263d40433aaf3acc94c27a0c1abf9b8e9fcedf` | 稳定 VAR、L96 噪声设定 |
| GVAR | `e06268f87fbe923580af14b2cab26399191747a6` | L96 与捕食者—猎物方程/参数 |
| JMLR 双尺度 L96 | `6f28942f6a703c2b52501d01258ca2708539f209` | 双尺度 L96 方程/参数 |
| Interfere | `adfa3f730019f17c3554dd7e0c181248f785bb8b` | CML 公开公式证据 |
| PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | 模型结构和公开超参数 |
| Time-Series-Library | `4e938a1767106324dd753b2a44832bf870a0252e` | iTransformer 官方实现行为 |

证据文件 SHA-256 保存在 `configs/stage1b/worlds_v2.yaml`；简表保存在
`third_party_manifest/stage1b_sources_v2.yaml`。未知许可证仓库只作为 `REFERENCE_ONLY`
证据，没有复制其代码。

## 3. 短轨迹健康探针

固定 seed 701、seen regime、24 个观测点。`线性残差比` 越接近 0，表示一步线性模型越
容易拟合；它只用于诊断，不是删世界或调参数的标准。

| 世界 | 最小时间标准差 | 一步线性残差比 | 边界事件 | 结果 |
|---|---:|---:|---:|---|
| `control_var_v2` | 0.0668 | 0.1956 | 0 | 通过 |
| `lorenz96_f10_v2` | 2.1932 | 0.00024 | 0 | 通过，但短期 VAR 可能很强 |
| `lorenz96_f40_v2` | 8.2367 | 0.0614 | 0 | 通过 |
| `lorenz96_twoscale_v2` | 4.3574 | 0.3982 | 0 | 通过，非线性余量明显 |
| `gvar_predator_prey_v2` | 3.2621 | 0.000006 | 554 | 通过；官方零裁剪已声明，仅作辅助 |
| `corrected_cml_v2` | 0.3980 | 0.4737 | 0 | 通过 |

这里没有得出“某神经模型已经胜过 VAR”的结论。特别是 F=10 L96 的短期线性诊断是一个
真实风险信号，必须在完整 horizon 和盲轨迹上检验，不能按结果修改世界。

## 4. 硬件门槛

最小代表性探针使用正式尺寸的 `ITransformerReference`、少量 L96 资格轨迹和 1 个 epoch：

- 本机：6 个物理核 / 12 线程、约 16GB RAM、无可用 GPU；
- 探针时间：约 10.70 秒；
- 固定完整工作量：384,000 个批次工作单元；
- 本机外推：约 570.4 小时；
- 内存外推：约 1.77GB；
- 结论：`FAIL(runtime)`，超过 120 小时上限，因此完整资格没有启动。

最低服务器建议：16 CPU 核、32GB RAM、12GB VRAM NVIDIA GPU、40GB 存储，并在服务器
探针中证明小于 120 小时。推荐：32 CPU 核、64GB RAM、24GB VRAM GPU、80GB 存储，
目标 24–72 小时；真正时间必须由服务器端同一探针确认。

探针收据位于 `artifacts/stage1b/runtime/hardware_probe_v2.json`，该运行产物不进入 Git。

## 5. 验证结果

- 全仓测试：`181 passed`；
- 全仓 branch coverage：`83.61%`，超过 80% 门槛；
- Ruff：通过；
- mypy（`src/tarca/stage1b`）：通过；
- 活动世界/配置合同、方程一步对照、paired replay、内部位置交换和微型端到端资格：通过；
- `docs/auth`：未修改；
- E01/E02：未运行；
- v2 freeze：未创建。

覆盖率运行中，既有 Stage0 doctor 子进程曾发生一次不可复现的 60 秒瞬时超时；独立运行
与带覆盖率运行分别约 13–16 秒并通过，随后全仓测试 181/181 通过，因此没有修改 Stage0。

## 6. v1 处置

活动 v1 YAML、资格收据、来源草案、候选草案和旧实施计划已从仓库删除。仓库只保留
`docs/research/stage1b_world_qualification_report_v1.md`，集中记录历史运行、失败事实和根因。
旧本地 v1 二进制检查点缓存也已删除；它们不进入 Git，需要时只能通过历史流程重新生成。

## 7. 下一步

1. 在满足硬件要求的服务器上使用同一 v2 配置运行 `probe`；
2. 只有服务器探针通过 120 小时门槛，才运行完整 Stage1B `qualify`；
3. 审阅全部失败单元、胜率、skill、bootstrap 和护栏；
4. 资格套件通过并获得用户确认后，才创建 `FROZEN v2`；
5. 之后再单独授权 E01/E02。
