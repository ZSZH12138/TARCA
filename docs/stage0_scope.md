# TARCA Stage 0 范围

冻结日期：`2026-07-23`

Stage 0 status: COMPLETED_AND_FROZEN

最终裁决日期：`2026-07-25`。Stage 0 已通过范围与交付验收；下述禁止项未在 Stage 0 实现中发现。该裁决只确认本阶段基线完成，不表示 TARCA 方法已经得到科学验证。

## 目标

Stage 0 只建立可审计研究契约、可恢复 CPU 环境、诊断/测试基础设施、第三方来源清单与资源受限 reference smoke 证据。Stage 0 不产生论文方法结果。

## 允许

- 文献审计、碰撞查询、新颖性降级与来源追踪。
- 术语/因果边界、假设台账、预注册和可公开审计的范围文档。
- 任意可配置路径下的隔离 Python 3.11 环境、`uv.lock`、CPU-only tests/CI。
- doctor、manifest、commit resolution、资源门禁、PLOT/DiRoCA 静态/import/极小数值 smoke。
- 在 `configs/`、`data/raw/`、`data/interim/`、`data/processed/`、`experiments/` 创建**空边界**：目录可为空或仅含解释范围的 README。

## 严格禁止（Stage 1+）

- synthetic/regime-switching SCM 正式生成器、counterfactual oracle、数据契约实现。
- PatchTST/iTransformer/Chronos/任何预测器正式训练、微调或数据评估。
- activation cache/intervention engine、时序交换干预、IIC/Cause/Isolation/Completeness 正式实现。
- OT/UOT 机制定位、PLOT 变体、DAS/HyperDAS 训练或子空间学习。
- Group-DRO/Wasserstein-DRO/DiRoCA 正式训练与 sweep。
- 金融/私有数据下载、回测、交易或真实市场因果结论。
- MCQA、Gemma-2-2B、GPU/Slurm、大模型下载、完整 contamination/seed/config sweep。
- 将第三方代码复制进 `src/tarca`，或把无许可证代码当作可改编代码。
- 用占位 Python/YAML/config/测试提前实现 Stage 1 接口。

## Stage 0 / Stage 1 接口裁决

唯一允许的未来接口是**空边界**。Stage 0 不定义 Stage 1 的 dataclass、函数签名、schema、默认 config 或伪实现。README 只能说明“此处属于 Stage 1+，当前为空”，不能包含可执行算法。

Stage 1 只能从以下已冻结交付物开始：`terminology.md`、`assumption_ledger.md`、`preregistration_v0.md`、`novelty_claims.md` 及已通过的 Stage 0 环境/诊断证据。任何 Stage 1 开始前先运行 Gate 0 文献复核。

## 合规判定

Stage 0 合规需要同时满足：

- 必需文档 UTF-8、非空、因果边界逐字存在。
- 环境/doctor/tests/smoke 的实际状态可追溯；import-only 不写成复现。
- `configs/`、`data/*/`、`experiments/` 只有空目录/README。
- `src/tarca` 不含 SCM、intervention、localization、DAS、DRO、finance pipeline。
- 没有“已完成定位/鲁棒抽象/真实市场因果/服务器必需”的越界陈述。

若后续发现 Stage 0 实际包含任一禁止项，则当前 `COMPLETED_AND_FROZEN` 裁决立即失效，阶段状态应改为 `SCOPE_VIOLATION`，修复并重新验收后方可恢复。`PARTIALLY_COMPLETED` 仅用于整个科研项目的 `Research status`，不再与 Stage 0 交付状态混用。
