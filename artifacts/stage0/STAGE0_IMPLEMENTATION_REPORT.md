# TARCA Stage 0 公开实施报告

本报告描述 Stage 0 的研究与工程基线，不代表 TARCA 方法已经得到科学验证。

## 1. 最终状态

Research status: PARTIALLY_COMPLETED

Stage 0 已建立研究契约、文献审计、锁定环境、诊断、来源清单、受限 reference smoke、测试与质量门禁。状态不能写为完成，原因有二：

1. 环境创建前的初始硬件门禁没有持久化全部必需字段；后续刷新不能倒填为初始事实。
2. PLOT 仅达到 `PARTIAL`，DiRoCA 仅达到 `IMPORT_ONLY`；二者都不是论文结果复现。

没有发现 Stage 1 算法越界实现。

## 2. 研究目标与因果边界

TARCA 研究高层概念干预与神经网络内部干预能否在时序模型中对齐，以及这种模型内部对齐在分布变化下是否稳健。Stage 0 只为后续可证伪实验建立合同与工具。

TARCA 首篇研究只能对模型内部计算机制提出因果陈述。模型内部干预一致性不能自动推出真实金融市场中的因果关系。

## 3. 已交付模块

| 模块 | Stage 0 作用 |
| --- | --- |
| 术语、假设与预注册 | 冻结研究问题、指标、控制、停止规则和因果边界 |
| 文献与新颖性审计 | 记录来源、碰撞查询、不确定性和声明降级 |
| 环境与 Doctor | 用 Python 3.11、`uv.lock` 和 CPU-only 检查建立可恢复环境 |
| 第三方 manifest | 保存官方 HTTPS 来源、固定 commit、许可证状态和解析结果 |
| reference smoke | 在命令、路径、资源和产物 allowlist 内执行极小检查 |
| 测试与质量门禁 | 检查范围、文档、CLI、安全策略、覆盖率和发布边界 |

`configs/`、`data/` 和 `experiments/` 只定义 Stage 1+ 空边界，不包含算法配置、数据载荷或正式实验。

## 4. 环境、算力与 Doctor

- Stage 0 核心检查只需要普通 CPU、内存和磁盘，不需要 GPU、Slurm 或服务器租用。
- Python 3.11 环境路径由使用者配置；依赖以 `uv.lock` 和 frozen sync 为准。
- Doctor 检查 OS/Python、CPU/RAM、GPU/CUDA、磁盘、写权限、Git/uv、NumPy/Torch、随机性、POT、PyTorch hook 与 pyvene。
- CPU-only 环境中的 GPU/CUDA 项为预期 `SKIP`，不是核心失败。
- Doctor 支持终端、JSON 和 Markdown 输出；包含机器信息的原始输出只保留在本地。

## 5. 文献与新颖性结论

审计覆盖 PLOT、DiRoCA、causal abstraction、IIT/DAS/HyperDAS、时序解释与预测反事实等最近邻方向。当前声明分为 `PARTIALLY_SUPPORTED`、`COLLISION_RISK` 和 `NOT_NOVEL`，没有把检索结果写成“已证明新颖”。

主要结论：

- 多步概率输出的时序交换干预误差，以及容量限制与反信息注入协议，暂时保留为收窄后的候选贡献。
- horizon/lag、四轴渐进定位、状态切换鲁棒抽象和双重真值基准存在碰撞风险，必须由 Stage 1 可证伪实验检验。
- PLOT 引导的 DAS 不能作为新贡献；金融序列只能作为压力测试，不是方法贡献。
- 新文献若直接覆盖候选声明，必须重开 Gate 0 并继续降级。

## 6. 第三方来源与 reference smoke

第三方 commit 解析结果公开保存在 `third_party_commits.json`。该文件只记录来源、固定 SHA、状态与解析方式，不包含本地 clone 路径。

- PLOT：`PARTIAL` / `COMPONENT`。固定源码的静态编译和一个受审计纯 CPU transport primitive 通过；没有运行端到端训练或论文实验。
- DiRoCA：`IMPORT_ONLY` / `STATIC`。只完成静态/import 级检查，避免在缺少 OS/network sandbox 时执行第三方顶层运行时路径。
- 已缓存仓库只有在官方 origin、固定 commit、detached HEAD 和完全干净的工作树同时成立时才允许复用；临时 Python 字节码不会写入第三方源码树。
- 两项摘要均明确 `used_gpu=false`，并明确不是论文结果复现。
- PLOT、DiRoCA 与 HyperDAS 的代码许可证状态仍为 `UNVERIFIED`，其源码没有复制到 TARCA。

## 7. 测试与质量门禁

Stage 0 的发布门禁包括：

1. `uv sync --frozen --extra research --group dev`；
2. Python compileall；
3. pytest 与不低于 80% 的 branch coverage；
4. Ruff check/format；
5. pre-commit 的官方固定 hooks 与仓库卫生策略；
6. Doctor 的 CPU-only 诊断；
7. 从候选发布文件构建的干净检出验证。

发布候选在 2026-07-24 重新验证为：`uv sync --frozen --extra research --group dev` 通过，compileall 通过，pytest `159 passed, 1 skipped`，branch coverage `91.17%`，干净候选检出中的 `pre-commit run --all-files` 全通过，Doctor 为 18 `PASS` / 2 预期 `SKIP`。GitHub Actions 工作流采用 frozen、offline、CPU-only 配置；托管结果以对应提交的远端 Actions 记录为准。

提交前依赖审计在旧锁文件中发现 pytest 1 项、uv 2 项已知漏洞；开发依赖下限已分别提高到 pytest `9.0.3` 与 uv `0.11.15`，当前锁定版本为 pytest `9.1.1` 与 uv `0.11.32`。升级后使用 pip-audit `2.10.1` 复扫，可映射的已安装依赖未发现已知漏洞；`torch 2.13.0+cpu` 不在 PyPI，审计器无法映射该 wheel，因此本报告不声明完整依赖集合“零漏洞”。

## 8. 公开文件与本地证据边界

公开 Stage 0 evidence 仅包含四项：

- 本实施报告；
- `third_party_commits.json`；
- PLOT 的 `result_summary.md`；
- DiRoCA 的 `result_summary.md`。

以下原始证据类别只保留在本地，不进入公共仓库：`hardware_gate.json`、Doctor 的 JSON/Markdown 输出、`command_log.json`、Git inventory/state、环境与包快照，以及 reference smoke 的 command、environment、status、stdout、stderr 和临时 clone。内部工作流笔记、缓存、模型文件、数据载荷与临时日志同样排除。

## 9. 未完成项与残余风险

1. 初始硬件门禁缺失字段无法追溯补录。
2. PLOT 没有执行 HEQ 训练、端到端 PLOT、MCQA、Gemma、Slurm 或 sweep。
3. DiRoCA 没有在可信 sandbox 中执行 YAML/eval、数据生成或优化路径。
4. PLOT、DiRoCA 与 HyperDAS 的代码许可证仍待核验。
5. 第三方可执行文件解析会拒绝工作区内候选，但仍信任 PATH 中工作区外的可执行文件；这是已知非阻断残余风险。
6. Stage 0 不能证明机制定位、鲁棒抽象、预测性能或真实市场因果关系成立。

## 10. Stage 1 交接边界

以下内容尚未实现：

- synthetic/regime-switching SCM、counterfactual oracle 和正式数据契约；
- PatchTST、iTransformer、Chronos 的训练、微调或评估；
- activation cache、intervention engine、交换干预与完整机制指标；
- OT/UOT 定位、PLOT 变体、DAS/HyperDAS 子空间训练；
- Group-DRO、Wasserstein-DRO、DiRoCA 训练或 sweep；
- 金融数据下载、回测、交易或真实市场因果结论。

三份中文 TARCA 项目文档中的 Stage 1+ 描述是未来研究路线图，不是当前实现。Stage 1 只能从冻结的术语、假设台账、预注册、新颖性声明和 Stage 0 环境开始，并在获得单独实施授权前重开 Gate 0 文献复核。
