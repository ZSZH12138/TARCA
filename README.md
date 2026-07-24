# TARCA

TARCA（Temporal Abstract Robust Causal Alignment）研究模型内部的时序机制：高层概念干预能否与神经网络内部干预对齐，以及这种对齐在分布变化下是否稳健。它不是股票预测器，也不把模型内部实验外推为真实市场因果关系。

当前仓库只实施 **Stage 0：研究契约、文献审计与可复现基础设施**。

Research status: PARTIALLY_COMPLETED

## 当前阶段与因果边界

Stage 0 建立术语、假设、预注册、文献与新颖性审计、Python 3.11 锁定环境、CPU 诊断、第三方来源清单、受限 reference smoke、测试和质量入口。Stage 1 的数据、模型训练、机制干预、定位和鲁棒优化内容当前 `不实现`。

TARCA 首篇研究只能对模型内部计算机制提出因果陈述。
模型内部干预一致性不能自动推出真实金融市场中的因果关系。

三份中文项目文档 `docs/TARCA_项目计划书.md`、`docs/TARCA_项目汇报书.md` 和 `docs/TARCA_具体实施计划.md` 描述完整研究构想，其中 Stage 1+ 内容是**未来路线图**，不是本仓库已经实现的代码或实验结果。当前实现边界以本 README、`docs/stage0_scope.md` 和测试合同为准。

## 目录与模块连接

```text
.
├── docs/                         # 术语、假设、预注册、文献、新颖性与范围契约
├── src/tarca/stage0/             # 诊断、来源解析、资源门禁与安全 smoke 框架
├── scripts/                      # doctor 和 reference-smoke 命令入口
├── tests/                        # Stage 0 合同、单元、集成与安全测试
├── third_party_manifest/         # 官方来源、固定 commit 与许可证状态
├── artifacts/stage0/             # 四项公开摘要；原始运行证据仅保留在本地
├── configs/                      # Stage 1+ 空边界；当前没有算法配置
├── data/                         # 数据空边界；当前没有数据载荷
├── experiments/                  # Stage 1+ 空边界；当前没有正式实验
├── pyproject.toml
└── uv.lock
```

文献与术语契约约束 `src/tarca/stage0/` 的实现；manifest 为 reference smoke 提供可信来源和固定 commit；两个 `scripts/` 入口调用 Stage 0 模块并产生本地证据；测试、pre-commit 和 CPU-only CI 检查这些边界。公开仓库只保留实施报告、commit 解析结果及 PLOT/DiRoCA 两份摘要，原始日志和机器信息不发布。

## Windows：可配置 Conda 环境

在 Windows 上先创建任意路径的隔离 Python 3.11 Conda 环境，然后通过 `TARCA_CONDA_PREFIX` 指定它。不要把个人机器路径写入仓库。

```powershell
$env:TARCA_CONDA_PREFIX='C:\path\to\conda\envs\tarca-stage0'
$env:UV_PROJECT_ENVIRONMENT=$env:TARCA_CONDA_PREFIX
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" sync --frozen --extra research --group dev
```

后续 Windows 命令均沿用这两个环境变量和同一个 `uv.exe`。

## Linux/macOS/CI：plain uv

Linux、macOS 和 CI 使用 Python 3.11 与 plain uv；默认项目环境为仓库内的 `.venv`：

```bash
uv sync --frozen --extra research --group dev
uv run python scripts/doctor.py
uv run pytest -q
uv run pre-commit run --all-files
```

所有平台都以 `uv.lock` 为依赖图，并使用 `--frozen` 防止验证时隐式改写锁文件。

## Doctor：三种输出模式

Windows 示例：

```powershell
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/doctor.py
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/doctor.py --json artifacts/stage0/doctor_report.json
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/doctor.py --markdown artifacts/stage0/doctor_report.md
```

Linux/macOS/CI 将前缀替换为 `uv run`。Doctor 检查解释器、CPU/内存/磁盘、项目写权限、CPU 数值、可复现性、POT、PyTorch hook 与 pyvene；它不下载模型或数据。

诊断状态：

- `PASS`：满足 Stage 0 合同。
- `WARN`：非核心限制，需要记录。
- `SKIP`：按边界有意不执行，例如 CPU-only 环境中的 CUDA/GPU 路径。
- `FAIL`：核心检查失败。

## 测试、覆盖率、pre-commit 与 Make

```powershell
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run pytest -q
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run pytest --cov=tarca.stage0 --cov-report=term-missing
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run pre-commit run --all-files
```

Linux/macOS/CI 使用对应的 `uv run pytest`、`uv run pytest --cov=tarca.stage0` 和 `uv run pre-commit run --all-files`。覆盖率门禁为 80%。

如果系统提供 GNU Make，可使用：

```text
make doctor
make smoke
make test
make lint
make stage0-check
```

Windows Make 从 `TARCA_CONDA_PREFIX` 或 `UV_PROJECT_ENVIRONMENT` 获取环境前缀；其他平台直接使用 `uv`。`make smoke` 只运行本项目的 doctor/POT/hook 小检查，不运行 PLOT 或 DiRoCA 论文实验。

## 本地 CPU smoke 与第三方 reference smoke

```powershell
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/run_reference_smoke.py plot
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/run_reference_smoke.py diroca
```

Linux/macOS 使用：

```bash
uv run python scripts/run_reference_smoke.py plot
uv run python scripts/run_reference_smoke.py diroca
```

Reference 状态：

- `IMPORT_ONLY`：仅完成静态编译、help 或 import 级检查；`不是论文复现`。
- `SMOKE_PASSED`：受限 allowlist smoke 与预期微型产物通过；仍不是论文结果复现。
- `PARTIAL`：部分受限步骤通过，但不是端到端论文复现。
- `BLOCKED_BY_HARDWARE`：资源门禁阻止继续运行。
- `BLOCKED_BY_DEPENDENCY`：继续执行会污染主环境或需要未锁定依赖，因此停止。
- `FAILED`：allowlist 步骤、策略或证据合同失败。

当前公开摘要记录：

- PLOT：`PARTIAL` / `COMPONENT`。固定 commit `96dbec5f04bc03aea6e55c430eeafd5c9be27fb2` 的静态编译和一个受审计纯 CPU transport primitive 通过，未运行端到端 PLOT。
- DiRoCA：`IMPORT_ONLY` / `STATIC`。固定 commit `7002947b4954abea1f3d11fcb6f36e7f3c43e8bd` 只完成静态/import 级检查。
- PLOT、DiRoCA 与 HyperDAS 的代码许可证元数据保持 `UNVERIFIED`，因此第三方源码不复制到 TARCA。

## 硬件与第三方门禁

`LOCAL_OK` 表示普通 CPU 工作站足以运行 Stage 0 核心检查；它不是特定机器的性能承诺。Stage 0 `不需要 GPU`，CUDA/GPU 不可用是预期 `SKIP`，不是核心失败，也不需要租用服务器。

第三方运行必须来自 manifest 中的 HTTPS 官方仓库和固定 commit，并受命令、路径、链接、资源与产物 allowlist 限制。没有 OS/network sandbox 时，可能执行第三方顶层代码的路径必须降级为静态或 import-only 检查。

## Stage 0 禁止事项

- 不实现 synthetic `SCM`、正式模型训练、微调、activation cache、intervention engine、OT localization、`DAS`/HyperDAS 或 `DRO`。
- 不下载 `金融数据`，不进行回测或交易，不声称真实市场因果关系。
- 不下载或运行大模型、`MCQA`、`Gemma`、GPU/`Slurm`、全量 sweep 或多种子正式实验。
- 不执行第三方不可信 YAML/eval 路径，不把无明确许可的第三方源码复制进 `src/tarca/`。

## 已知限制

- 初始硬件门禁没有在环境创建前持久化全部必需字段；后续验收刷新不能倒填为初始事实，因此研究状态保持部分完成。
- PLOT 仅为 `PARTIAL`，DiRoCA 仅为 `IMPORT_ONLY`；二者均未复现论文结果。
- 第三方代码和文献元数据中的不确定字段保持 `UNVERIFIED` 或 `UNRESOLVED`。
- GitHub Actions 工作流已配置为 frozen、offline、CPU-only 门禁；本地验证不能替代托管工作流的实际运行结果，本 README 不宣称它已经通过。

## Stage 1 交接

Stage 1 只能从冻结的术语、假设台账、预注册、文献复核、新颖性声明和 Stage 0 验收结果开始，并先重开 Gate 0 文献复核。当前 `configs/`、`data/` 与 `experiments/` 仅保留空边界；schema、数据契约、SCM、训练、定位、干预与鲁棒优化入口都尚未实现。
