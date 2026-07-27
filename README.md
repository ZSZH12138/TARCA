# TARCA

TARCA（Temporal Abstract Robust Causal Alignment）研究模型内部的时序机制：高层概念干预能否与神经网络内部干预对齐，以及这种对齐在分布变化下是否稳健。它不是股票预测器，也不把模型内部实验外推为真实市场因果关系。

当前仓库已完成并冻结 **Stage 0：研究契约、文献审计与可复现基础设施**，并完成
**Stage 1A：统一数据契约**与 **Stage 1B：合成 regime-switching SCM** 的工程实现和验证。

Stage 0 status: COMPLETED_AND_FROZEN
Stage 1A status: COMPLETED
Stage 1B status: COMPLETED_ENGINEERING
Scientific status: ENGINEERING_SMOKE_ONLY
Research status: PARTIALLY_COMPLETED

## 当前阶段与因果边界

Stage 0 的交付状态、Stage 1 的工程状态与整个项目的科研完成度是三个独立字段。
Stage 0 已验收并冻结；Stage 1A/1B 已交付统一契约、合成真值、paired counterfactual
oracle 和 E01 工程 smoke；整个科研项目仍为 `PARTIALLY_COMPLETED`，因为 Stage 2 预测器及
Gate A/1/2、正式统计实验、跨域实验和金融压力测试尚未实施。当前
`ENGINEERING_SMOKE_PASS` 只表示工程不变量和受限 CPU smoke 通过，**不构成正式科学验证**。

TARCA 首篇研究只能对模型内部计算机制提出因果陈述。
模型内部干预一致性不能自动推出真实金融市场中的因果关系。

两份中文项目文档 `docs/TARCA_项目计划书.md` 和 `docs/TARCA_具体实施计划.md` 描述完整
研究构想。Stage 1A/1B 已按各自权威合同实现；Stage 2+ 仍是**未来路线图**。当前边界同时
受冻结研究契约、两份 Stage 1 设计文档、实施报告和测试合同约束。

## 目录与模块连接

```text
.
├── docs/                         # 术语、假设、预注册、文献、新颖性与范围契约
├── src/tarca/stage0/             # 诊断、来源解析、资源门禁与安全 smoke 框架
├── src/tarca/contracts/          # Stage 1A 唯一跨模块数据与产物契约
├── src/tarca/data/synthetic/     # Stage 1B 合成 SCM、oracle、持久化与验证
├── scripts/                      # doctor、reference smoke、合成构建和 oracle smoke
├── tests/                        # Stage 0/1 合同、单元、集成与安全测试
├── third_party_manifest/         # 官方来源、固定 commit 与许可证状态
├── artifacts/stage0/             # 四项公开摘要；原始运行证据仅保留在本地
├── artifacts/stage1/             # Stage 1A/1B 工程实施报告与小型 smoke 摘要
├── configs/synthetic/            # easy/medium/hard 合成配置
├── data/                         # 生成载荷不入 Git，仅保留边界 README
├── experiments/                  # Stage 2+ 空边界；当前没有正式实验
├── pyproject.toml
└── uv.lock
```

冻结文献、术语、假设和预注册约束全部后续模块；`tarca.contracts` 是 Stage 1 起唯一跨模块
契约来源；`tarca.data.synthetic` 只消费该契约并产生合成真值。测试、pre-commit 和
CPU-only CI 同时检查 Stage 0/1，生成数据和原始机器日志不进入 Git。

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
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run pytest --cov --cov-report=term-missing
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run pre-commit run --all-files
```

Linux/macOS/CI 使用对应的 `uv run pytest`、`uv run pytest --cov` 和
`uv run pre-commit run --all-files`。覆盖率门禁同时覆盖 Stage 0、`tarca.contracts` 与
`tarca.data.synthetic`，最低为 80%。

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

第三方运行必须来自 manifest 中的 HTTPS 官方仓库和固定 commit，并受命令、路径、链接、资源与产物 allowlist 限制。复用的源码缓存还必须处于 detached HEAD 且工作树完全干净（包括 ignored/untracked 文件）；Python 字节码写入独立临时目录，避免 smoke 自身污染缓存。没有 OS/network sandbox 时，可能执行第三方顶层代码的路径必须降级为静态或 import-only 检查。

缓存校验失败时工具会 fail closed，并且不会自动覆盖本地修改。请先检查 `.cache/third_party/<name>`，再将该目录移到备份位置或手动删除，随后重试；工具会从 manifest 中的固定 commit 重新获取源码。

## Stage 0 冻结边界与当前禁止事项

- Stage 0 冻结提交本身不包含 synthetic `SCM`；随后单独授权的 Stage 1A/1B 实现不得反写
  `src/tarca/stage0/` 或篡改 Stage 0 证据。
- 当前仍`不实现`正式模型训练、微调、activation cache、内部表示 intervention engine、
  OT localization、`DAS`/HyperDAS 或 `DRO`。
- 不下载 `金融数据`，不进行回测或交易，不声称真实市场因果关系。
- 不下载或运行大模型、`MCQA`、`Gemma`、GPU/`Slurm`、全量 sweep 或多种子正式实验。
- 不执行第三方不可信 YAML/eval 路径，不把无明确许可的第三方源码复制进 `src/tarca/`。

## 已知限制

- 初始硬件门禁没有在环境创建前持久化全部必需字段；后续验收刷新不能倒填为初始事实，因此研究状态保持部分完成。
- PLOT 仅为 `PARTIAL`，DiRoCA 仅为 `IMPORT_ONLY`；二者均未复现论文结果。
- 第三方代码和文献元数据中的不确定字段保持 `UNVERIFIED` 或 `UNRESOLVED`。
- GitHub Actions 工作流已配置为 frozen、offline、CPU-only 门禁；本地验证不能替代托管工作流的实际运行结果，本 README 不宣称它已经通过。

## Stage 1 工程状态与 Stage 2 交接

Stage 1A 已提供 `WindowBatch`、预测/概念/干预契约、严格 manifest、Arrow Schema 和安全产物
布局。Stage 1B 已提供 Markov regime、trend/scale 潜概念、稳定非线性 VAR、共享未来随机量的
paired oracle、显式缺失、连续 `60/20/10/10` 切分、train-only normalizer、确定性持久化和
CPU-only E01 工程 smoke。

```powershell
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/build_synthetic_dataset.py --config configs/synthetic/synthetic_easy.yaml --output data/processed/synthetic-easy --smoke
& "$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" run python scripts/run_synthetic_oracle_smoke.py --config configs/synthetic/synthetic_easy.yaml --output artifacts/local/synthetic-smoke
```

进入 Stage 2 前必须以最新 Gate 0 记录、Stage 1 实施报告和全仓质量门禁为准。Stage 1B 的
工程 smoke 不得升级为 `E01_FORMAL_PASS`、Gate A 通过或 TARCA 科学假设已验证。
