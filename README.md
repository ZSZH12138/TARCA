# TARCA

TARCA（Temporal Abstraction and Robust Causal Alignment）是一个面向非平稳多变量时间序列的研究项目，目标是在冻结的概率预测模型中定位、验证并比较具有时间语义的内部机制。

项目关注三个彼此关联的问题：神经网络内部位置是否与预先定义的高层因果变量保持干预一致性；forecast horizon 与 causal lag 能否被独立定位；已经识别的机制能否在未见状态变化下保持 zero-refit 的解释有效性。金融序列只作为后期高难度压力测试，不构成方法新颖性来源。

## 研究路线

TARCA 的计划研究链路为：

1. 冻结多步概率时间序列预测器及其数据、切分、种子和 checkpoint 身份；
2. 使用合成 SCM 与 paired interventions 定义可核验的高层因果效应；
3. 在变量、causal lag、forecast horizon 和受限表示子空间上执行内部干预；
4. 比较高层与低层效应签名，并使用 OT/DAS 类方法进行机制定位；
5. 在 sequential unseen regimes 上评估冻结解释器的 zero-refit 鲁棒性；
6. 通过负对照、反信息注入检查、统计检验和跨域实验限制可声明的结论。

所有候选贡献都必须保持可证伪。一般性因果抽象、PLOT-guided DAS、通用 Wasserstein 鲁棒抽象、一般时间序列机制解释和金融应用本身均不作为 TARCA 的新颖性声明。

## 科学边界

- 模型内部干预支持的是模型计算因果，不自动等同于真实世界因果；
- forecast horizon 与 causal lag 是两条不同的时间轴；
- predictor、位置、映射、normalizer 或环境定义在 test/unseen 阶段重新拟合时，不得声明 zero-refit；
- 服务器、本地 CPU 和多卡环境都只是可替换的 Execution Plane backend，不构成项目算力上限；
- 计划书和协议书定义规范，不保存某次实施运行的状态。

## 仓库范围

本仓库保留 Stage 0 研究合同、Stage 1A 的统一数据/接口/制品边界，以及已构建但尚未执行
完整资格的 Stage1B v2 官方运行时：

```text
artifacts/stage0/       结构化研究合同、环境、Gate 与完成凭证
docs/auth/              项目计划、实施计划、端到端协议和服务器规范
docs/                   范围、相关工作、新颖性、术语、假设和预注册
src/tarca/contracts/    严格、冻结的公共契约
src/tarca/artifacts/    Stage 1A 类型化原子制品仓库与冻结目录
src/tarca/data/         registry 驱动的既有物理窗口读取边界
src/tarca/stage0/       Stage 0 冻结与核验逻辑
src/tarca/stage1b/      官方世界、生成器真值、模型适配、资格 Gate 与冻结修订
src/tarca/execution/    不可变任务、双 GPU 调度、恢复和资源遥测
src/tarca/monitoring/   只读监控 API
frontend/stage1b-monitor/ Stage1B 中文运行监控前端
deploy/stage1b/         Python 3.10/CUDA 12.1 容器与 Compose 入口
scripts/                环境、来源、Stage 0 与 Stage 1A 检查入口
tests/stage0/           Stage 0 自动化验证
tests/stage1a/          Stage 1A 高风险边界和最小闭环验证
third_party_manifest/   第三方论文、仓库、版本和许可证边界
```

Stage 1A 自身仍不训练正式模型、不下载或生成正式数据。Stage1B v2 的实现与服务器运行时
已经存在，但当前状态严格是 `BUILT_NOT_QUALIFIED`：未运行完整 Stage1B、未运行 E01/E02、
未冻结。Stage 1A 范围见 [`docs/stage1a_scope.md`](docs/stage1a_scope.md)，Stage1B 交接见
[`stage1b_v2_official_runtime_build_report.md`](docs/research/stage1b_v2_official_runtime_build_report.md)。

## Stage1B v2 服务器入口

全新服务器先构建镜像并用独立可写容器初始化固定官方来源；正式资格容器随后只读使用该
来源卷：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml build
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b-source-init
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b preflight
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm --service-ports stage1b launch
```

正式运行会先把完整 74 节点计划登记到状态库。调度器每轮都扣除正在运行任务已经占用的
CPU、内存和 GPU；20 GiB 神经任务默认每张 RTX 4090 只运行一个，数据任务最多使用 24 核，
资源释放后再补位。SQLite 状态、已验证制品和 checkpoint 保存在命名卷中，断电或人为中断后
使用 `resume`，不能重新 `launch`。

监控页面只绑定 `http://127.0.0.1:8765`。远程服务器建议通过 SSH 隧道访问：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

正式运行每 2 秒采集一次主机、进程和 NVML 数据。页面把 10 秒内样本显示为“数据正常”，
旧样本显示为“数据过期”，没有真实样本显示为“遥测不可用”；缺失值不会伪装成 0。训练
任务 ETA 使用真实开始时间和 `completed_steps / total_steps`，证据不足时显示“校准中”。只有
完整资格 Gate 通过并审阅后，才冻结为 `v2-r1`；用户授权覆盖会创建下一不可变 v2 修订，
不会覆盖已有修订。完整硬件合同、恢复、状态查询和冻结命令见上述交接报告。

## 权威文档

发生冲突时按以下职责解释：

1. [`TARCA_项目计划书.md`](docs/auth/TARCA_项目计划书.md)：研究问题、候选贡献、证据等级和 Gate；
2. [`TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`](docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md)：公共类型、函数和 Stage I/O；
3. [`TARCA_具体实施计划.md`](docs/auth/TARCA_具体实施计划.md)：实施顺序、验证方式和验收边界；
4. [`TARCA_SERVER_ACCESS_RUNBOOK.md`](docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md)：服务器接入的安全规范。

冻结的研究边界由 `docs/preregistration_v0.md`、`docs/assumption_ledger.md`、`docs/novelty_claims.md` 和 `docs/terminology.md` 共同给出。

## 环境恢复与核验

项目使用 `pyproject.toml` 与 `uv.lock` 固定依赖。既有 Conda 环境只作为 bootstrap 解释器，依赖安装到仓库内隔离的 `.venv`，不向既有环境安装或升级包。

Windows：

```powershell
D:\software\MyAnaconda\Scripts\conda.exe run -n tarca-local-py311 python -m uv sync --frozen --extra research --group dev
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe scripts/run_reference_smoke.py --network
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/check_stage0.py
.\.venv\Scripts\python.exe scripts/check_stage1a.py --json
```

Linux 或服务器：

```bash
python -m uv sync --frozen --extra research --group dev
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/run_reference_smoke.py --network
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python scripts/check_stage0.py
.venv/bin/python scripts/check_stage1a.py --json
```

`artifacts/stage0/environment_profile.json` 只记录默认可复现起点。更换本地或服务器 backend 时，仍须保持依赖、数据、模型、种子、指标和 Gate 的 scientific identity 不变。

## Gate 0 与结构化产物

Gate 0 是经人工核验并授权签发的新颖性决策。仓库不实现自动新颖性判断器，只校验 Gate decision 的 schema、状态、证据类型和内容哈希。

Stage 0 的结构化产物位于 `artifacts/stage0/`：

- `research_contract_manifest.json`
- `gate0_decision.json`
- `artifact_index.json`
- `environment_profile.json`
- `environment_bundle.json`
- `related_work_bundle.json`
- `stage0_completion_receipt.json`

这些 JSON 文件是可机器核验的合同或凭证；阶段性说明、运行日志、缓存、覆盖率文件和本地环境不进入版本库。
