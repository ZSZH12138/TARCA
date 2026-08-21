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

本仓库保留 Stage 0 研究合同，以及 Stage 1A 建立的统一数据、接口和制品边界：

```text
artifacts/stage0/       结构化研究合同、环境、Gate 与完成凭证
docs/auth/              项目计划、实施计划、端到端协议和服务器规范
docs/                   范围、相关工作、新颖性、术语、假设和预注册
src/tarca/contracts/    严格、冻结的公共契约
src/tarca/artifacts/    Stage 1A 类型化原子制品仓库与冻结目录
src/tarca/data/         registry 驱动的既有物理窗口读取边界
src/tarca/stage0/       Stage 0 冻结与核验逻辑
scripts/                环境、来源、Stage 0 与 Stage 1A 检查入口
tests/stage0/           Stage 0 自动化验证
tests/stage1a/          Stage 1A 高风险边界和最小闭环验证
third_party_manifest/   第三方论文、仓库、版本和许可证边界
```

Stage 1A 不训练正式模型、不下载或生成正式数据，也不包含 Stage 1B 的 SCM 生成、内部干预、OT、DAS 或 DRO 实现。其范围和交接见 [`docs/stage1a_scope.md`](docs/stage1a_scope.md)。

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
