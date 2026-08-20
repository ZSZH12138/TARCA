# TARCA

TARCA 研究面向非平稳多变量时间序列中的时序因果抽象、机制定位和跨状态鲁棒性。金融数据只作为后期压力测试，不构成方法新颖性。

当前仓库从 Stage 0 开始建立研究契约、证据边界和可复现环境。Stage 0 不训练正式模型、不下载正式数据，也不实现 SCM、内部干预、OT、DAS 或 DRO。

## 权威文档

1. `docs/auth/TARCA_项目计划书.md`：研究问题、候选贡献和 Gate。
2. `docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md`：类型、函数和 Stage I/O。
3. `docs/auth/TARCA_具体实施计划.md`：执行顺序、测试和验收。
4. `docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md`：服务器接入的唯一安全规范。

## Windows 环境恢复

既有 Conda 环境只作为 bootstrap 解释器，不向其中安装或升级依赖。项目依赖安装到仓库内的 `.venv`。

```powershell
D:\software\MyAnaconda\Scripts\conda.exe run -n tarca-local-py311 python -m uv sync --frozen --extra research --group dev
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe scripts/run_reference_smoke.py --network
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/check_stage0.py
```

## Linux / 服务器环境恢复

服务器只是可替换的 Execution Plane backend。接入前必须获得用户单独授权。

```bash
python -m uv sync --frozen --extra research --group dev
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/run_reference_smoke.py --network
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python scripts/check_stage0.py
```

`environment_profile.json` 记录当前默认执行环境，用于给出可复现起点；它不是算力上限。用户可授权切换到本地其他环境、单机服务器或多卡 backend，只要继续使用冻结依赖并保持 scientific identity 不变。

首次生成或经授权替换 Stage 0 冻结产物时，先运行 `scripts/check_stage0.py --freeze`。该命令正常停在等待人工 Gate 0 决策/完成凭证的状态；人工决策到位后运行 `scripts/check_stage0.py --complete`，日常核验则运行不带动作参数的 `scripts/check_stage0.py`。替换既有冻结产物还必须同时提供 `--allow-frozen-overwrite` 与 `--authorization-reason`，旧版本会先进入 history。

## Stage 0 输出

- `docs/stage0_scope.md`
- `docs/related_work_matrix.csv`
- `docs/novelty_claims.md`
- `docs/assumption_ledger.md`
- `docs/terminology.md`
- `docs/preregistration_v0.md`
- `third_party_manifest/sources.yaml`
- `artifacts/stage0/research_contract_manifest.json`
- `artifacts/stage0/gate0_decision.json`
- `artifacts/stage0/artifact_index.json`
- `artifacts/stage0/environment_profile.json`
- `artifacts/stage0/stage0_completion_receipt.json`
- `artifacts/stage0/authorized_overwrite_receipt.json`（仅发生授权替换时）

计划书和协议书只定义规范，不保存实施状态。可验证执行结果只存在于结构化 artifact 和 Gate decision 中。

Gate 0 是经人工核验并授权签发的新颖性决策。仓库不实现自动文献判断器，只校验 `gate0_decision.json` 的结构、证据类型和 content hash；若该文件缺失或证据失配，Stage 0 fail closed。
