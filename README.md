# TARCA

## E01 SCM 真值验证：`PASS / v2`

E01 已完成并统一冻结为 `v2`。最终判定由两部分组成：v2 正式运行的解析环境 E01-A，以及
v1 中已经通过并经原字节哈希复核的 Lorenz-96 E01-B。v1 的失败运行链已经移除，仅保留一份
可校验历史记录；v1 的失败事实没有被改写成成功。

活动指针和正式收据位于：

```text
artifacts/e01/active.json
artifacts/e01/frozen/v2/qualification_receipt.json
artifacts/e01/history/e01_v1_history_record.json
```

重新生成确定性 v2 服务器包：

```powershell
$env:PYTHONPATH = 'deploy/e01/py310;src'
& 'D:\software\MyAnaconda\envs\tarca-stage1b-runtime-py310\python.exe' `
  scripts/prepare_e01_v2_server_bundle.py --repository-root .
```

服务器包、整包 SHA-256 和密封 receipt 只保存在本地 `artifacts/e01/bundle/`，不上传 GitHub。
科学规则见 `docs/research/e01_execution_spec_v2.md`；服务器安全流程见
`docs/research/e01_server_handoff_v2.md`；权威结果与下一任务入口见
`docs/auth/TARCA_E01_HANDOFF_SNAPSHOT_2026-08-30.md`。

TARCA（Temporal Abstraction and Robust Causal Alignment）是一个面向非平稳多变量时间序列的研究项目，目标是在冻结的概率预测模型中定位、验证并比较具有时间语义的内部机制。

## Stage 2 / E02 当前状态

方案 B 的 Stage 2 与 E02 本地实现已完成并进入 `LOCAL_IMPLEMENTATION_COMPLETE`：科学配置、
执行图、双 RTX 4090 调度、正式访问隔离、恢复、只读监控和离线服务器包均已落地。当前仍为
`NOT_RUN_FULL_STAGE2_E02 / REMOTE_SERVER_NOT_CONNECTED`；尚未连接服务器、未执行完整训练，
也未打开 E02 formal 数据。实现证据见
[`stage2_e02_local_implementation_report_v1.md`](docs/research/stage2_e02_local_implementation_report_v1.md)，
开机后固定流程见
[`stage2_e02_server_handoff_v1.md`](docs/research/stage2_e02_server_handoff_v1.md)。

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

本仓库保留 Stage 0 研究合同、Stage 1A 的统一数据/接口/制品边界，以及已经完成资格并冻结
的 Stage1B v2 官方运行时：

```text
artifacts/stage0/       结构化研究合同、环境、Gate 与完成凭证
docs/auth/              项目计划、实施计划、端到端协议和服务器规范
docs/                   范围、相关工作、新颖性、术语、假设和预注册
src/tarca/contracts/    严格、冻结的公共契约
src/tarca/artifacts/    Stage 1A 类型化原子制品仓库与冻结目录
src/tarca/data/         registry 驱动的既有物理窗口读取边界
src/tarca/stage0/       Stage 0 冻结与核验逻辑
src/tarca/stage1b/      官方世界、生成器真值、模型适配、资格 Gate 与单一 v2 冻结
src/tarca/e01/          E01-v2 解析真值、证据验证、任务图与可恢复运行时
src/tarca/stage2/       Stage 2 概率预测、模型选择、冻结和双 GPU 运行时
src/tarca/e02/          E02 formal 隔离、配对评分、bootstrap、决策与 receipt
src/tarca/execution/    不可变任务、双 GPU 调度、恢复和资源遥测
src/tarca/monitoring/   只读监控 API
frontend/stage1b-monitor/ Stage1B 中文运行监控前端
deploy/stage1b/         Python 3.10/CUDA 12.1 容器与 Compose 入口
deploy/e01/             E01-v2 服务器入口与资源监督脚本
deploy/stage2/          Stage 2/E02 CUDA 12.1 容器、preflight 与 supervisor
scripts/                环境、来源、Stage 0 与 Stage 1A 检查入口
tests/stage0/           Stage 0 自动化验证
tests/stage1a/          Stage 1A 高风险边界和最小闭环验证
tests/e01/              E01-v2 科学、证据、运行时和服务器包验证
third_party_manifest/   第三方论文、仓库、版本和许可证边界
```

Stage 1A 自身仍不训练正式模型、不下载或生成正式数据。Stage1B 当前状态为 `FROZEN_V2`：
`lorenz96_twoscale_v2 + ITransformerReference` 已在独立确认种子上通过 h1–6 资格门禁，并以
内容哈希绑定为唯一活动 `v2`；E01 已以 `v2/PASS` 完成，E02 尚未运行。Stage 1A 范围见
[`docs/stage1a_scope.md`](docs/stage1a_scope.md)，Stage1B 当前权威交接见
[`TARCA_STAGE1B_HANDOFF_SNAPSHOT_2026-08-29.md`](docs/auth/TARCA_STAGE1B_HANDOFF_SNAPSHOT_2026-08-29.md)。

## Stage1B v2 服务器入口

服务器不再自行从 GitHub 下载官方来源。先在本地审核并打包六个精确 commit，然后通过
既有安全通道上传源码包与其 receipt；服务器只从这个包重建并验证只读来源缓存。这样远端
GitHub 的可用性、TLS 差异或仓库状态不会影响正式运行。

本地生成可上传包：

```powershell
.\.venv\Scripts\python.exe scripts/package_stage1b_source_capsule.py
```

生成的两个忽略版本控制的文件位于
`artifacts/stage1b/source-capsules/`：`stage1b-v2-official-sources.tar.gz` 与同名
`.receipt.json`。必须一起上传，不能解包、修改或用普通工作目录替代。

在已按服务器 runbook 恢复的运行容器/仓库根目录中，先导入，再预检和启动：

```bash
export PYTHONPATH="$PWD/deploy/stage1b/py310:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TARCA_STAGE1B_SOURCE_MODE=offline-capsule
export TARCA_STAGE1B_SOURCE_CACHE_ROOT="${TARCA_STAGE1B_SOURCE_CACHE_ROOT:-$PWD/third_party/stage1b}"
export TARCA_STAGE1B_ARTIFACT_ROOT="${TARCA_STAGE1B_ARTIFACT_ROOT:-$PWD/artifacts/stage1b/runtime}"
export TARCA_STAGE1B_QUALIFICATION_CONFIG="${TARCA_STAGE1B_QUALIFICATION_CONFIG:-$PWD/configs/stage1b/qualification_v2_confirmation_r2.yaml}"
export TARCA_STAGE1B_DATABASE="${TARCA_STAGE1B_DATABASE:-$TARCA_STAGE1B_ARTIFACT_ROOT/runtime/execution.sqlite3}"
export TARCA_STAGE1B_STATIC_ROOT="${TARCA_STAGE1B_STATIC_ROOT:-$PWD/frontend/stage1b-monitor/dist}"

python scripts/import_stage1b_source_capsule.py \
  --capsule /secure-transfer/stage1b-v2-official-sources.tar.gz \
  --receipt /secure-transfer/stage1b-v2-official-sources.tar.gz.receipt.json \
  --cache-root "$TARCA_STAGE1B_SOURCE_CACHE_ROOT"

bash deploy/stage1b/entrypoint.sh preflight
bash deploy/stage1b/entrypoint.sh launch
```

`import` 会在临时目录中逐个核对外层 SHA-256、manifest、bundle、仓库 URL、commit、
授权 ID、关键资产和完整树哈希；全部通过后才发布缓存。`offline-capsule` 模式下缓存缺失
会直接失败，绝不会退回 GitHub 下载。断电或人为中断后仍使用原有的 `resume`，而不是重跑
`launch`。

若使用 Compose，同样先上传两个文件到服务器的本地目录，再把该目录显式挂载为只读传输目录：

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml build
export TARCA_STAGE1B_SOURCE_TRANSFER_DIR=/secure-transfer
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b-source-import \
  --capsule /opt/tarca/source-transfer/stage1b-v2-official-sources.tar.gz \
  --receipt /opt/tarca/source-transfer/stage1b-v2-official-sources.tar.gz.receipt.json
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b preflight
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm --service-ports stage1b launch
```

当前 Compose 默认运行双尺度短期资格配置；其原始确认配置文件名作为实验来源标识保留，
不构成活动版本号。调度器会把运行目录与已经冻结的 `artifacts/stage1b/frozen/v2` 分开。
调度器每轮都扣除正在运行任务已经占用的 CPU、内存和 GPU；20 GiB 神经任务默认每张满足
准入条件的 GPU 只运行一个，数据任务按探针允许的核数运行，
资源释放后再补位。SQLite 状态、已验证制品和 checkpoint 保存在命名卷中，断电或人为中断后
使用 `resume`，不能重新 `launch`。

监控页面只绑定 `http://127.0.0.1:8765`。远程服务器建议通过 SSH 隧道访问：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

正式运行每 2 秒采集一次主机、进程和 NVML 数据。页面把 10 秒内样本显示为“数据正常”，
旧样本显示为“数据过期”，没有真实样本显示为“遥测不可用”；缺失值不会伪装成 0。训练
任务 ETA 使用真实开始时间和 `completed_steps / total_steps`，证据不足时显示“校准中”。
Stage1B 已冻结为唯一 `v2`；正常情况下禁止覆盖，用户明确授权时必须绑定当前 manifest 哈希
进行原子替换，活动名称仍为 `v2`。完整结果与边界见权威交接快照，硬件合同、恢复和状态
查询见服务器 runbook。

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
D:\software\MyAnaconda\Scripts\conda.exe run -n tarca-local-py311 python -m uv pip install --python .venv\Scripts\python.exe --require-hashes -r deploy/stage1b/requirements-test.lock
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
python -m uv pip install --python .venv/bin/python --require-hashes -r deploy/stage1b/requirements-test.lock
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
