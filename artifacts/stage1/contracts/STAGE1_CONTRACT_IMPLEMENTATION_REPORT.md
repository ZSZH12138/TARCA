# TARCA Stage 1A Unified Contract Implementation Report

初始生成时间：2026-07-25；最后复核：2026-07-27

## A. 状态

COMPLETED

本报告覆盖以 `383b767` 为基线的 Stage 1A 合同分支变更。实现严格限定在 `src/tarca/contracts/`、`tests/contracts/`、`docs/stage1_unified_data_contract.md` 和本报告，不提前实现 Stage 1 后续模块，不修改 Stage 0 基线，不新增真实训练、真实 adapter、真实干预执行、SCM、OT/DAS/DRO、金融实验或 hooks。

## B. 实现了什么

### 1) 运行时数据契约

- 实现 `WindowBatch` 冻结 dataclass，字段名与权威合同一致：`x_observed_mask`、`y_observed_mask`、`input_feature_names`。
- 对 `x`、`y`、`observed_covariates`、`known_future_covariates`、mask、`regime`、窗口 ID、名称、UTC 时间边界、`forecast_time`、metadata 执行 fail-fast 校验。
- 明确拒绝 `meta` tensor，并将 locked Torch 不支持 finite-validation 的 sparse/non-strided tensor 稳定拒绝；非法输入路径保持 tensor identity/device/dtype/shape/stride/requires_grad 不变。

### 2) 分布输出契约

- 实现 `ForecastDistribution`，覆盖 `mean`、`scale`、`quantiles`、`logits`、`samples`、`window_id`、`target_names`。
- 校验 rank、dtype、device、有限值、严格正尺度、quantile 键规范化和不交叉。

### 3) 概念契约

- 实现 `ConceptBatch`，覆盖 `values`、`valid_mask`、`names`、`window_id`、`computed_from_history_only`、`definition_version`。
- 严格校验概念矩阵、布尔 mask、名称唯一性和 shape 对齐。

### 4) 干预 site/spec 契约

- 实现 `InterventionSite`、`InterventionSpec`、`validate_spec_against_site`。
- 支持 `full_swap` / `subspace_swap` 两类 `InterventionKind`。
- 提供公开工程容差函数 `basis_orthonormality_tolerance(dtype)`，用于数值正交性验证。

### 5) Manifest 与集合级校验

- 实现 `DataSplitSummary`、`WindowContractSummary`、`InterventionPair`、`DataManifest`、`RunManifest`、`MetricRecord`。
- 实现 `validate_disjoint_window_partitions` 与 `validate_intervention_pair_partitions`，阻止 split/pair 泄漏。
- `WindowContractSummary` 强制 `target_names` 与 `known_future_covariate_names` 不相交，
  与运行时 `WindowBatch` 的防泄漏语义一致。
- `pair_id` 为稳定 `sha256:` 前缀的 64 位小写十六进制哈希，只基于规定字段生成，不混入诊断浮点和 partition。

### 6) Adapter 静态协议

- 实现 `ForecastModelAdapter` 纯 `Protocol` 边界，只定义 `adapter_name`、`model_hash`、`is_frozen`、`predict_distribution`、`list_intervention_sites`、`capture`、`intervene`。
- 未提供真实运行时 adapter，实现边界止于静态契约。

### 7) Arrow / Parquet Schema

- 实现 `metrics_by_regime_schema()`、`predictions_schema()`、`intervention_pairs_schema()`。
- 实现 `validate_arrow_schema()`，严格校验字段顺序、类型、nullable 与 schema/field metadata。
- 在 `tmp_path` 中完成最小 Parquet round-trip 测试，验证 `1.0.0` metadata 持久化。

### 8) ArtifactLayout

- 实现安全相对路径与根目录布局验证。
- 明确拒绝 Unix 绝对路径、Windows drive 路径、UNC、`..`、空段、`.`、分隔符注入、resolve 后越狱、父组件 symlink/junction/reparse 逃逸。
- 仅在测试临时目录中验证，不创建正式运行目录。

### 9) 文档与公开 API

- 实现 `src/tarca/contracts/__init__.py` 作为唯一公开导出面，导出 29 个权威符号，不导出 `StrictContractModel`。
- 新增 `docs/stage1_unified_data_contract.md`，说明字段、边界、哈希格式、Arrow schema、ArtifactLayout、版本规则和已知限制。

## C. 没有实现什么

以下内容刻意未实现，属于本阶段边界外：

- 真实 SCM、真实数据下载/生成、真实训练、真实推理流水线；
- 真实 `ForecastModelAdapter` 实现、hooks、干预执行引擎；
- OT、DAS、DRO、局部化、稳健性、指标生产流水线；
- 金融结论、科学结论、因果正确性声明；
- 正式运行目录、正式实验结果、正式模型权重、缓存、数据 payload 入库。

## D. 依赖变化

- 未新增、移除或升级任何直接依赖或传递依赖。
- `pyproject.toml`：仅把既有 80% 分支覆盖门禁扩展到
  `tarca.stage0`、`tarca.contracts` 与 `tarca.data.synthetic`；没有依赖变化。
- `uv.lock`：无变更。
- 无下载新数据、无联网拉包、无外部 LLM 调用。

## E. 验证证据

以下命令均在 `D:\software\MyAnaconda\envs\tarca-stage0` 中执行，工作目录为 `C:\Users\DELL\Desktop\TARCA`。

1. `uv lock --check`
   - 退出码：0
   - 结果：lock 一致。

2. `uv sync --frozen --extra research --group dev`
   - 退出码：0
   - 结果：环境与锁文件一致。

3. `uv run pytest tests/contracts/test_window_batch.py -q -o addopts='--strict-config --strict-markers --no-cov'`
   - 退出码：0
   - 结果：45 passed（包含 meta、sparse/non-strided 及 dense 后端异常分类回归）。

4. `uv run pytest tests/contracts -q -o addopts='--strict-config --strict-markers --no-cov'`
   - 退出码：0
   - 结果：427 passed。

5. `uv run pytest tests/contracts -q -o addopts='--strict-config --strict-markers --cov=tarca.contracts --cov-report=term-missing --cov-fail-under=80'`
   - 退出码：0
   - 结果：427 passed。
   - 合同覆盖率：93.14%。

6. `uv run pytest tests/contracts -q`
   - 退出码：1（2026-07-26 串行复核）
   - 结果：427 passed，但失败原因为冻结的 Stage 0 覆盖 addopts 仍只统计 `tarca.stage0`，导致 focused 合同集 coverage=0.00%。
   - 说明：这是现有 Stage 0 覆盖门禁与 focused Stage 1 合同测试之间的配置限制；未改动该基线配置。

7. `uv run pytest -q`
   - 退出码：0
   - 结果：587 passed, 1 skipped。
   - 冻结的 Stage 0 覆盖率：91.17%。

8. `uv run python -m compileall -q src tests`
   - 退出码：0
   - 结果：源码与测试可编译。

9. `uv run ruff check .`
   - 退出码：0
   - 结果：通过。

10. `uv run ruff format --check .`
    - 退出码：0
    - 结果：通过。

11. `uv run pre-commit run --all-files`
    - 退出码：0
    - 结果：全部 hooks 通过；其中 `reject secrets, models, caches, and data payloads` 通过。

12. `uv run python scripts/doctor.py`
    - 退出码：0
    - 结果：Overall status PASS；GPU/CUDA 按 Stage 0 CPU-only 预期 SKIP，其余 PASS。

13. 静态类型检查
    - 仓库未声明 mypy、pyright 或等价的静态类型检查入口，因此未虚构或临时新增类型门禁。

14. `git diff --check 383b767..HEAD`
    - 退出码：0
    - 结果：无空白/补丁格式错误。

15. `git diff --stat 383b767..HEAD`
    - 结果：
      `26 files changed, 5592 insertions(+)`；文件均位于合同源码、合同测试、Stage 1 合同文档和本报告范围内。

16. `git status --short`
    - Stage 1 合同源码、测试、文档和本报告均已提交，无 Stage 1 未提交文件。
    - 仅保留任务开始前已有的用户工作区变更：`README.md`、`artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md`、`docs/TARCA_具体实施计划.md`、已删除的 `docs/TARCA_项目汇报书.md`、`docs/TARCA_项目计划书.md`、`docs/stage0_scope.md`、`tests/test_operator_docs.py`；这些文件未被本阶段提交。

### 2026-07-27 修复与重新验收

- TDD RED：新增的 manifest 回归首先证明 target 与 known-future 重叠仍会被接受；
  同批四个安全/契约回归共 `4 failed`。
- TDD GREEN：加入跨字段 validator 和共享目录身份守卫后，同一批回归 `4 passed`；
  合同与 synthetic 聚焦回归 `133 passed, 1 skipped`。
- 当前合同测试收集数：`429`；这些测试已包含在全仓通过结果中。
- `python -m pytest -q --cache-clear`：退出码 0，`999 passed, 2 skipped`，
  Stage 0/1 统一分支覆盖率 `90.76%`，高于 80% 门禁。
- 文档、Gate 0 与质量门禁聚焦测试：`16 passed, 1 skipped`。
- `uv lock --check`、`compileall`、`ruff check .`、`ruff format --check .`、
  `pre-commit run --all-files` 均退出 0；Doctor 总状态 `PASS`，CPU-only 的 GPU/CUDA
  项按预期 `SKIP`。
- 原先 focused contracts 被 Stage 0-only coverage 配置误判的问题已消除：仓库现在使用裸
  `--cov` 和多包 `coverage.run.source`，CI 的单一 `pytest -q` 同时约束 Stage 0/1。

## F. 算力结论

- 实际执行环境：Windows 10，CPython 3.11.15，CPU-only PyTorch，12 logical cores / 6 physical cores，15.75 GiB（约 16.9 GB）RAM，无 CUDA GPU，验证时 C 盘可用约 170 GiB。
- 未启用峰值资源 profiler；并行审查期间的两次近似快照显示可用内存约 1.35–3.05 GiB，全部 CPU-only 门禁仍完成。
- 本阶段可在普通日常电脑完成：是。
- 是否需要租用 GPU/CPU 服务器：否。
- 实际任务为 schema/validation/tests/docs 工作负载，未出现显存需求；CPU 与内存占用在本机可承受范围内。

## G. 已知限制

- `frozen=True` 只能冻结 dataclass 字段绑定，不能阻止用户对已传入 tensor 做原地修改；locked Torch 不支持 finite-validation 的 sparse/non-strided tensor 会 fail-closed，而不是获得隐式支持。
- `ForecastModelAdapter` 是静态 `Protocol`，不提供运行时签名校验或真实行为保证。
- `1.0.0` schema 目前只被测试消费，尚未被真实数据生产链路或真实 adapter 消费。
- 本阶段只实现合同与验证，不证明 TARCA 方法本身的科学有效性、金融有效性或因果有效性。
- `1.0.0` 仍是工程 schema，不因本次增加既有文档已经要求的防泄漏校验而升级；
  若未来改变字段或语义，必须按版本策略处理。

## H. 人工核对步骤

1. 运行 `git diff --stat 383b767..HEAD`，确认修改仅限合同、测试、文档和本报告。
2. 检查 Stage 1A 提交 `0863d86` 本身没有提前创建后续模块；当前仓库只因后续单独授权的
   Stage 1B 放行 `src/tarca/data/synthetic/`，仍不得出现 `models/`、`localization/`、
   `robustness/`、真实 hooks/training、OT、DAS、DRO 或 financial 实现。
3. 运行 `uv lock --check`。
4. 运行 `uv run pytest tests/contracts -q`；当前应收集并通过 429 个合同测试，且不再受
   Stage 0-only coverage 配置误判。
5. 若只核对行为，运行
   `uv run pytest tests/contracts -q -o addopts='--strict-config --strict-markers --no-cov'`。
6. 运行 `uv run pytest -q`；当前重新验收基线为 `999 passed, 2 skipped`，
   Stage 0/1 统一分支覆盖率至少 80%。
7. 检查 `pyproject.toml` 的 coverage source 同时包含 `tarca.stage0`、
   `tarca.contracts` 与 `tarca.data.synthetic`。
8. 运行 `uv run ruff check .`、`uv run ruff format --check .`、`uv run pre-commit run --all-files`、`uv run python scripts/doctor.py`，预期全部通过或按 Stage 0 CPU-only 显示 GPU/CUDA SKIP。
9. 打开 `docs/stage1_unified_data_contract.md`，逐项核对字段名、`sha256:` 哈希格式、`strict=True`、Arrow schema、ArtifactLayout 边界。
10. 在 Python REPL 构造一个合法 `WindowBatch`，记录每个输入 tensor 的 `id`、`is`、device、dtype、shape、stride、`requires_grad`，再分别注入 NaN、naive datetime、错误 mask、`meta` tensor、后端不支持 finite-validation 的 sparse tensor；确认立即 `ValueError` 且输入属性未变化。
11. 构造 `ForecastDistribution` 的 `scale=0`、quantile crossing、`meta` tensor 和后端不支持 finite-validation 的 sparse tensor，确认立即报错。
12. 在临时目录写入并读回最小 Parquet，确认 schema metadata 中版本为 `1.0.0`。
13. 通读本报告，确认没有把“合同实现通过”夸大成“方法验证完成”。
