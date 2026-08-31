# Stage 2 / E02 Local Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. TARCA project policy requires inline single-agent execution; subagents are prohibited. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, freeze, and package the complete local Stage 2 probabilistic forecasting and E02 validation runtime so the approved two-RTX-4090 server can enter preflight immediately after upload.

**Architecture:** Stage 2 reuses the frozen Stage1B Lorenz-96 world, standard `WindowBatch`/`ForecastDistribution` contracts, official PatchTST/iTransformer adapters, checkpoint machinery, execution scheduler, and E01 runtime/bundle patterns. New focused modules own exact v1 configuration, seed isolation, TRAIN-only probabilistic baselines, DLinear, Stage 2 selection/freeze, sealed E02 scoring/decision, resource admission, CLI orchestration, recovery, and deterministic server packaging. Science-plane hashes never depend on worker placement; the execution plane may only choose allowed placement, CPU backfill, inference packing, DDP, and precision after bounded probes.

**Tech Stack:** Python 3.11 local validation, Python 3.10 server compatibility, PyTorch 2.2.2/CUDA 12.1 server, PyTorch 2.13 CPU local tests, Pydantic 2, PyYAML, pytest/pytest-cov, Ruff, mypy, SQLite execution ledger, Docker Compose, Bash.

**Spec:** `docs/superpowers/specs/2026-08-31-stage2-e02-local-runtime-design.md`

## Global Constraints

- Run every local Python command with `D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe`; do not modify that environment. Set process-local `PYTHONPATH` to the repository `src` directory before tests.
- Use `apply_patch` for every repository file edit.
- Preserve the frozen Stage1B manifest SHA-256 `d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25`.
- Preserve the E01-v2 receipt SHA-256 `16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14`.
- Stage 2 identity is `stage2_probabilistic_forecasting_v1`; E02 identity is `e02_predictor_validity_v1`.
- Formal seeds are exactly `(1729, 2718, 3141, 5772, 8111)` and remain unreadable without an E02 grant.
- Development data has three fixed seeds, 24 TRAIN and 8 VALIDATION trajectories per seed, trajectory length 512, history 64, horizon 24.
- E02 has 12 seen and 12 unseen trajectories per formal seed, totaling 120 complete trajectory units.
- The only primary comparison is frozen primary iTransformer against the validation-selected strongest linear baseline, VAR or DLinear.
- PASS requires CRPS skill at least 0.02, 5000 paired stratified trajectory bootstraps with 90% CI lower bound above zero, at least 3/5 positive data seeds, at least 2/3 positive initialization seeds, and every preregistered guardrail.
- Do not run full Stage 2 training, generate formal E02 data, or execute E02 during local implementation.
- Do not connect to, upload to, configure, or start the remote server during this plan.
- Default GPU execution is two independent exclusive workers. DDP is allowed only at at least 30% measured wall-clock reduction.
- Reserve 1 CPU core for scheduler/monitor and 3 for system/I/O; admit at most 24 work cores and 200GiB RAM.
- Require at least 200GiB free local storage and recommend 300GiB before server launch.
- No task may silently change batch size, seeds, epochs, trajectories, horizons, models, precision, or gates after manifest freeze.
- New Stage 2/E02 branch coverage must be at least 80%; the full existing suite, Ruff, and mypy must pass.
- Commit after every task only after its focused tests pass. Preserve unrelated user changes.

## File Responsibility Map

| File or directory | Responsibility |
|---|---|
| `configs/stage2/stage2_v1.yaml` | Exact Stage 2 data, sources, models, training, resource, and authorization configuration |
| `configs/e02/e02_v1.yaml` | Exact formal split, metrics, bootstrap, guardrails, decision, and authorization configuration |
| `src/tarca/stage2/config.py` | Strict configuration models and scientific/runtime hash separation |
| `src/tarca/stage2/seeds.py` | SHA-256 seed derivation and exclusion enforcement |
| `src/tarca/stage2/data.py` | Stage1B bridge, trajectory grouping, TRAIN-only normalizer, development/formal access boundary |
| `src/tarca/stage2/distributions.py` | Gaussian validation, quantiles, scale calibration, serialization helpers |
| `src/tarca/stage2/baselines.py` | Last-value, seasonal-naive, and corrected TRAIN-only VAR predictors |
| `src/tarca/stage2/dlinear.py` | Verified official DLinear loader and `ForecastPredictor` wrapper |
| `src/tarca/stage2/sources.py` | Four-source materialization, capsule verification, and offline import |
| `src/tarca/stage2/training.py` | DLinear/neural training, deterministic checkpoint selection, and resume |
| `src/tarca/stage2/selection.py` | Validation-only strongest-linear and primary-initialization selection |
| `src/tarca/stage2/manifest.py` | Science-plane manifest compilation and hash-stable graph identity |
| `src/tarca/stage2/freeze.py` | Stage 2 evidence validation, freeze publication, and active pointer |
| `src/tarca/stage2/tasks.py` / `jobs.py` / `runner.py` | Execution graph, executor functions, artifact flow, scheduler integration |
| `src/tarca/stage2/resources.py` | 2×4090 admission, CPU backfill, packing, precision, DDP, and ETA gates |
| `src/tarca/stage2/runtime.py` | prepare/dry-run/preflight/launch/resume/status/freeze/recover lifecycle |
| `src/tarca/e02/config.py` | Strict E02 configuration and exact gate validation |
| `src/tarca/e02/grant.py` | Formal acknowledgement and sealed-access grant validation |
| `src/tarca/e02/scoring.py` | Per-trajectory Gaussian metrics, coverage, horizon and regime aggregation |
| `src/tarca/e02/bootstrap.py` | Deterministic 5000-replicate paired stratified trajectory bootstrap |
| `src/tarca/e02/decision.py` | PASS/FAIL/INCONCLUSIVE/NOT_EVALUABLE state machine |
| `src/tarca/e02/tasks.py` / `jobs.py` / `runner.py` | Formal task graph and science-blind execution |
| `src/tarca/e02/receipt.py` / `runtime.py` | Final receipt, finalize boundary, status, resume, recovery |
| `scripts/run_stage2_v1.py` / `run_e02_v1.py` | Stable CLI surfaces |
| `scripts/prepare_stage2_v1_server_bundle.py` | Deterministic offline bundle and receipt |
| `deploy/stage2/` | Python 3.10/CUDA image, Compose, preflight, supervisor, and entrypoint |
| `tests/stage2/` / `tests/e02/` | Unit, integration, runtime, security, and bundle contract tests |

---

### Task 1: Freeze Exact Stage 2 and E02 Configuration Contracts

**Files:**
- Create: `configs/stage2/stage2_v1.yaml`
- Create: `configs/e02/e02_v1.yaml`
- Create: `src/tarca/stage2/__init__.py`
- Create: `src/tarca/e02/__init__.py`
- Create: `src/tarca/stage2/config.py`
- Create: `src/tarca/stage2/seeds.py`
- Create: `src/tarca/e02/config.py`
- Create: `tests/stage2/test_config.py`
- Create: `tests/stage2/test_seeds.py`
- Create: `tests/e02/test_config.py`

**Interfaces:**
- Produces: `derive_namespaced_seed(namespace: str) -> int`
- Produces: `Stage2Config`, `load_stage2_config(path: Path) -> Stage2Config`
- Produces: `E02Config`, `load_e02_config(path: Path) -> E02Config`
- Produces: `Stage2Config.scientific_hash() -> str`, `Stage2Config.runtime_hash() -> str`
- Produces: `E02Config.scientific_hash() -> str`, `E02Config.runtime_hash() -> str`

- [x] **Step 1: Write failing exact-identity and mutation-rejection tests**

```python
def test_repository_stage2_config_has_frozen_identity() -> None:
    config = load_stage2_config(Path("configs/stage2/stage2_v1.yaml"))
    assert config.experiment_id == "stage2_probabilistic_forecasting_v1"
    assert config.data.development_seeds == (669591429, 1840764098, 1185077341)
    assert config.training.initialization_seeds == (1797287582, 883082243, 1933050005)
    assert config.data.history == 64 and config.data.horizon == 24
    assert config.data.trajectories_per_development_seed == {"TRAIN": 24, "VALIDATION": 8}


def test_repository_e02_config_has_frozen_gate() -> None:
    config = load_e02_config(Path("configs/e02/e02_v1.yaml"))
    assert config.formal_seeds == (1729, 2718, 3141, 5772, 8111)
    assert config.bootstrap.replicates == 5000
    assert config.bootstrap.confidence == 0.90
    assert config.gate.minimum_crps_skill == 0.02
    assert config.gate.minimum_positive_data_seeds == 3
    assert config.gate.minimum_positive_initializations == 2
```

- [x] **Step 2: Run the tests and verify import/config failures**

Run:

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2/test_config.py tests/stage2/test_seeds.py tests/e02/test_config.py -q
```

Expected: FAIL because the Stage 2/E02 modules and YAML files do not exist.

- [x] **Step 3: Implement strict immutable models and exact YAML values**

Use `StrictContractModel`, `Literal`, tuple-converting validators, and model validators. `Stage2Config.scientific_payload()` excludes only `runtime_profile`; `E02Config.scientific_payload()` excludes only runtime placement and monitor fields. Add the DLinear source with:

```yaml
source_id: dlinear
repository_url: https://github.com/cure-lab/LTSF-Linear.git
commit: 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
asset_path: models/DLinear.py
asset_sha256: 0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc
```

Implement the derivation exactly:

```python
def derive_namespaced_seed(namespace: str) -> int:
    if not namespace.strip():
        raise ValueError("seed namespace must not be blank")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return 1 + value % 2_147_483_646
```

Validators must compare all configured derived seeds with recalculated values and reject any overlap with Stage1B qualification seeds, E01 seeds, or formal seeds.

- [x] **Step 4: Run focused tests and hash-stability checks**

Run the Task 1 test command twice and assert the two printed scientific hashes are identical.

Expected: all Task 1 tests PASS; changing a runtime CPU value changes only `runtime_hash()`.

- [x] **Step 5: Commit Task 1**

```powershell
git add configs/stage2 configs/e02 src/tarca/stage2 src/tarca/e02/__init__.py src/tarca/e02/config.py tests/stage2 tests/e02/test_config.py
git commit -m "feat: freeze stage2 and e02 configuration"
```

### Task 2: Build the Development Data Bridge and Formal Access Boundary

**Files:**
- Create: `src/tarca/stage2/data.py`
- Create: `tests/stage2/test_data.py`
- Create: `tests/e02/test_formal_boundary.py`
- Modify: `src/tarca/stage2/__init__.py`

**Interfaces:**
- Consumes: `Stage2Config`, Stage1B `TrajectoryRecord`, `prepare_dataset`, `WindowBatch`
- Produces: `Stage2Trajectory`, `Stage2WindowSet`, `Stage2DataBundle`
- Produces: `generate_development_bundle(config, world, *, worker_count: int) -> Stage2DataBundle`
- Produces: `open_formal_bundle(config, e02_config, grant, *, accessed_at) -> Stage2DataBundle`
- Produces: `stack_partition(bundle, partition) -> tuple[Tensor, Tensor, tuple[str, ...]]`

- [x] **Step 1: Write failing tests for counts, lineage, TRAIN-only normalization, and sealed refusal**

```python
def test_development_bundle_has_exact_trajectory_counts(stage2_config: Stage2Config) -> None:
    bundle = tiny_development_bundle(stage2_config)
    assert bundle.trajectory_count("TRAIN") == 72
    assert bundle.trajectory_count("VALIDATION") == 24
    assert set(bundle.normalizer.trajectory_ids) == set(bundle.trajectory_ids("TRAIN"))


def test_formal_bundle_refuses_read_without_grant(stage2_config: Stage2Config, e02_config: E02Config) -> None:
    with pytest.raises(PermissionError, match="sealed access requires a grant"):
        open_formal_bundle(stage2_config, e02_config, None, accessed_at=UTC_NOW)
```

- [x] **Step 2: Run focused tests and observe missing data interfaces**

Run:

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2/test_data.py tests/e02/test_formal_boundary.py -q
```

Expected: FAIL on missing `tarca.stage2.data` symbols.

- [x] **Step 3: Implement immutable trajectory grouping and access-before-read validation**

Map development partitions to standard `TRAIN` and `VALIDATION`; map E02 to `TEST_SEEN_REGIME` and `TEST_UNSEEN_REGIME`. Reuse Stage1B world generation and bridge logic, but create fresh manifests with the Stage 2/E02 seeds. Call `validate_sealed_access(...)` before resolving any formal physical path. Store trajectory ID and regime in each window lineage; forbid windows spanning trajectory or partition boundaries.

```python
@dataclass(frozen=True, slots=True)
class Stage2DataBundle:
    dataset_id: str
    records: tuple[Stage2Trajectory, ...]
    windows: tuple[tuple[DatasetWindowPartition, tuple[WindowSample, ...]], ...]
    normalizer: NormalizationStatistics
    manifest_sha256: str
```

- [x] **Step 4: Run focused tests including a monkeypatch proving no formal reader call occurs**

Expected: counts, lineage, normalization, deterministic replay, collision rejection, and pre-read denial PASS.

- [x] **Step 5: Commit Task 2**

```powershell
git add src/tarca/stage2/data.py src/tarca/stage2/__init__.py tests/stage2/test_data.py tests/e02/test_formal_boundary.py
git commit -m "feat: add stage2 data and sealed formal boundary"
```

### Task 3: Implement Gaussian Utilities and TRAIN-Only Naive/VAR Baselines

**Files:**
- Create: `src/tarca/stage2/distributions.py`
- Create: `src/tarca/stage2/baselines.py`
- Create: `tests/stage2/test_distributions.py`
- Create: `tests/stage2/test_baselines.py`

**Interfaces:**
- Produces: `residual_scale(residuals: Tensor, *, floor: float, ceiling: Tensor) -> Tensor`
- Produces: `gaussian_quantiles(mean: Tensor, scale: Tensor, levels: tuple[float, ...]) -> Mapping[float, Tensor]`
- Produces: `LastValueGaussian.fit(...)`, `SeasonalNaiveGaussian.fit(...)`, `Stage2VARGaussian.fit(...)`
- All predictor classes satisfy `ForecastPredictor.predict_distribution(batch) -> ForecastDistribution`

- [x] **Step 1: Write failing tests that distinguish TRAIN-only from validation leakage**

```python
def test_last_value_scale_ignores_validation_targets() -> None:
    original = LastValueGaussian.fit(TRAIN_X, TRAIN_Y, TARGET_NAMES)
    changed = LastValueGaussian.fit(TRAIN_X, TRAIN_Y, TARGET_NAMES)
    assert torch.equal(original.scale, changed.scale)


def test_seasonal_lag_uses_validation_but_scale_uses_train() -> None:
    model = SeasonalNaiveGaussian.fit(
        TRAIN_X, TRAIN_Y, VALID_X, VALID_Y, lags=(8, 16, 32), target_names=TARGET_NAMES
    )
    assert model.selected_lag in (8, 16, 32)
    assert model.scale_source == "TRAIN_ONLY"
```

- [x] **Step 2: Run focused tests and verify missing predictor failures**

Run:

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2/test_distributions.py tests/stage2/test_baselines.py -q
```

- [x] **Step 3: Implement exact diagonal Gaussian behavior**

For Last and Seasonal, compute per-horizon/per-variable TRAIN residual RMS and clamp it to `[1e-4, max(10, 10 × train_target_std)]`. Seasonal chooses the lag only by validation h1-6 Gaussian CRPS. Refactor Stage1B VAR into a Stage 2 wrapper that selects lag/ridge by validation CRPS but estimates residual innovation/covariance from TRAIN only; never reuse the current Stage1B `tune_y` residual scale.

```python
scale = torch.sqrt(torch.mean(residuals.to(torch.float64).square(), dim=0))
scale = torch.minimum(torch.maximum(scale, floor_tensor), ceiling)
```

Generate quantiles for levels `(0.025, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.975)` through `torch.distributions.Normal.icdf` and pass the result through `validate_forecast_distribution`.

- [x] **Step 4: Run tests and existing Stage1B VAR regression tests**

Run Task 3 tests plus `tests/stage1b/test_var_predictor.py`.

Expected: new tests PASS and frozen Stage1B behavior is unchanged.

- [x] **Step 5: Commit Task 3**

```powershell
git add src/tarca/stage2/distributions.py src/tarca/stage2/baselines.py tests/stage2/test_distributions.py tests/stage2/test_baselines.py
git commit -m "feat: add stage2 probabilistic baselines"
```

### Task 4: Add the Verified Official DLinear Predictor

**Files:**
- Create: `src/tarca/stage2/dlinear.py`
- Create: `src/tarca/stage2/sources.py`
- Create: `scripts/materialize_stage2_sources.py`
- Create: `scripts/package_stage2_source_capsule.py`
- Create: `scripts/import_stage2_source_capsule.py`
- Create: `tests/stage2/test_dlinear.py`
- Create: `tests/stage2/test_sources.py`
- Create: `tests/stage2/test_source_capsule.py`
- Modify: `src/tarca/stage2/distributions.py`

**Interfaces:**
- Consumes: verified DLinear source root and `Stage2Config.models.dlinear`
- Produces: `load_official_dlinear(source_root: Path, config: DLinearModelConfig) -> nn.Module`
- Produces: `DLinearGaussian(mean_model, scale, target_names)`
- Produces: `fit_dlinear_cross_fitted(...) -> DLinearTrainingResult`
- Produces: `build_stage2_source_capsule(config, cache_root, output) -> SourceCapsuleReceipt`
- Produces: offline source materialization/import CLIs for all four fixed sources

- [x] **Step 1: Write failing official-source hash, shape, and scale-isolation tests**

```python
def test_official_dlinear_asset_hash_is_required(tmp_path: Path) -> None:
    source = tmp_path / "models" / "DLinear.py"
    source.parent.mkdir(parents=True)
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="DLinear source hash"):
        load_official_dlinear(tmp_path, DLINEAR_CONFIG)


def test_dlinear_distribution_matches_contract(fitted_dlinear: DLinearGaussian) -> None:
    forecast = fitted_dlinear.predict_distribution(WINDOW_BATCH)
    assert forecast.mean.shape == forecast.scale.shape == (2, 24, 8)
    assert bool((forecast.scale > 0).all())


def test_stage2_capsule_contains_exact_four_sources(capsule_receipt: SourceCapsuleReceipt) -> None:
    assert tuple(item.source_id for item in capsule_receipt.sources) == (
        "dlinear", "itransformer", "patchtst", "scoring_rules_l96"
    )
```

- [x] **Step 2: Run the DLinear test and observe missing adapter failures**

Run `pytest tests/stage2/test_dlinear.py tests/stage2/test_sources.py tests/stage2/test_source_capsule.py -q` with the project interpreter.

- [x] **Step 3: Implement verified dynamic loading and deterministic five-fold cross-fitting**

Verify `models/DLinear.py` SHA-256 before `importlib.util.spec_from_file_location`. Construct the official `Model` with `seq_len=64`, `pred_len=24`, `enc_in=dimension`, and `individual=False`. Assign folds by `int(SHA256(trajectory_id), 16) % 5`; use exact fold seeds from the spec; train fold models only on four TRAIN folds and calculate held-out TRAIN residuals. Train the final mean model on all TRAIN with validation MSE early stopping and seed `1797287582`.

Reuse Stage1B `materialize_source`, `build_source_capsule`, and `import_source_capsule` with the four exact `SourceConfig` entries from `Stage2Config`. The package command must create `artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz`; the server import command verifies the outer archive, manifest, Git bundles, commits, trees, and pinned assets before publishing any checkout.

```python
@dataclass(frozen=True, slots=True)
class DLinearTrainingResult:
    predictor: DLinearGaussian
    checkpoint_sha256: str
    cross_fit_scale_sha256: str
    best_epoch: int
    best_validation_mse: float
```

- [x] **Step 4: Run official DLinear, checkpoint/reload, and deterministic repeat tests**

Expected: same inputs and seed produce identical hashes; tampering fails closed; no validation target enters scale estimation.

- [x] **Step 5: Commit Task 4**

```powershell
git add src/tarca/stage2/dlinear.py src/tarca/stage2/distributions.py src/tarca/stage2/sources.py scripts/materialize_stage2_sources.py scripts/package_stage2_source_capsule.py scripts/import_stage2_source_capsule.py tests/stage2/test_dlinear.py tests/stage2/test_sources.py tests/stage2/test_source_capsule.py
git commit -m "feat: integrate official dlinear predictor"
```

### Task 5: Train and Reload the Official Neural Gaussian Predictors

**Files:**
- Create: `src/tarca/stage2/training.py`
- Create: `tests/stage2/test_neural_training.py`
- Modify: `src/tarca/stage1b/training.py`
- Modify: `src/tarca/stage1b/training_checkpoints.py`

**Interfaces:**
- Consumes: `OfficialPatchTSTPredictor`, `OfficialITransformerPredictor`, Stage1B checkpoint primitives
- Produces: `Stage2TrainingPolicy`
- Produces: `train_stage2_neural(model, train, validation, policy, seed, progress_sink) -> Stage2TrainingResult`
- Produces: one best-validation-NLL checkpoint per initialization and a reload verifier

- [ ] **Step 1: Write failing tests for optimizer identity, NLL early stopping, deterministic resume, and probability validity**

```python
def test_stage2_training_policy_is_exact(tmp_path: Path) -> None:
    policy = stage2_policy(tmp_path, model_id="itransformer")
    assert policy.optimizer == "ADAMW"
    assert policy.weight_decay == 0.01
    assert policy.gradient_clip_norm == 1.0
    assert policy.scheduler == "NONE"


def test_resume_preserves_model_optimizer_scaler_and_rng(tmp_path: Path) -> None:
    uninterrupted = run_tiny_training(tmp_path / "a", interrupt_after_epoch=None)
    resumed = run_tiny_training(tmp_path / "b", interrupt_after_epoch=1)
    assert resumed.checkpoint_sha256 == uninterrupted.checkpoint_sha256
```

- [ ] **Step 2: Run Stage 2 and existing Stage1B training tests to capture the RED state**

Run `tests/stage2/test_neural_training.py` and `tests/stage1b/test_training_reproducibility.py`.

- [ ] **Step 3: Extend checkpoint policy without changing Stage1B defaults**

Add optional exact optimizer fields to a new Stage 2 policy rather than changing legacy behavior. Save optimizer hyperparameters, gradient clip, deterministic flags, DataLoader generator state, best validation NLL, model state, optimizer state, scaler state, Python/Torch/CUDA RNG, data hash, source hash, and precision hash. Use:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=policy.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

Early stop and the retained epoch use validation NLL only. After reload, validate finite mean/scale, scale `>0`, output shape, state hash, and an identical fixed-batch forecast.

- [ ] **Step 4: Run new and legacy training suites**

Expected: Stage 2 deterministic/resume tests PASS; all Stage1B training tests PASS unchanged.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/tarca/stage2/training.py src/tarca/stage1b/training.py src/tarca/stage1b/training_checkpoints.py tests/stage2/test_neural_training.py
git commit -m "feat: add reproducible stage2 neural training"
```

### Task 6: Compile Validation Selection, Science Manifest, and Stage 2 Freeze

**Files:**
- Create: `src/tarca/stage2/selection.py`
- Create: `src/tarca/stage2/manifest.py`
- Create: `src/tarca/stage2/freeze.py`
- Create: `tests/stage2/test_selection.py`
- Create: `tests/stage2/test_manifest.py`
- Create: `tests/stage2/test_freeze.py`

**Interfaces:**
- Produces: `select_strongest_linear(validation_scores) -> ModelSelection`
- Produces: `select_primary_initialization(model_id, validation_scores) -> ModelSelection`
- Produces: `Stage2CompilationInputs`, `Stage2Manifest`
- Produces: `compile_stage2_manifest(config, upstream, sources) -> Stage2Manifest`
- Produces: `freeze_stage2_suite(artifact_root, evidence) -> Stage2FreezeReceipt`
- Produces: `verify_frozen_stage2_suite(artifact_root) -> Stage2FreezeReceipt`

- [ ] **Step 1: Write failing tests for validation-only selection and placement-independent hashes**

```python
def test_strongest_linear_uses_only_validation_crps() -> None:
    selected = select_strongest_linear({"VAR": 0.31, "DLINEAR": 0.29})
    assert selected.model_id == "DLINEAR"


def test_science_hash_ignores_worker_placement(stage2_inputs: Stage2CompilationInputs) -> None:
    first = compile_stage2_manifest(stage2_inputs.with_gpu_order((0, 1)))
    second = compile_stage2_manifest(stage2_inputs.with_gpu_order((1, 0)))
    assert first.scientific_sha256 == second.scientific_sha256
```

- [ ] **Step 2: Run selection/manifest/freeze tests and verify missing modules**

Run the three Task 6 test files.

- [ ] **Step 3: Implement deterministic tie-breaking and evidence validation**

Break linear ties by fixed order `DLINEAR`, then `VAR`; break initialization ties by the configured seed order. Every selection receipt contains only VALIDATION artifact refs. Freeze requires six predictor identities, three checkpoints for each neural architecture, the selected linear model, selected primary iTransformer seed, source receipt, normalizer, data manifest, precision receipt, runtime failure list, and zero formal-access events.

```python
class Stage2FreezeReceipt(StrictContractModel):
    schema_version: Literal["tarca-stage2-freeze-v1"]
    status: Literal["FROZEN"]
    scientific_sha256: Sha256Hash
    strongest_linear_model_id: Literal["VAR", "DLINEAR"]
    primary_itransformer_seed: int
    formal_access_event_count: Literal[0]
    receipt_sha256: Sha256Hash
```

- [ ] **Step 4: Run tamper, overwrite, reserved-seed, and reload tests**

Expected: first freeze succeeds on complete fixture; second unapproved freeze refuses overwrite; any hash drift or formal event rejects.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src/tarca/stage2/selection.py src/tarca/stage2/manifest.py src/tarca/stage2/freeze.py tests/stage2/test_selection.py tests/stage2/test_manifest.py tests/stage2/test_freeze.py
git commit -m "feat: freeze stage2 model selection"
```

### Task 7: Implement Per-Trajectory E02 Metrics and Stratified Bootstrap

**Files:**
- Modify: `src/tarca/e02/__init__.py`
- Create: `src/tarca/e02/scoring.py`
- Create: `src/tarca/e02/bootstrap.py`
- Create: `tests/e02/test_scoring.py`
- Create: `tests/e02/test_bootstrap.py`

**Interfaces:**
- Consumes: frozen predictions and complete trajectory lineage
- Produces: `TrajectoryScore`, `ScoreSummary`, `BootstrapInterval`
- Produces: `score_trajectory(prediction, target, lineage) -> TrajectoryScore`
- Produces: `summarize_scores(scores, baseline_scores) -> ScoreSummary`
- Produces: `paired_stratified_bootstrap(neural, baseline, config) -> BootstrapInterval`

- [ ] **Step 1: Write failing analytical Gaussian metric and whole-trajectory bootstrap tests**

```python
def test_bootstrap_resamples_twelve_trajectories_inside_each_stratum() -> None:
    interval = paired_stratified_bootstrap(FIXTURE_NEURAL, FIXTURE_BASELINE, E02_BOOTSTRAP)
    assert interval.replicates == 5000
    assert interval.units_per_replicate == 120
    assert interval.stratum_count == 10


def test_coverage_error_uses_four_nominal_levels() -> None:
    summary = summarize_scores(CALIBRATED_SCORES, BASELINE_SCORES)
    assert summary.coverage_levels == (0.50, 0.80, 0.90, 0.95)
```

- [ ] **Step 2: Run metric/bootstrap tests and verify the RED state**

Run `pytest tests/e02/test_scoring.py tests/e02/test_bootstrap.py -q`.

- [ ] **Step 3: Implement stable Gaussian scoring and deterministic paired resampling**

Compute trajectory means after averaging equally over origins, variables, and the requested horizons. Use Stage1B Gaussian CRPS/NLL primitives after shape validation. For each of 5000 replicates, use `torch.Generator().manual_seed(172657089)`, sample 12 trajectory indices with replacement inside each formal-seed × regime stratum, retain neural/baseline pairing, combine exactly 120 units, and compute skill. Return 5th/95th percentile bounds.

```python
skill = 1.0 - neural_crps / baseline_crps
coverage_error = statistics.fmean(abs(observed[level] - level) for level in LEVELS)
```

- [ ] **Step 4: Run deterministic repeat, ordering-invariance, and malformed-lineage tests**

Expected: identical results under score input reordering; duplicate/missing trajectory IDs, 119 units, wrong strata, or window-level lineage fail closed.

- [ ] **Step 5: Commit Task 7**

```powershell
git add src/tarca/e02/__init__.py src/tarca/e02/scoring.py src/tarca/e02/bootstrap.py tests/e02/test_scoring.py tests/e02/test_bootstrap.py
git commit -m "feat: add e02 trajectory scoring and bootstrap"
```

### Task 8: Implement the Four-State E02 Decision and Final Receipt

**Files:**
- Create: `src/tarca/e02/decision.py`
- Create: `src/tarca/e02/receipt.py`
- Create: `tests/e02/test_decision.py`
- Create: `tests/e02/test_receipt.py`

**Interfaces:**
- Consumes: `E02Config`, `ScoreSummary`, `BootstrapInterval`, completion/integrity evidence
- Produces: `E02Outcome = Literal["PASS", "FAIL", "INCONCLUSIVE", "NOT_EVALUABLE"]`
- Produces: `evaluate_e02(evidence: E02Evidence, config: E02Config) -> E02Decision`
- Produces: `build_e02_receipt(decision, evidence) -> E02Receipt`

- [ ] **Step 1: Write a table-driven failing test for every equality boundary and precedence rule**

```python
@pytest.mark.parametrize(
    ("skill", "ci_lower", "expected"),
    ((0.02, 0.0001, "PASS"), (0.019999, 0.0001, "INCONCLUSIVE"),
     (0.02, 0.0, "INCONCLUSIVE"), (-0.0001, 0.01, "FAIL")),
)
def test_primary_gate_boundaries(skill: float, ci_lower: float, expected: str) -> None:
    evidence = passing_evidence().with_primary(skill=skill, ci_lower=ci_lower)
    assert evaluate_e02(evidence, E02_CONFIG).outcome == expected
```

- [ ] **Step 2: Run decision/receipt tests and observe missing state machine**

Run Task 8 tests.

- [ ] **Step 3: Implement explicit precedence**

Order checks as: integrity violation or scientific guardrail breach → `FAIL`; incomplete operational run without integrity breach → `NOT_EVALUABLE`; all PASS conditions → `PASS`; remaining nonnegative primary skill → `INCONCLUSIVE`; negative primary skill → `FAIL`. Record every condition as a named immutable `GateResult` so receipt review does not infer reasons from free text.

```python
class GateResult(StrictContractModel):
    gate_id: str
    passed: bool
    observed: float | int | str
    required: float | int | str
```

- [ ] **Step 4: Run all PASS/FAIL/INCONCLUSIVE/NOT_EVALUABLE and receipt-tamper tests**

Expected: all exact thresholds and precedence cases PASS; receipt hash changes when any evidence changes.

- [ ] **Step 5: Commit Task 8**

```powershell
git add src/tarca/e02/decision.py src/tarca/e02/receipt.py tests/e02/test_decision.py tests/e02/test_receipt.py
git commit -m "feat: add e02 decision and receipt"
```

### Task 9: Compile Stage 2 and E02 Task Graphs and Executor Jobs

**Files:**
- Create: `src/tarca/stage2/tasks.py`
- Create: `src/tarca/stage2/jobs.py`
- Create: `src/tarca/stage2/runner.py`
- Create: `src/tarca/e02/tasks.py`
- Create: `src/tarca/e02/jobs.py`
- Create: `src/tarca/e02/runner.py`
- Create: `tests/stage2/test_tasks.py`
- Create: `tests/stage2/test_runner.py`
- Create: `tests/e02/test_tasks.py`
- Create: `tests/e02/test_runner.py`

**Interfaces:**
- Consumes: `TaskSpec`, `RunPlanNode`, `ExecutorRegistry`, `ExecutionStateStore`, prior Task outputs
- Produces: `compile_stage2_graph(config, inputs) -> Stage2Graph`
- Produces: `stage2_executor_registry(repository_root) -> ExecutorRegistry`
- Produces: `run_stage2(plan, capacity, ...) -> Stage2RunResult`
- Produces: `compile_e02_graph(config, frozen_stage2) -> E02Graph`
- Produces: `e02_executor_registry(repository_root) -> ExecutorRegistry`
- Produces: `run_e02_formal(...) -> E02RunResult`

- [ ] **Step 1: Write failing graph-count, dependency, resource, and no-formal-access tests**

```python
def test_stage2_graph_has_six_independent_large_gpu_training_tasks() -> None:
    graph = compile_stage2_graph(STAGE2_CONFIG, FROZEN_UPSTREAM)
    gpu_train = tuple(node for node in graph.nodes if node.phase == "NEURAL_TRAIN")
    assert len(gpu_train) == 6
    assert all(node.resource_request.gpu_count == 1 for node in gpu_train)
    assert all(node.resource_request.cpu_threads == 4 for node in gpu_train)


def test_e02_graph_requires_frozen_stage2_and_grant() -> None:
    with pytest.raises(PermissionError):
        compile_e02_graph(E02_CONFIG, unfrozen_stage2_fixture())
```

- [ ] **Step 2: Run graph/runner tests and verify the RED state**

Run the four Task 9 test files.

- [ ] **Step 3: Implement immutable artifact-driven DAGs**

Stage 2 phases are source verification, development data, baseline fits, six neural trainings, checkpoint validation, validation prediction, selection, freeze candidate, and receipt. E02 phases are grant verification, formal data generation/open, prediction for the fixed linear baseline and all three iTransformer initializations, per-trajectory scoring, bootstrap, decision, and final receipt. Every input is an `ArtifactRef`; no job locates “latest” files.

Use `completed_task_policy="NEVER_RERUN"`. Bind executors by exact phase names in allowlisted registries. Keep formal metrics out of progress events.

- [ ] **Step 4: Run scheduler integration, retry-history, and placement-invariance tests**

Expected: all dependencies execute once, completed artifacts are reused, attempts remain in SQLite, and swapping GPU assignment preserves science hashes.

- [ ] **Step 5: Commit Task 9**

```powershell
git add src/tarca/stage2/tasks.py src/tarca/stage2/jobs.py src/tarca/stage2/runner.py src/tarca/e02/tasks.py src/tarca/e02/jobs.py src/tarca/e02/runner.py tests/stage2/test_tasks.py tests/stage2/test_runner.py tests/e02/test_tasks.py tests/e02/test_runner.py
git commit -m "feat: add stage2 and e02 execution graphs"
```

### Task 10: Add Two-GPU Admission, CPU Backfill, Packing, DDP, Precision, and ETA Gates

**Files:**
- Create: `src/tarca/stage2/resources.py`
- Create: `tests/stage2/test_resources.py`
- Modify: `src/tarca/execution/resources.py`
- Modify: `src/tarca/execution/scheduler.py`
- Modify: `tests/execution/test_resources.py`
- Modify: `tests/execution/test_scheduler.py`

**Interfaces:**
- Produces: `Stage2ServerInventory`, `Stage2CapacityPlan`, `Stage2ProbeObservation`
- Produces: `stage2_server_admission_check(...) -> None`
- Produces: `choose_stage2_capacity_plan(...) -> Stage2CapacityPlan`
- Produces: `InferenceBundleController.observe(...) -> GpuPackingDecision`
- Existing `select_ddp_mode` retains the exact 30% threshold

- [ ] **Step 1: Write failing tests for exact 28/224/2×24 admission and useful saturation**

```python
def test_stage2_admits_target_server_and_reserves_four_cores() -> None:
    plan = choose_stage2_capacity_plan(TARGET_4090_SERVER, SAFE_PROBES)
    assert plan.work_cpu_cores == 24
    assert plan.scheduler_monitor_cores == 1
    assert plan.system_io_cores == 3
    assert plan.gpu_worker_count == 2
    assert plan.host_memory_ceiling_gib == 200


def test_two_active_gpu_tasks_leave_sixteen_cpu_cores_for_backfill() -> None:
    allocations = plan_stage2_wave(TWO_GPU_TRAIN_TASKS, CPU_BACKFILL_TASKS, TARGET_CAPACITY)
    assert sum(item.cpu_threads for item in allocations) == 24
    assert {item.gpu_ids for item in allocations if item.gpu_ids} == {(0,), (1,)}
```

- [ ] **Step 2: Run resource and scheduler tests to verify the missing behavior**

Run Task 10 tests plus existing execution resource/scheduler tests.

- [ ] **Step 3: Implement bounded policy wrappers and exclusive GPU ownership**

Keep one active OS training task per GPU. Admit two 4-core/32GiB GPU tasks, then fill remaining cores with dependency-ready CPU tasks without exceeding 24 cores or 200GiB. Set the Stage 2 storage floor to 200GiB. Place 2–3 small inference bundles inside one GPU worker using existing 180-second/70%/8GiB and 80%/18GiB rules. Back off on over-20GiB, OOM, throttle, data wait, or throughput loss. Select AMP only when finite, within configured maximum absolute error, and faster than FP32.

- [ ] **Step 4: Run fake-NVML wave, OOM backoff, DDP 29.9/30.0%, and ETA-margin tests**

Expected: 29.9% keeps task parallel; 30.0% selects DDP; ETA plus one hour equal to the rental boundary refuses launch.

- [ ] **Step 5: Commit Task 10**

```powershell
git add src/tarca/stage2/resources.py src/tarca/execution/resources.py src/tarca/execution/scheduler.py tests/stage2/test_resources.py tests/execution/test_resources.py tests/execution/test_scheduler.py
git commit -m "feat: schedule stage2 across two gpus"
```

### Task 11: Implement Runtime Lifecycles, Authorization, Recovery, and CLIs

**Files:**
- Create: `src/tarca/stage2/runtime.py`
- Create: `src/tarca/e02/grant.py`
- Create: `src/tarca/e02/runtime.py`
- Create: `scripts/run_stage2_v1.py`
- Create: `scripts/run_e02_v1.py`
- Create: `tests/stage2/test_runtime.py`
- Create: `tests/stage2/test_cli.py`
- Create: `tests/e02/test_grant.py`
- Create: `tests/e02/test_runtime.py`
- Create: `tests/e02/test_cli.py`

**Interfaces:**
- Produces: `dispatch_stage2_runtime_command(...) -> dict[str, object]`
- Produces: `dispatch_e02_runtime_command(...) -> dict[str, object]`
- Produces: `create_e02_grant(...) -> SealedAccessGrant`
- Produces exact CLI commands from the approved spec

- [ ] **Step 1: Write failing authorization and side-effect-boundary tests**

```python
def test_stage2_launch_requires_exact_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(Stage2RuntimeAuthorizationError):
        launch_stage2(tmp_path, acknowledgement="close")


def test_e02_prepare_does_not_open_formal_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tarca.stage2.data._read_formal_storage", forbidden_call)
    prepare_e02(REPOSITORY_ROOT, E02_CONFIG_PATH, ARTIFACT_ROOT)
```

- [ ] **Step 2: Run runtime/CLI tests and observe missing entrypoints**

Run the five Task 11 test files.

- [ ] **Step 3: Implement atomic receipts and exact acknowledgements**

Stage 2 launch accepts only `I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN`; E02 launch accepts only `I_ACKNOWLEDGE_E02_V1_FORMAL_RUN`. `prepare`, `dry-run`, `status`, bundle creation, and Stage 2 preflight never create a formal grant. `recover` snapshots the latest consistent checkpoints, execution DB, manifests, and log index into a hash-addressed capsule. `resume` uses the same run ID and preserves all attempts.

Expose the exact command sets `prepare`, `dry-run`, `preflight`, `launch`, `resume`, `status`, `freeze/finalize`, and `recover` through argparse and sorted JSON stdout.

- [ ] **Step 4: Run all lifecycle transitions, corrupted-receipt, wrong-token, and resume tests**

Expected: wrong token and missing preflight fail before state mutation; formal open occurs only after a valid grant; recovery/reload preserves run identity.

- [ ] **Step 5: Commit Task 11**

```powershell
git add src/tarca/stage2/runtime.py src/tarca/e02/grant.py src/tarca/e02/runtime.py scripts/run_stage2_v1.py scripts/run_e02_v1.py tests/stage2/test_runtime.py tests/stage2/test_cli.py tests/e02/test_grant.py tests/e02/test_runtime.py tests/e02/test_cli.py
git commit -m "feat: add stage2 and e02 runtime lifecycle"
```

### Task 12: Enforce Read-Only, Science-Blind Monitoring

**Files:**
- Modify: `src/tarca/monitoring/api.py`
- Modify: `src/tarca/monitoring/server.py`
- Modify: `frontend/stage1b-monitor/src/App.tsx`
- Create: `tests/stage2/test_monitor_contract.py`
- Create: `tests/e02/test_monitor_contract.py`

**Interfaces:**
- Consumes: existing execution SQLite state and telemetry
- Produces: Stage 2/E02 runtime labels and resource-only snapshots
- Preserves: no control endpoint and no partial science score fields

- [ ] **Step 1: Write failing API schema and static-bundle tests**

```python
FORBIDDEN_KEYS = {"crps", "nll", "mae", "coverage", "ranking", "best_seed", "skill"}


def test_running_e02_api_never_exposes_partial_science(client: TestClient) -> None:
    payload = client.get("/api/v1/run").json()
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert not any(key in serialized for key in FORBIDDEN_KEYS)
```

- [ ] **Step 2: Run monitor contract tests and capture the RED label/schema gaps**

- [ ] **Step 3: Add allowlisted runtime labels and resource-only serializers**

Support `stage2-v1` and `e02-v1` execution kinds. Serialize task phase, attempt, heartbeat, progress units, ETA, CPU/RAM/GPU/I/O, and alerts only. Do not add POST/PUT/PATCH/DELETE routes. Build labels from compile-time environment variables without embedding local paths.

- [ ] **Step 4: Run monitor tests and existing frontend/API tests**

Expected: Stage1B/E01 labels remain valid; Stage 2/E02 labels render; forbidden science keys never appear during running state.

- [ ] **Step 5: Commit Task 12**

```powershell
git add src/tarca/monitoring frontend/stage1b-monitor/src/App.tsx tests/stage2/test_monitor_contract.py tests/e02/test_monitor_contract.py
git commit -m "feat: monitor stage2 without science leakage"
```

### Task 13: Build the Python 3.10/CUDA Deployment Surface

**Files:**
- Create: `deploy/stage2/Dockerfile`
- Create: `deploy/stage2/compose.yaml`
- Create: `deploy/stage2/entrypoint.sh`
- Create: `deploy/stage2/bootstrap.sh`
- Create: `deploy/stage2/supervisor.sh`
- Create: `deploy/stage2/py310/sitecustomize.py`
- Create: `deploy/stage2/requirements-server.in`
- Create: `deploy/stage2/requirements-server.lock`
- Create: `tests/stage2/test_container_contract.py`
- Create: `tests/stage2/test_server_scripts.py`

**Interfaces:**
- Produces: immutable image based exactly on `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04`
- Produces: `bash deploy/stage2/bootstrap.sh --mode preflight --remaining-rental-hours N`
- Produces: supervisor launch/resume for Stage 2 and E02 with exact tokens supplied by user

- [ ] **Step 1: Write failing static container/security tests**

```python
def test_dockerfile_preserves_base_cuda_torch() -> None:
    text = Path("deploy/stage2/Dockerfile").read_text(encoding="utf-8")
    assert text.startswith("FROM node:20-bookworm-slim AS ui-build")
    assert "FROM pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04" in text
    assert "pip install torch" not in text
    assert "USER tarca" in text
```

- [ ] **Step 2: Run container contract tests and verify missing files**

- [ ] **Step 3: Implement non-root, read-only, offline-compatible deployment**

Use a Node build stage only for static monitoring assets. Runtime uses `PYTHONPATH` instead of installing the Python-3.11-declared project package. Install only hash-locked non-torch dependencies. Compose binds monitor port to `127.0.0.1`, mounts only the artifact directory writable, uses `read_only: true`, `tmpfs /tmp`, `shm_size: 16gb`, and requests all NVIDIA GPUs. Bootstrap runs environment validation, source/import hashes, two-card CUDA/AMP/checkpoint probes, bounded throughput/ETA/storage admission, and stops before training.

- [ ] **Step 4: Run Bash syntax checks, static security tests, and Docker config validation when Docker is available**

Run `bash -n` through the available Git Bash/WSL executable for all scripts. If Docker is unavailable, record `NOT_RUN_LOCAL_NO_DOCKER` in the implementation report and keep `docker compose config` as a mandatory server preflight step.

- [ ] **Step 5: Commit Task 13**

```powershell
git add deploy/stage2 tests/stage2/test_container_contract.py tests/stage2/test_server_scripts.py
git commit -m "feat: add stage2 cuda server deployment"
```

### Task 14: Create the Deterministic Offline Server Bundle

**Files:**
- Create: `scripts/prepare_stage2_v1_server_bundle.py`
- Create: `tests/stage2/test_bundle.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `build_stage2_server_bundle(repository_root: Path, output: Path) -> dict[str, object]`
- Produces: deterministic `tar.gz`, `.sha256`, and receipt JSON
- Bundle contains no secret, absolute user path, formal result, or mutable source checkout

- [ ] **Step 1: Write failing deterministic, tamper, secret, and contents tests**

```python
def test_stage2_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_stage2_server_bundle(REPOSITORY_ROOT, tmp_path / "a.tar.gz")
    second = build_stage2_server_bundle(REPOSITORY_ROOT, tmp_path / "b.tar.gz")
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["formal_tasks_executed"] == 0
```

- [ ] **Step 2: Run bundle tests and verify the missing builder**

- [ ] **Step 3: Implement canonical archive and pre-bundle evidence checks**

Reuse E01 canonical tar metadata: mtime/uid/gid fixed to zero, sorted POSIX names, mode 0644/0755, gzip mtime zero. Include source capsules for DLinear, PatchTST, iTransformer, and Lorenz-96; frozen upstream artifacts; configs; `src`; scripts; deploy; prebuilt frontend; tests/smoke; authority documents; approved design; and implementation report. Verify upstream hashes and Stage 2/E02 config hashes before archive creation. Reject private-key markers, `C:\Users\DELL`, credential-like environment assignments, unexpected symlinks, and artifact paths containing formal predictions or scores.

- [ ] **Step 4: Run bundle twice, extract to two temporary directories, and compare manifests**

Expected: byte-identical hash, complete `SHA256SUMS.json`, correct executable modes, zero formal task count, and no local absolute paths.

- [ ] **Step 5: Commit Task 14**

```powershell
git add scripts/prepare_stage2_v1_server_bundle.py tests/stage2/test_bundle.py .gitignore
git commit -m "feat: package deterministic stage2 server bundle"
```

### Task 15: Produce Local Smoke Evidence and Implementation Documentation

**Files:**
- Create: `docs/research/stage2_e02_local_implementation_report_v1.md`
- Create: `docs/research/stage2_e02_server_handoff_v1.md`
- Create: `tests/stage2/test_documentation_contract.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: test outputs, config hashes, bundle receipt, local hardware facts
- Produces: exact local/server boundary, first-open commands, expected artifacts, recovery procedure

- [ ] **Step 1: Write a failing documentation contract test**

```python
def test_handoff_contains_exact_first_open_boundary() -> None:
    text = Path("docs/research/stage2_e02_server_handoff_v1.md").read_text(encoding="utf-8")
    assert "bash deploy/stage2/bootstrap.sh --mode preflight" in text
    assert "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN" in text
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" in text
    assert "尚未执行完整 Stage 2/E02" in text
```

- [ ] **Step 2: Run the documentation test and verify the missing reports**

- [ ] **Step 3: Write evidence-backed reports without claiming unavailable GPU/Docker validation**

Document local interpreter, CPU-only torch, all test commands and results, exact config/science/bundle hashes, source commits, files produced, local limitations, server minimum/recommended resources, preflight command, launch authorization separation, monitoring URL, recovery commands, 24-hour reset behavior, and the fact that complete training/formal E02 were not run locally.

- [ ] **Step 4: Run documentation test and link checks**

Expected: paths exist, commands match implemented CLI help, hashes match receipts, and no report claims remote success.

- [ ] **Step 5: Commit Task 15**

```powershell
git add docs/research/stage2_e02_local_implementation_report_v1.md docs/research/stage2_e02_server_handoff_v1.md tests/stage2/test_documentation_contract.py README.md
git commit -m "docs: hand off stage2 server runtime"
```

### Task 16: Run Full Verification and Freeze the Local Handoff

**Files:**
- Modify only files required to fix evidence-backed failures
- Update: `docs/research/stage2_e02_local_implementation_report_v1.md` with final command results

**Interfaces:**
- Produces: clean worktree, passing verification, deterministic bundle receipt, no formal access, server-ready handoff

- [ ] **Step 1: Run focused Stage 2/E02 coverage**

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest tests/stage2 tests/e02 --cov=tarca.stage2 --cov=tarca.e02 --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Expected: PASS with branch coverage at least 80%.

- [ ] **Step 2: Run the complete regression suite**

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m pytest -q
```

Expected: all tests PASS; no Stage1B/E01 regression.

- [ ] **Step 3: Run static and security verification**

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m ruff check src tests scripts
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' -m mypy src/tarca
git diff --check
git grep -n -E 'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|DEEPSEEK_API_KEY=|token=[A-Za-z0-9_-]{16,}' -- . ':(exclude)docs/superpowers/plans/*'
```

Expected: Ruff/mypy/diff PASS and secret scan returns no match.

- [ ] **Step 4: Build and verify the final local bundle without running formal science**

```powershell
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/prepare_stage2_v1_server_bundle.py --output artifacts/stage2/server_bundle/tarca-stage2-v1.tar.gz
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/run_stage2_v1.py prepare
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/run_stage2_v1.py dry-run
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/run_e02_v1.py prepare
& 'D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe' scripts/run_e02_v1.py dry-run
```

Expected: bundle receipt reports `formal_tasks_executed: 0`; commands do not create formal data or predictions.

- [ ] **Step 5: Review exact spec-to-code coverage and commit final evidence**

Check every section of the approved design against code/tests/report. Confirm `git status --short`, inspect every diff, and commit only final report corrections:

```powershell
git add docs/research/stage2_e02_local_implementation_report_v1.md
git commit -m "test: verify stage2 server readiness"
```

Expected final state: clean worktree, local verification evidence complete, server bundle present under ignored artifacts, no remote connection, no full Stage 2 training, and no E02 formal access.
