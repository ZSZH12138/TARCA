# Stage1B World Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. TARCA repository policy forbids subagents, so execution is inline only. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an independent, leakage-safe Stage1B qualification pipeline that freezes a world suite only when at least two project-valid scientific families each have a TARCA-operable neural predictor that stably beats tuned VAR.

**Architecture:** Keep completed Stage0/Stage1A code and authority documents immutable. Add a new `tarca.stage1b` package with frozen config contracts, thin adapters over the pinned Interfere generator, qualification-only partitions, fair VAR and small neural candidates, automated promotion gates, and non-destructive versioned freezing. Formal E01/E02 partitions and result seeds are never read or written.

**Tech Stack:** Python 3.11, PyTorch 2.13 CPU, NumPy 1.26.x, Pydantic 2, PyYAML, pytest, mypy, Ruff, pinned Interfere commit `adfa3f730019f17c3554dd7e0c181248f785bb8b`.

**Spec:** `docs/research/stage1b_world_qualification_spec.md`

## Global Constraints

- Authority files under `docs/auth`, completed Stage0/Stage1A Python, and existing frozen artifacts remain byte-for-byte unchanged.
- Do not execute E01 or E02 and do not use their formal seeds, splits, sealed tests, or result identifiers.
- Use `QUAL_TRAIN`, `QUAL_TUNE`, `QUAL_SEEN`, and `QUAL_UNSEEN`; split whole trajectories, never adjacent windows from one trajectory.
- External dynamics come from the pinned MIT-licensed Interfere source; TARCA adds only configuration, replay, delay/path truth, contracts, and orchestration.
- WQ-01 through WQ-11 must pass before any neural score can promote a world.
- Every formal `PRIMARY_MECHANISTIC` world must pass WQ-13; the suite needs at least two independent passing primary families.
- `CONTROL_LINEAR` must allow VAR to win and is excluded from the neural-win aggregate.
- Qualification candidates are small PatchTST and small iTransformer with `d_model=64`, `n_layers=3`, `n_heads=4`, `dropout=0.1`.
- Three qualification seeds must show the same CRPS direction; paired trajectory bootstrap 95% improvement lower bound must exceed zero.
- A failed world, configuration, or seed remains in the evidence ledger; no post-result parameter changes within v1.
- Frozen versions are immutable. User-authorized modification creates a new version and updates an active pointer while retaining all prior versions.
- Before the full training run, execute the smallest representative probe and estimate local runtime and memory. Stop if the required run is unlikely to finish within 120 hours or risks system stability.
- New Python behavior follows RED-GREEN-REFACTOR and new Stage1B code must maintain at least 80% branch coverage.

---

### Task 1: Environment, Source Lock, and Configuration Contracts

**Files:**
- Create: `third_party_manifest/stage1b_sources.yaml`
- Create: `configs/stage1b/worlds_v1.yaml`
- Create: `configs/stage1b/qualification_v1.yaml`
- Create: `src/tarca/stage1b/__init__.py`
- Create: `src/tarca/stage1b/config.py`
- Create: `tests/stage1b/test_config.py`

**Interfaces:**
- Consumes: YAML files and repository root `Path`.
- Produces: `load_world_suite(path: Path) -> WorldSuiteConfig`, `load_qualification_config(path: Path) -> QualificationConfig`, and `verify_source_lock(source_root: Path, expected_commit: str, expected_license_sha256: str) -> SourceLockEvidence`.

- [x] **Step 1: Create a dedicated Conda environment without modifying existing environments**

Run:

```powershell
conda create -n tarca-stage1b-py311 --clone tarca-local-py311 -y
conda run -n tarca-stage1b-py311 python -m pip install "numpy>=1.26,<2" "pytest>=9,<10" "pytest-cov>=7,<8" "mypy>=1.18,<2" "ruff>=0.15,<0.16" "types-PyYAML>=6,<7"
conda run -n tarca-stage1b-py311 python -m pip install "git+https://github.com/djpasseyjr/interfere.git@adfa3f730019f17c3554dd7e0c181248f785bb8b"
```

Expected: Python 3.11, NumPy below 2, PyTorch importable, Interfere 1.0.2 importable, and the original `tarca-local-py311` environment unchanged.

- [x] **Step 2: Write failing configuration and source-lock tests**

```python
def test_world_suite_rejects_primary_without_required_truth(tmp_path: Path) -> None:
    config_path = write_yaml(tmp_path, primary_world_without_shared_noise())
    with pytest.raises(ValueError, match="shared-noise"):
        load_world_suite(config_path)


def test_source_lock_rejects_wrong_commit(interfere_source_root: Path) -> None:
    with pytest.raises(SourceLockError, match="commit"):
        verify_source_lock(interfere_source_root, "0" * 40, INTERFERE_LICENSE_SHA256)
```

- [x] **Step 3: Verify RED**

Run: `python -m pytest tests/stage1b/test_config.py -q`

Expected: collection fails because `tarca.stage1b.config` does not exist.

- [x] **Step 4: Implement immutable config models and exact source verification**

Use frozen dataclasses or Pydantic models with forbidden extra fields. Validate role-specific requirements, unique IDs, positive dimensions, disjoint qualification seed namespaces, explicit graph/lag/regime truth, source commit, license hash, and safe repository-relative paths. Source verification invokes `git -C <source_root> rev-parse HEAD` without shell evaluation and hashes the license bytes.

- [x] **Step 5: Verify GREEN and coverage**

Run:

```powershell
python -m pytest tests/stage1b/test_config.py -q
python -m pytest tests/stage1b/test_config.py --cov=tarca.stage1b.config --cov-branch --cov-report=term-missing
```

Expected: all tests pass and configuration module branch coverage is at least 80%.

- [ ] **Step 6: Commit**

```text
feat: add stage1b source and qualification contracts
```

### Task 2: External World Adapters and Paired Replay

**Files:**
- Create: `src/tarca/stage1b/worlds.py`
- Create: `src/tarca/stage1b/truth.py`
- Create: `tests/stage1b/test_worlds.py`
- Create: `tests/stage1b/test_paired_replay.py`

**Interfaces:**
- Consumes: `WorldConfig`, Interfere installed at the pinned commit, trajectory seed, partition, regime, and optional concept intervention.
- Produces: `ExternalWorldAdapter.simulate(request: SimulationRequest) -> SimulatedTrajectory`, `ExternalWorldAdapter.paired_counterfactual(request: PairedSimulationRequest) -> PairedTrajectory`, and `WorldTruth` containing graph, path-derived lags, regime parameters, concept mapping, and future-noise hash.

- [ ] **Step 1: Write failing adapter tests against the real pinned Interfere package**

```python
@pytest.mark.parametrize("world_id", ["network_cml_v1", "ecology_lv_sde_v1"])
def test_world_replay_is_deterministic(world_suite: WorldSuiteConfig, world_id: str) -> None:
    world = build_world(world_suite.world(world_id))
    first = world.simulate(simulation_request(seed=701))
    second = world.simulate(simulation_request(seed=701))
    torch.testing.assert_close(first.values, second.values, rtol=0.0, atol=0.0)


def test_no_intervention_pair_is_exact_for_same_future_noise(ecology_world: ExternalWorldAdapter) -> None:
    pair = ecology_world.paired_counterfactual(identity_pair_request(seed=702))
    torch.testing.assert_close(pair.factual.values, pair.counterfactual.values, rtol=0.0, atol=0.0)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/stage1b/test_worlds.py tests/stage1b/test_paired_replay.py -q`

Expected: failure because world adapters are missing.

- [ ] **Step 3: Implement thin external adapters**

Implement one external nonlinear coupled-map world and one external stochastic Lotka-Volterra world. Keep upstream equations untouched. Pre-generate and persist Wiener increments for SDE replay; deterministic worlds record an explicit zero-noise artifact. Derive causal lag truth from directed shortest-path length in the fixed graph. Reject non-finite, clipped, silently replaced, or topology-mismatched trajectories.

- [ ] **Step 4: Add intervention-isolation and truth tests**

```python
def test_target_intervention_changes_only_reachable_future_nodes(network_world: ExternalWorldAdapter) -> None:
    pair = network_world.paired_counterfactual(node_shock_request(source_node=0, seed=703))
    assert pair.truth.first_effect_lag[target_node] == pair.truth.shortest_path_lag[target_node]
    assert torch.equal(pair.factual.future_noise, pair.counterfactual.future_noise)


def test_nonfinite_trajectory_fails_closed(ecology_world: ExternalWorldAdapter) -> None:
    with pytest.raises(TrajectoryValidationError, match="non-finite"):
        ecology_world.validate_values(torch.tensor([[float("inf")]]))
```

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/stage1b/test_worlds.py tests/stage1b/test_paired_replay.py -q`

Expected: all real-adapter tests pass.

- [ ] **Step 6: Commit**

```text
feat: add pinned external world adapters and paired replay
```

### Task 3: Qualification-Only Dataset and Leakage-Safe Splits

**Files:**
- Create: `src/tarca/stage1b/dataset.py`
- Create: `src/tarca/stage1b/splits.py`
- Create: `tests/stage1b/test_dataset.py`
- Create: `tests/stage1b/test_qualification_splits.py`

**Interfaces:**
- Consumes: simulated trajectories and `QualificationConfig`.
- Produces: `QualificationDataset`, immutable training-only normalization statistics, and batches for `QUAL_TRAIN`, `QUAL_TUNE`, `QUAL_SEEN`, `QUAL_UNSEEN`.

- [ ] **Step 1: Write failing whole-trajectory isolation tests**

```python
def test_windows_from_one_trajectory_never_cross_partitions() -> None:
    split = build_qualification_split(example_trajectories(), split_config())
    owners = split.partition_by_trajectory_id()
    assert all(len(partitions) == 1 for partitions in owners.values())


def test_normalizer_uses_qual_train_only() -> None:
    dataset = dataset_with_extreme_qual_unseen_values()
    normalized = prepare_dataset(dataset)
    assert normalized.statistics.mean == pytest.approx((1.0, 2.0))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/stage1b/test_dataset.py tests/stage1b/test_qualification_splits.py -q`

Expected: missing dataset/split modules.

- [ ] **Step 3: Implement windowing, immutable normalization, and lineage**

Each sample carries world ID, family ID, regime ID, trajectory ID, time range, horizon group, graph hash, source commit, config hash, and qualification partition. Windows may overlap inside a trajectory but trajectories are indivisible across partitions. No class exposes a `TEST` partition.

- [ ] **Step 4: Verify GREEN and Stage1A boundary compatibility**

Run:

```powershell
python -m pytest tests/stage1b/test_dataset.py tests/stage1b/test_qualification_splits.py -q
python -m pytest tests/stage1a/test_window_batch.py tests/stage1a/test_data_metadata_contracts.py -q
```

Expected: Stage1B tests and unchanged Stage1A contract tests pass.

- [ ] **Step 5: Commit**

```text
feat: add leakage-safe stage1b qualification datasets
```

### Task 4: Fair VAR and TARCA-Operable Neural Predictors

**Files:**
- Create: `src/tarca/stage1b/predictors.py`
- Create: `src/tarca/stage1b/neural.py`
- Create: `src/tarca/stage1b/training.py`
- Create: `tests/stage1b/test_var_predictor.py`
- Create: `tests/stage1b/test_neural_predictors.py`
- Create: `tests/stage1b/test_training_reproducibility.py`

**Interfaces:**
- Consumes: qualification batches and fixed candidate configs.
- Produces: `TunedVAR`, `SmallPatchTST`, `SmallITransformer`, `TrainingReceipt`, and models satisfying `ForecastPredictor`; neural models also satisfy `MechanisticModelAdapter` for predeclared sites.

- [ ] **Step 1: Write failing VAR behavior tests**

```python
def test_var_recovers_known_var1_and_emits_positive_scale() -> None:
    predictor = TunedVAR.fit(known_var1_train(), var_search_space())
    forecast = predictor.predict_distribution(known_var1_batch())
    assert forecast.mean.shape == (4, 8, 3)
    assert bool((forecast.scale > 0).all())
```

- [ ] **Step 2: Verify VAR RED, implement ridge VAR, then verify GREEN**

Run before implementation: `python -m pytest tests/stage1b/test_var_predictor.py -q`

Implement lag-order and ridge selection using `QUAL_TUNE`, recursive forecasts, horizon-specific residual scale fitted without qualification evaluation partitions, and `ForecastDistribution` validation.

Run after implementation: `python -m pytest tests/stage1b/test_var_predictor.py -q`

- [ ] **Step 3: Write failing neural contract and intervention tests**

```python
@pytest.mark.parametrize("model_factory", [small_patchtst, small_itransformer])
def test_neural_predictor_exposes_stable_intervention_sites(model_factory: ModelFactory) -> None:
    model = model_factory()
    sites = model.list_intervention_sites()
    assert sites
    assert len({site.site_name for site in sites}) == len(sites)


def test_source_swap_changes_forecast_without_mutating_frozen_weights() -> None:
    model = trained_and_frozen_patchtst()
    before = parameter_hash(model)
    result = model.intervene(base_batch(), source_batch(), approved_swap_spec())
    assert parameter_hash(model) == before
    assert not torch.equal(result.mean, model.predict_distribution(base_batch()).mean)
```

- [ ] **Step 4: Verify neural RED, implement minimal models, then verify GREEN**

Run before implementation: `python -m pytest tests/stage1b/test_neural_predictors.py -q`

Implement the predeclared architectures, diagonal Gaussian heads, strictly positive scale, deterministic seed control, stable named capture sites, immutable source/base swap, frozen inference, and model hashing.

Run after implementation: `python -m pytest tests/stage1b/test_neural_predictors.py -q`

- [ ] **Step 5: Add and pass reproducibility tests**

```python
def test_same_seed_produces_same_receipt_and_predictions() -> None:
    first = train_candidate(tiny_training_problem(), seed=901)
    second = train_candidate(tiny_training_problem(), seed=901)
    assert first.receipt == second.receipt
    torch.testing.assert_close(first.prediction.mean, second.prediction.mean, rtol=0.0, atol=0.0)
```

Run: `python -m pytest tests/stage1b/test_training_reproducibility.py -q`

- [ ] **Step 6: Commit**

```text
feat: add fair var and operable neural qualifiers
```

### Task 5: Metrics, Bootstrap, and Fail-Closed WQ-13 Gate

**Files:**
- Create: `src/tarca/stage1b/metrics.py`
- Create: `src/tarca/stage1b/gates.py`
- Create: `tests/stage1b/test_metrics.py`
- Create: `tests/stage1b/test_gates.py`

**Interfaces:**
- Consumes: predictions and targets grouped by trajectory, seed, regime, and horizon group.
- Produces: `MetricBundle`, `BootstrapInterval`, `WorldGateDecision`, and `SuiteGateDecision`.

- [ ] **Step 1: Write failing hand-derived Gaussian metric tests**

```python
def test_standard_normal_crps_at_mean_matches_closed_form() -> None:
    value = gaussian_crps(mean=0.0, scale=1.0, target=0.0)
    assert value == pytest.approx((2.0**0.5 - 1.0) / math.pi**0.5)


def test_gaussian_nll_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(torch.zeros(1), torch.zeros(1), torch.zeros(1))
```

- [ ] **Step 2: Verify metrics RED, implement metrics, verify GREEN**

Run before: `python -m pytest tests/stage1b/test_metrics.py -q`

Run after: `python -m pytest tests/stage1b/test_metrics.py -q`

- [ ] **Step 3: Write failing promotion-gate tests**

```python
def test_world_gate_fails_if_one_seed_loses_to_var() -> None:
    evidence = evidence_with_seed_improvements((0.08, 0.04, -0.01))
    decision = evaluate_world_gate(evidence, gate_config())
    assert decision.status == "FAIL"
    assert "seed_direction" in decision.failed_checks


def test_suite_gate_requires_two_independent_primary_families() -> None:
    decision = evaluate_suite_gate(two_worlds_same_family(), suite_gate_config())
    assert decision.status == "FAIL"
```

- [ ] **Step 4: Verify gate RED, implement paired trajectory bootstrap and gates, verify GREEN**

Run before: `python -m pytest tests/stage1b/test_gates.py -q`

Implement all-seed CRPS direction, deterministic paired trajectory bootstrap, horizon consistency, NLL/MAE/worst-regime guardrails, finite probabilistic outputs, model-operability evidence, world role exemptions, and two-family suite aggregation.

Run after: `python -m pytest tests/stage1b/test_gates.py -q`

- [ ] **Step 5: Commit**

```text
feat: automate stage1b neural headroom gates
```

### Task 6: Qualification Runner, Hardware Gate, and Versioned Freeze

**Files:**
- Create: `src/tarca/stage1b/hardware.py`
- Create: `src/tarca/stage1b/runner.py`
- Create: `src/tarca/stage1b/freeze.py`
- Create: `scripts/run_stage1b_qualification.py`
- Create: `scripts/check_stage1b.py`
- Create: `tests/stage1b/test_hardware_gate.py`
- Create: `tests/stage1b/test_runner_integration.py`
- Create: `tests/stage1b/test_freeze.py`
- Create: `tests/stage1b/test_stage1b_cli.py`

**Interfaces:**
- Consumes: source root, approved configs, runtime artifact directory, and explicit command (`probe`, `qualify`, `freeze`, `verify`).
- Produces: hardware receipt, immutable run receipts, complete failure ledger, qualification decision, frozen version manifest, and active pointer.

- [ ] **Step 1: Write failing hardware extrapolation tests**

```python
def test_hardware_gate_blocks_estimate_over_120_hours() -> None:
    decision = estimate_full_run(probe_seconds=120.0, probe_work_units=1, full_work_units=4000)
    assert not decision.feasible
    assert decision.estimated_hours > 120.0
```

- [ ] **Step 2: Verify RED, implement hardware receipts, verify GREEN**

Run before: `python -m pytest tests/stage1b/test_hardware_gate.py -q`

Run after: `python -m pytest tests/stage1b/test_hardware_gate.py -q`

- [ ] **Step 3: Write failing runner and freeze tests**

```python
def test_runner_never_exposes_formal_test_partition(tmp_path: Path) -> None:
    receipt = run_tiny_qualification(tiny_approved_config(), tmp_path)
    assert "TEST" not in receipt.partition_names
    assert "E02" not in receipt.experiment_ids


def test_freeze_fails_when_suite_gate_does_not_pass(tmp_path: Path) -> None:
    with pytest.raises(FreezeRejected, match="suite gate"):
        freeze_suite(failing_qualification_receipt(), tmp_path, version="v1")


def test_authorized_override_keeps_v1_and_moves_active_pointer(tmp_path: Path) -> None:
    freeze_suite(passing_receipt_v1(), tmp_path, version="v1")
    freeze_suite(passing_receipt_v2(), tmp_path, version="v2", authorization=authorization())
    assert (tmp_path / "versions/v1/manifest.json").is_file()
    assert load_active_pointer(tmp_path).version == "v2"
```

- [ ] **Step 4: Verify RED, implement orchestration and fail-closed freezing, verify GREEN**

Run before:

```powershell
python -m pytest tests/stage1b/test_runner_integration.py tests/stage1b/test_freeze.py tests/stage1b/test_stage1b_cli.py -q
```

The runner writes checkpoints and generated trajectories only under ignored `artifacts/stage1b/runtime/`; small JSON evidence uses canonical serialization and SHA-256. Freeze refuses missing evidence, failed gates, source drift, config drift, prior-version overwrite without authorization, or any formal E01/E02 identifier.

Run after:

```powershell
python -m pytest tests/stage1b/test_runner_integration.py tests/stage1b/test_freeze.py tests/stage1b/test_stage1b_cli.py -q
```

- [ ] **Step 5: Commit**

```text
feat: add stage1b qualification and versioned freeze runner
```

### Task 7: Execute Probe, Qualification, and Conditional Freeze

**Files:**
- Generate ignored runtime: `artifacts/stage1b/runtime/**`
- Generate if evidence passes: `artifacts/stage1b/versions/v1/*.json`
- Generate if evidence passes: `artifacts/stage1b/active.json`
- Generate regardless of pass/fail: `artifacts/stage1b/qualification_v1_summary.json`

**Interfaces:**
- Consumes: all implementation from Tasks 1-6 and pinned Interfere checkout at `data/third_party/interfere`.
- Produces: empirical qualification evidence or an explicit non-freeze decision.

- [ ] **Step 1: Materialize the exact upstream checkout without modifying it**

Run:

```powershell
git clone --no-checkout https://github.com/djpasseyjr/interfere.git data/third_party/interfere
git -C data/third_party/interfere checkout --detach adfa3f730019f17c3554dd7e0c181248f785bb8b
```

Expected: detached exact commit and matching MIT license hash.

- [ ] **Step 2: Run the smallest representative hardware probe**

Run:

```powershell
python scripts/run_stage1b_qualification.py probe --worlds configs/stage1b/worlds_v1.yaml --qualification configs/stage1b/qualification_v1.yaml --source-root data/third_party/interfere
```

Expected: a receipt containing measured seconds, peak working set, projected full-run hours, CPU/RAM/GPU inventory, and `feasible=true|false`.

- [ ] **Step 3: Apply the hardware feasibility gate**

If `feasible=false`, stop before full training and report measured evidence plus minimum/recommended server specifications. Do not shrink the frozen workload.

If `feasible=true`, continue with the exact frozen workload.

- [ ] **Step 4: Run the full independent qualification**

Run:

```powershell
python scripts/run_stage1b_qualification.py qualify --worlds configs/stage1b/worlds_v1.yaml --qualification configs/stage1b/qualification_v1.yaml --source-root data/third_party/interfere
```

Expected: all configured worlds and seeds are reported, including failures; no E01/E02 artifact is created.

- [ ] **Step 5: Freeze only on automated PASS**

Run:

```powershell
python scripts/run_stage1b_qualification.py freeze --receipt artifacts/stage1b/qualification_v1_summary.json --version v1
python scripts/check_stage1b.py --version v1
```

Expected on suite PASS: immutable v1 manifest and active pointer verify successfully.

Expected on suite FAIL: freeze command exits non-zero, no active pointer is created, and the failure ledger remains available.

- [ ] **Step 6: Commit only small evidence, never runtime data or checkpoints**

```text
exp: record stage1b world qualification evidence
```

### Task 8: Documentation, Full Verification, and Boundary Audit

**Files:**
- Create: `docs/research/stage1b_world_qualification_report_v1.md`
- Modify: `docs/research/stage1b_candidate_world_report_draft.md`
- Modify: `docs/research/stage1b_world_sources_draft.yaml`
- Modify: `docs/superpowers/plans/2026-08-22-stage1b-world-qualification-implementation.md`

**Interfaces:**
- Consumes: actual receipts, gate decisions, git diff, and test outputs.
- Produces: evidence-backed human report with no claim stronger than the automated result.

- [ ] **Step 1: Write the report from actual artifacts**

Report source commit, license, worlds, concepts, graph/lag truth, split counts, hardware estimate, model budgets, per-seed metrics, bootstrap intervals, failed checks, downstream operation smoke, freeze status, and exact reason for any non-freeze result. Do not label a conditional or failed world as frozen.

- [ ] **Step 2: Run unit, integration, and CLI E2E tests with coverage**

Run:

```powershell
python -m pytest tests/stage1b -q --cov=tarca.stage1b --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Expected: zero failed/skipped tests and at least 80% branch-aware coverage for new Stage1B code.

- [ ] **Step 3: Run the full repository verification suite**

Run:

```powershell
python -m pytest -q
ruff check src tests scripts
mypy src
python scripts/check_stage0.py
python scripts/check_stage1a.py
python scripts/check_stage1b.py --allow-unfrozen
```

Expected: all existing tests and Stage0/Stage1A checks pass; Stage1B check truthfully reports frozen PASS or qualified-but-unfrozen FAIL.

- [ ] **Step 4: Audit frozen boundaries and ignored runtime data**

Run:

```powershell
git diff --exit-code b4616d2 -- docs/auth artifacts/stage0 src/tarca/contracts src/tarca/data src/tarca/artifacts src/tarca/stage0 tests/stage0 tests/stage1a
git status --short
git check-ignore artifacts/stage1b/runtime data/third_party/interfere
```

Expected: no completed-stage boundary changes; runtime data and external checkout are ignored; the user's pre-existing untracked authority snapshot remains untouched.

- [ ] **Step 5: Review requirements line by line and commit documentation**

```text
docs: report stage1b world qualification outcome
```

## Plan Self-Review

- Spec coverage: WQ-01 through WQ-13, two-family aggregation, control exemption, sealed E02 separation, hardware gate, downstream operability, failure ledger, and versioned override all map to Tasks 1-8.
- Placeholder scan: no `TBD`, `TODO`, “implement later,” or unspecified test step remains.
- Type consistency: configs flow into world adapters and datasets; predictors emit existing `ForecastDistribution`; metric evidence flows into gates; gates flow into runner and freeze; freeze consumes only a passing receipt.
- Repository boundary: all new production code is under `src/tarca/stage1b`; no completed Stage0/Stage1A implementation file is scheduled for modification.
