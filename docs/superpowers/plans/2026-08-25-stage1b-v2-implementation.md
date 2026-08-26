# Stage1B v2 Official Runtime Implementation Plan

> **For inline agentic execution:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task with review checkpoints. TARCA's
> project instructions prohibit subagents, so every step must be executed and
> reviewed inline by the primary agent.

**Goal:** Replace the active Stage1B v2 approximations with pinned official
world/model integrations, a generator-owned oracle, a dual-RTX-4090
science-blind scheduler, and a read-only monitoring dashboard without running
the full qualification, E01, or E02.

**Architecture:** Keep the frozen Stage1A contracts as the only science bridge.
Stage1B materializes official sources into a hash-verified cache, separates
official reproduction from TARCA oracle qualification, and compiles immutable
scientific jobs into resource-only execution plans. Workers publish verified
artifacts and resource-only telemetry to SQLite; FastAPI and React expose a
read-only dashboard that cannot see partial scientific metrics or truth.

**Tech Stack:** Python 3.10/3.11, PyTorch 2.2.2 + CUDA 12.1, Pydantic, NumPy,
PyArrow, psutil, NVIDIA NVML, SQLite WAL, FastAPI, Uvicorn, React, TypeScript,
Vite, ECharts, Vitest, Playwright, Docker.

**Spec:** `docs/superpowers/specs/2026-08-25-stage1b-v2-design.md`

## Global Constraints

- Execute inline and single-agent; do not dispatch or use subagents.
- Preserve every file under `docs/auth` byte-for-byte.
- Preserve the read-only Stage1A sources, `pyproject.toml`, and `uv.lock`; the
  Python 3.10 server uses an isolated compatibility/bootstrap layer.
- Use `D:\software\MyAnaconda`; do not install new packages into an existing
  Conda environment.
- Create the isolated local environment
  `tarca-stage1b-runtime-py310` for new runtime tests.
- Runtime image is exactly
  `pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04`.
- Do not let dependency installation replace the image's PyTorch 2.2.2 CUDA
  build with a CPU wheel.
- Keep `worlds_v2.yaml`, `qualification_v2.yaml`, schema `2.0.0`, and the
  Stage1B v2 scientific series; do not create v3.
- Record license status and the user's direct-use authorization, but do not use
  license uncertainty as an implementation blocker.
- Keep official source/data bytes unchanged and bind URL, commit, hash, and
  retrieval date in receipts.
- Keep `OFFICIAL_REPRODUCTION` and `TARCA_ORACLE_QUALIFICATION` data,
  partitions, artifacts, and receipts physically separate.
- Keep the v2 gate unchanged: 40 comparison units, CRPS win rate 65%, positive
  skill, seen/unseen majority, bootstrap and guardrail checks.
- Do not read or generate reserved formal seeds.
- Do not run the full Stage1B qualification, E01, or E02 during implementation.
- Scheduler decisions may use resources, throughput, heartbeat, and task state
  only; they may not use partial NLL/CRPS/MAE, model ranking, seed ranking, or
  sealed truth.
- The dashboard is read-only and must not expose partial scientific metrics,
  truth, model-selection advice, arbitrary files, secrets, or Docker control.
- CPU/GPU probes must stay representative and bounded; never reduce scientific
  seeds, epochs, horizons, or comparison units to meet the 24-hour target.
- Every behavior change follows RED → GREEN → refactor and ends in a focused
  conventional commit.
- Python branch coverage and frontend coverage must each remain at least 80%.

## File Structure

| Area | Files | Responsibility |
|---|---|---|
| Server bootstrap | `deploy/stage1b/*` | Python 3.10 compatibility, locked server dependencies, Docker runtime and entrypoint |
| Official sources | `src/tarca/stage1b/sources.py`, `scripts/materialize_stage1b_sources.py` | Allowlisted checkout/download, hash verification, immutable source receipts |
| Official reproduction | `src/tarca/stage1b/reproduction.py`, `configs/stage1b/official_reproduction_v2.yaml` | Upstream generator/model parity without TARCA qualification |
| Oracle bridge | `src/tarca/stage1b/oracle_contracts.py`, `oracle.py`, `persistence.py` | SyntheticConfig, SCM truth, concept schedules, paired replay, verified artifacts |
| World drivers | `src/tarca/stage1b/official_worlds.py`, `worlds.py`, `truth.py`, `dataset.py` | Direct official generator execution and Stage1A partition mapping |
| Model adapters | `src/tarca/stage1b/modeling/*`, `neural.py`, `training.py` | Official PatchTST/iTransformer backbones, probabilistic head, operable sites, CUDA training |
| Execution plane | `src/tarca/execution/*` | Protocol types, ready-manifest compilation, state, workers, resource planning, scheduler |
| Stage1B job graph | `src/tarca/stage1b/compiler.py`, `jobs.py`, `runner.py` | Scientific DAG, allowlisted task executors, aggregation and receipts |
| Monitoring API | `src/tarca/monitoring/*` | Safe read models, SQLite queries, REST/WebSocket application |
| Monitoring UI | `frontend/stage1b-monitor/*` | React dashboard, charts, job table, alerts and frontend tests |
| User entrypoints | `scripts/run_stage1b_runtime.py`, `scripts/run_stage1b_qualification.py` | Preflight, launch/resume/status, qualification receipt and freeze commands |
| Verification | `tests/execution/*`, `tests/monitoring/*`, `tests/stage1b/*`, `docs/research/*` | Unit, integration, E2E, security, runtime and handoff evidence |

---

### Task 0: Seal the Existing v2 Baseline Before New Runtime Work

**Files:**
- Verify: current tracked Stage1B v2 changes shown by `git status`
- Preserve: `docs/auth/*`
- Commit: existing v2 configs, Stage1B source/tests, v1 historical report, and
  v2 build/spec documents

**Interfaces:**
- Consumes: current dirty v2 worktree on
  `codex/stage1b-v2-official-runtime`
- Produces: one verified baseline commit that later tasks can diff against

- [ ] **Step 1: Confirm branch and rollback refs**

```powershell
git branch --show-current
git show-ref --verify refs/heads/codex/stage1b-v2-pre-official-runtime
git show-ref --verify refs/heads/codex/stage1b-v2-official-runtime
```

Expected: active branch is `codex/stage1b-v2-official-runtime`; both refs exist.

- [ ] **Step 2: Verify the current v2 baseline**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-py311 python -m pytest tests/stage1b tests/stage1a -q
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-py311 python -m ruff check src tests scripts
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-py311 python -m mypy src/tarca/stage1b
git diff --check
```

Expected: all commands pass. Do not run `qualify`, E01, or E02.

- [ ] **Step 3: Audit and stage the exact baseline**

```powershell
git diff --name-status
git status --short docs/auth
git diff -- docs/auth
git add .gitignore configs/stage1b docs/research scripts/run_stage1b_qualification.py src/tarca/stage1b tests/stage1b third_party_manifest
git add -u artifacts/stage1b docs/superpowers/plans/2026-08-22-stage1b-world-qualification-implementation.md
git diff --cached --check
git diff --cached --name-status
```

Expected: no file under `docs/auth` is staged. Leave the existing untracked
Stage0 handoff snapshot untouched.

- [ ] **Step 4: Commit the verified baseline**

```powershell
git commit -m "feat: establish stage1b v2 qualification baseline"
```

---

### Task 1: Python 3.10 and CUDA Runtime Contract

**Files:**
- Create: `deploy/stage1b/py310/sitecustomize.py`
- Create: `deploy/stage1b/requirements-server.in`
- Create: `deploy/stage1b/requirements-server.lock`
- Create: `deploy/stage1b/requirements-test.in`
- Create: `src/tarca/stage1b/server_environment.py`
- Create: `tests/stage1b/test_server_environment.py`

**Interfaces:**
- Produces:
  `validate_server_environment(expectation: ServerEnvironmentExpectation) -> ServerEnvironmentReceipt`

- [ ] **Step 1: Create an isolated Python 3.10 test environment**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' create -n tarca-stage1b-runtime-py310 python=3.10 pip -y
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

- [ ] **Step 2: Write failing compatibility and environment tests**

```python
def test_py310_bootstrap_supplies_frozen_stdlib_names() -> None:
    completed = run_py310(
        "from enum import StrEnum; from typing import Self; import tomllib; "
        "assert str(StrEnum('Probe', {'OK': 'OK'}).OK) == 'OK'"
    )
    assert completed.returncode == 0


def test_server_environment_rejects_cpu_torch(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    expectation = ServerEnvironmentExpectation(
        python_minor=(3, 10),
        torch_version="2.2.2",
        cuda_version="12.1",
        gpu_count=2,
        gpu_name_substring="RTX 4090",
        minimum_vram_bytes=24 * 1024**3,
        minimum_cpu_count=28,
        minimum_ram_bytes=224 * 1024**3,
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_server_environment(expectation)
```

- [ ] **Step 3: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_server_environment.py -q
```

Expected: FAIL because the compatibility bootstrap and environment contract do
not exist.

- [ ] **Step 4: Implement the compatibility bootstrap**

```python
import enum
import sys
import typing

import tomli
from typing_extensions import Self


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return str(self.value)

    setattr(enum, "StrEnum", StrEnum)

if not hasattr(typing, "Self"):
    setattr(typing, "Self", Self)

sys.modules.setdefault("tomllib", tomli)
```

Set `PYTHONPATH` in this order:
`/opt/tarca/deploy/stage1b/py310:/opt/tarca/src`.

- [ ] **Step 5: Implement immutable runtime expectations and receipts**

```python
@dataclass(frozen=True, slots=True)
class ServerEnvironmentExpectation:
    python_minor: tuple[int, int]
    torch_version: str
    cuda_version: str
    gpu_count: int
    gpu_name_substring: str
    minimum_vram_bytes: int
    minimum_cpu_count: int
    minimum_ram_bytes: int


@dataclass(frozen=True, slots=True)
class ServerEnvironmentReceipt:
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]
    cpu_count: int
    ram_bytes: int
    cuda_probe_passed: bool
```

Validate exact Python/PyTorch/CUDA versions, two visible 4090s, per-card VRAM,
CPU and RAM, then run one forward/backward/autocast/save/reload probe per GPU.

- [ ] **Step 6: Lock server dependencies without Torch**

`requirements-server.in` contains:

```text
numpy==1.26.4
pydantic>=2.12,<3
PyYAML>=6.0,<7
pyarrow>=25,<26
psutil>=7,<8
typing-extensions>=4.12,<5
tomli>=2.2,<3
fastapi>=0.116,<1
uvicorn[standard]>=0.35,<1
nvidia-ml-py>=13.580,<14
scipy>=1.11,<2
pandas>=2.1,<3
scikit-learn>=1.3,<2
einops>=0.7,<1
```

Generate the hash lock inside the fixed Linux image. Reject a lock containing
`torch`, `torchvision`, or `torchaudio`, then assert the installed image Torch
is still 2.2.2 with CUDA 12.1.

`requirements-test.in` extends the server input with pytest, pytest-cov, mypy,
Ruff, types-PyYAML, HTTPX and pip-audit. Install it only into the new
`tarca-stage1b-runtime-py310` environment.

```powershell
docker run --rm -v "${PWD}:/work" -w /work pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04 bash -lc "python -m pip install pip-tools && pip-compile --generate-hashes --output-file deploy/stage1b/requirements-server.lock deploy/stage1b/requirements-server.in"
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pip install -r deploy/stage1b/requirements-test.in
if (Select-String -LiteralPath deploy/stage1b/requirements-server.lock -Pattern '^(torch|torchvision|torchaudio)==') { throw 'server lock must not replace image torch' }
```

- [ ] **Step 7: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_server_environment.py -q
git diff --check
git add deploy/stage1b src/tarca/stage1b/server_environment.py tests/stage1b/test_server_environment.py
git commit -m "feat: add stage1b py310 cuda runtime contract"
```

---

### Task 2: Pinned Official Source Materialization

**Files:**
- Modify: `src/tarca/stage1b/config.py`
- Create: `src/tarca/stage1b/sources.py`
- Create: `scripts/materialize_stage1b_sources.py`
- Modify: `configs/stage1b/worlds_v2.yaml`
- Modify: `third_party_manifest/stage1b_sources_v2.yaml`
- Modify: `.gitignore`
- Modify: `tests/stage1b/test_config.py`
- Create: `tests/stage1b/test_sources.py`

**Interfaces:**
- Produces:
  `materialize_source(source: SourceConfig, cache_root: Path, runner: GitRunner) -> SourceMaterializationReceipt`
- Produces:
  `verify_materialized_source(receipt, cache_root) -> Path`

- [ ] **Step 1: Write failing policy and drift tests**

```python
def test_user_authorized_direct_source_is_valid() -> None:
    source = SourceConfig.model_validate(
        source_payload(
            license_id="UNDECLARED",
            code_usage="DIRECT_OFFICIAL_CODE_AND_DATA",
            authorization_policy="USER_AUTHORIZED_NO_LICENSE_BLOCK",
        )
    )
    assert source.code_usage is SourceCodeUsage.DIRECT_OFFICIAL_CODE_AND_DATA


def test_materializer_rejects_checkout_hash_drift(tmp_path: Path, fake_git: FakeGit) -> None:
    receipt = materialize_source(source_config(), tmp_path, fake_git)
    (receipt.checkout_root / "synthetic.py").write_text("changed", encoding="utf-8")
    with pytest.raises(SourceVerificationError, match="hash"):
        verify_materialized_source(receipt, tmp_path)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_config.py tests/stage1b/test_sources.py -q
```

- [ ] **Step 3: Extend immutable source configuration**

```python
class SourceCodeUsage(StrEnum):
    DIRECT_OFFICIAL_CODE = "DIRECT_OFFICIAL_CODE"
    DIRECT_OFFICIAL_DATA = "DIRECT_OFFICIAL_DATA"
    DIRECT_OFFICIAL_CODE_AND_DATA = "DIRECT_OFFICIAL_CODE_AND_DATA"
    REIMPLEMENTED_EQUATIONS = "REIMPLEMENTED_EQUATIONS"


class SourceAuthorizationPolicy(StrEnum):
    LICENSED = "LICENSED"
    USER_AUTHORIZED_NO_LICENSE_BLOCK = "USER_AUTHORIZED_NO_LICENSE_BLOCK"


class SourceAssetConfig(FrozenModel):
    asset_id: str
    relative_path: str
    sha256: str
    required_for: tuple[Literal["REPRODUCTION", "ORACLE", "MODEL"], ...]
```

Allow a nonblank license string, add authorization policy/ID and assets, and
reject traversal, duplicates, invalid hashes, and unauthorized direct use.

- [ ] **Step 4: Implement allowlisted materialization**

```python
class GitRunner(Protocol):
    def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SourceMaterializationReceipt:
    source_id: str
    repository_url: str
    commit: str
    checkout_root: Path
    tree_sha256: str
    asset_sha256: tuple[tuple[str, str], ...]
    authorization_id: str
    materialized_at_utc: datetime


@dataclass(frozen=True, slots=True)
class MaterializedSources:
    receipts: tuple[SourceMaterializationReceipt, ...]

    @classmethod
    def empty(cls) -> Self:
        return cls(receipts=())

    def root(self, source_id: str) -> Path:
        matches = tuple(item.checkout_root for item in self.receipts if item.source_id == source_id)
        if len(matches) != 1:
            raise KeyError(source_id)
        return matches[0]
```

Use `subprocess.run([git_executable, *arguments], shell=False, check=True)`,
detach at the exact
commit, hash required assets, and atomically publish to
`third_party/stage1b/<source_id>/<commit>/`. Add `/third_party/stage1b/` to
`.gitignore`.

- [ ] **Step 5: Update v2 manifests**

Use direct official code/data policies for all six sources. Where license is
absent or unclear, record:

```yaml
authorization_policy: USER_AUTHORIZED_NO_LICENSE_BLOCK
authorization_id: stage1b-v2-user-direct-official-use-2026-08-26
```

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_config.py tests/stage1b/test_sources.py -q
git add .gitignore configs/stage1b/worlds_v2.yaml third_party_manifest/stage1b_sources_v2.yaml src/tarca/stage1b/config.py src/tarca/stage1b/sources.py scripts/materialize_stage1b_sources.py tests/stage1b/test_config.py tests/stage1b/test_sources.py
git commit -m "feat: materialize pinned stage1b official sources"
```

---

### Task 3: Official Reproduction Channel

**Files:**
- Create: `configs/stage1b/official_reproduction_v2.yaml`
- Create: `src/tarca/stage1b/reproduction.py`
- Create: `tests/stage1b/test_reproduction.py`
- Create: `tests/stage1b/test_official_source_integration.py`

**Interfaces:**
- Produces:
  `run_reproduction(spec: ReproductionSpec, sources: MaterializedSources) -> ReproductionReceipt`

- [ ] **Step 1: Write failing isolation and parity tests**

```python
def test_reproduction_receipt_has_no_qualification_identity() -> None:
    receipt = run_reproduction(fake_reproduction_spec(), fake_sources())
    payload = receipt.model_dump(mode="json")
    assert "qualification_id" not in payload
    assert "QUAL_UNSEEN" not in json.dumps(payload)


@pytest.mark.official_source
def test_patchtst_wrapper_mean_matches_pinned_upstream() -> None:
    result = compare_official_model_output("patchtst", fixed_window())
    assert result.maximum_absolute_error <= 1e-6
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_reproduction.py -q
```

- [ ] **Step 3: Implement reproduction contracts**

```python
class ReproductionKind(StrEnum):
    GENERATOR = "GENERATOR"
    MODEL_FORWARD = "MODEL_FORWARD"


class ReproductionSpec(FrozenModel):
    schema_version: Literal["2.0.0"]
    case_id: str
    kind: ReproductionKind
    source_id: str
    source_commit: str
    asset_id: str
    adapter_key: str
    input_artifact: ArtifactRef
    absolute_tolerance: float


class ReproductionReceipt(FrozenModel):
    schema_version: Literal["2.0.0"]
    channel: Literal["OFFICIAL_REPRODUCTION"]
    case_id: str
    source_commit: str
    input_sha256: str
    upstream_output_sha256: str
    adapter_output_sha256: str
    maximum_absolute_error: float
    passed: bool
```

Register Neural-GC L96, GVAR predator-prey data, JMLR two-scale L96,
Interfere CML, PatchTST forward and iTransformer forward. Use tolerance
`1e-10` for float64 worlds and `1e-6` for float32 models. Reject unregistered
assets and never write qualification artifacts.

- [ ] **Step 4: Run unit and pinned-source integration tests**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_reproduction.py -q
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_official_source_integration.py -m official_source -q
```

- [ ] **Step 5: Commit**

```powershell
git add configs/stage1b/official_reproduction_v2.yaml src/tarca/stage1b/reproduction.py tests/stage1b/test_reproduction.py tests/stage1b/test_official_source_integration.py
git commit -m "feat: add isolated official reproduction channel"
```

---

### Task 4: Generator-Owned Oracle and Stage1A Partition Bridge

**Files:**
- Create: `src/tarca/stage1b/oracle_contracts.py`
- Create: `src/tarca/stage1b/oracle.py`
- Create: `src/tarca/stage1b/persistence.py`
- Modify: `src/tarca/stage1b/dataset.py`
- Modify: `src/tarca/stage1b/truth.py`
- Create: `tests/stage1b/test_oracle_contracts.py`
- Create: `tests/stage1b/test_partition_bridge.py`
- Modify: `tests/stage1b/test_paired_replay.py`

**Interfaces:**
- Produces:
  `build_scm_truth_manifest(dataset, store) -> SCMTruthManifest`
- Produces:
  `partition_for_qualification(partition) -> DatasetWindowPartition`
- Produces:
  `paired_rollout(driver, request) -> PairedTrajectory`

- [ ] **Step 1: Write failing contract and partition tests**

```python
@pytest.mark.parametrize(
    ("qualification", "physical"),
    [
        (QualificationPartition.QUAL_TRAIN, DatasetWindowPartition.TRAIN),
        (QualificationPartition.QUAL_TUNE, DatasetWindowPartition.VALIDATION),
        (QualificationPartition.QUAL_SEEN, DatasetWindowPartition.TEST_SEEN_REGIME),
        (QualificationPartition.QUAL_UNSEEN, DatasetWindowPartition.TEST_UNSEEN_REGIME),
    ],
)
def test_fixed_partition_mapping(qualification, physical) -> None:
    assert partition_for_qualification(qualification) is physical


def test_truth_manifest_is_rejected_from_window_metadata() -> None:
    with pytest.raises(ValueError, match="truth"):
        validate_qualification_window(window_batch(metadata={"scm_truth": "hidden"}))
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_oracle_contracts.py tests/stage1b/test_partition_bridge.py tests/stage1b/test_paired_replay.py -q
```

- [ ] **Step 3: Implement authority-shaped truth contracts**

```python
class SyntheticConfig(StrictContractModel):
    name: str
    D: int
    L: int
    H: int
    regimes: int
    true_delay: int | tuple[int, ...]
    root_seed: int
    burn_in: int
    total_steps: int
    generation_settings: Mapping[str, JSONValue]
    normalization_settings: Mapping[str, JSONValue]


class SCMTruthManifest(StrictContractModel):
    schema_version: str
    dataset_hash: Sha256Hash
    generator_config_hash: Sha256Hash
    concept_names: tuple[str, ...]
    regime_ids: tuple[str, ...]
    true_lags: Mapping[str, tuple[int, ...]]
    true_graph_ref: ArtifactRef
    latent_concepts_ref: ArtifactRef
    regime_sequence_ref: ArtifactRef
    exogenous_noise_ref: ArtifactRef
    shock_sequence_ref: ArtifactRef | None
    oracle_protocol_hash: Sha256Hash
    sealed: bool
```

Require unique concepts/regimes, valid hashes and `sealed=True`.

- [ ] **Step 4: Implement concept schedules and paired replay**

```python
@dataclass(frozen=True, slots=True)
class ConceptSchedule:
    trend: Tensor
    scale: Tensor


@dataclass(frozen=True, slots=True)
class OraclePairRequest:
    base: SimulationRequest
    factual_schedule: ConceptSchedule
    counterfactual_schedule: ConceptSchedule
    changed_concept: Literal["trend", "scale", "identity"]


@dataclass(frozen=True, slots=True)
class OfficialSimulation:
    values: Tensor
    times: Tensor
    initial_state: Tensor
    future_noise: Tensor
    regime_sequence: Tensor
    boundary_event_count: int


class OfficialWorldDriver(Protocol):
    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        raise NotImplementedError

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        raise NotImplementedError


def paired_rollout(driver: OfficialWorldDriver, request: OraclePairRequest) -> PairedTrajectory:
    noise = driver.sample_future_noise(request.base)
    factual = driver.simulate(request.base, request.factual_schedule, noise)
    counterfactual = driver.simulate(request.base, request.counterfactual_schedule, noise)
    return validate_pair(request, factual, counterfactual)
```

Enforce identical initial state/noise, bitwise identity for no-op, and unchanged
non-target concept schedule.

- [ ] **Step 5: Persist separate truth and physical partitions**

Publish graph, latent concepts, regime sequence, noise and shocks through
`LocalArtifactStore`; build `SCMTruthManifest` only from returned ArtifactRefs.
Map qualification partitions to the four Stage1A physical partitions without
re-splitting, fit normalization only on TRAIN, and require
`audit_partition_isolation()` to pass.

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_oracle_contracts.py tests/stage1b/test_partition_bridge.py tests/stage1b/test_paired_replay.py tests/stage1b/test_dataset.py tests/stage1a -q
git add src/tarca/stage1b/oracle_contracts.py src/tarca/stage1b/oracle.py src/tarca/stage1b/persistence.py src/tarca/stage1b/dataset.py src/tarca/stage1b/truth.py tests/stage1b
git commit -m "feat: add generator-owned stage1b oracle bridge"
```

---

### Task 5: Direct Official World Drivers

**Files:**
- Create: `src/tarca/stage1b/official_worlds.py`
- Modify: `src/tarca/stage1b/worlds.py`
- Modify: `configs/stage1b/worlds_v2.yaml`
- Modify: `tests/stage1b/test_worlds.py`
- Modify: `tests/stage1b/test_official_source_integration.py`

**Interfaces:**
- Produces:
  `build_official_world(config, sources) -> OfficialWorldDriver`

- [ ] **Step 1: Write failing formal-origin and noise tests**

```python
def test_oracle_world_rejects_local_equation_fallback() -> None:
    with pytest.raises(SourceVerificationError, match="official"):
        build_official_world(world_config(), MaterializedSources.empty())


def test_driver_reuses_explicit_future_noise() -> None:
    pair = paired_rollout(fake_official_driver(), identity_pair_request())
    assert torch.equal(pair.factual.future_noise, pair.counterfactual.future_noise)
    assert torch.equal(pair.factual.values, pair.counterfactual.values)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_worlds.py tests/stage1b/test_official_source_integration.py -q
```

- [ ] **Step 3: Implement the official driver boundary**

```python
class NeuralGcLorenz96Driver:
    source_id = "neural_gc"

    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        return explicit_noise_from_seed(request.seed, request.length, self.dimension)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        return run_pinned_neural_gc(request, schedule, future_noise)
```

Implement one driver each for Neural-GC VAR/L96, GVAR predator-prey, JMLR
two-scale L96 and Interfere CML. Load source only inside the worker process.
Retain local equations solely for one-step diagnostics; formal receipts reject
the local fallback.

- [ ] **Step 4: Add project-valid source/base pairs**

Add immutable `concept_pairs` to each primary world. Require one trend pair and
one scale pair, exact evidence asset IDs, shared initial/noise policy, and one
changed parameter family. Reject parameters not supported by the pinned paper
or official configuration.

```yaml
concept_pairs:
  - pair_id: trend_primary
    concept: trend
    factual_parameter_ref: paper_baseline
    counterfactual_parameter_ref: paper_high_forcing
    shared_initial_state: true
    shared_future_noise: true
    evidence_asset_ids: [paper_pdf, official_config]
  - pair_id: scale_primary
    concept: scale
    factual_parameter_ref: paper_baseline
    counterfactual_parameter_ref: paper_high_coupling
    shared_initial_state: true
    shared_future_noise: true
    evidence_asset_ids: [paper_pdf, official_config]
```

- [ ] **Step 5: Run parity/health tests and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_worlds.py tests/stage1b/test_paired_replay.py -q
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_official_source_integration.py -m official_source -q
git add src/tarca/stage1b/official_worlds.py src/tarca/stage1b/worlds.py configs/stage1b/worlds_v2.yaml tests/stage1b
git commit -m "feat: run stage1b worlds through official generators"
```

---

### Task 6: Official PatchTST Backbone and Operable Adapter

**Files:**
- Create: `src/tarca/stage1b/modeling/__init__.py`
- Create: `src/tarca/stage1b/modeling/base.py`
- Create: `src/tarca/stage1b/modeling/hooks.py`
- Create: `src/tarca/stage1b/modeling/patchtst.py`
- Modify: `src/tarca/stage1b/neural.py`
- Create: `tests/stage1b/test_official_patchtst.py`
- Modify: `tests/stage1b/test_neural_predictors.py`

**Interfaces:**
- Produces: `OfficialPatchTSTPredictor`
- Preserves export: `PatchTSTReference = OfficialPatchTSTPredictor`

- [ ] **Step 1: Write failing upstream-parity and site tests**

```python
@pytest.mark.official_source
def test_patchtst_mean_is_upstream_mean() -> None:
    upstream, adapter = paired_patchtst(seed=104729)
    x = torch.randn(2, 64, 20)
    assert torch.allclose(adapter.forward_distribution(x).mean, upstream(x), atol=1e-6, rtol=0)


def test_patchtst_does_not_claim_cross_variable_mechanism() -> None:
    assert official_patchtst_stub().supports_cross_variable_claim is False
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_neural_predictors.py tests/stage1b/test_official_patchtst.py -q
```

- [ ] **Step 3: Implement shared model/source types**

```python
@dataclass(frozen=True, slots=True)
class ModelSourceContext:
    source_id: str
    commit: str
    source_root: Path
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class WindowShape:
    history: int
    horizon: int
    variables: int
```

Model hash includes source commit, backbone weights, scale-head weights and
adapter config.

- [ ] **Step 4: Load the pinned official PatchTST**

Use context 64, target 24, patch 16, stride 8, d_model 128, three layers,
16 heads, d_ff 256, dropout 0.1 and RevIN. Preserve the official mean head.
Attach a conditional softplus scale head to the final representation without
changing mean output.

```python
class OfficialPatchTSTPredictor(nn.Module):
    def __init__(self, source: ModelSourceContext, shape: WindowShape) -> None:
        super().__init__()
        self._mean_backbone = load_pinned_patchtst(source, shape)
        self._scale_head = ConditionalScaleHead(shape)
        self._sites = build_patchtst_site_registry(self._mean_backbone)

    def forward_distribution(self, x: Tensor) -> ForecastDistribution:
        mean, representation = forward_with_representation(self._mean_backbone, x)
        scale = F.softplus(self._scale_head(representation)) + 1e-6
        return ForecastDistribution(mean=mean, scale=scale)
```

- [ ] **Step 5: Implement safe capture/swap hooks**

Resolve only module paths registered at construction. Clone captured output,
validate axes and replacement shape, remove hooks in `finally`, preserve
bitwise output for identity replacement, and assert all weights remain equal
after intervention.

```python
@contextmanager
def installed_site_swap(model: nn.Module, site: OperableSite, replacement: Tensor):
    module = resolve_registered_module(model, site)
    handle = module.register_forward_hook(make_validated_swap(site, replacement.clone()))
    try:
        yield
    finally:
        handle.remove()
```

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_neural_predictors.py tests/stage1b/test_official_patchtst.py -q
git add src/tarca/stage1b/modeling src/tarca/stage1b/neural.py tests/stage1b/test_neural_predictors.py tests/stage1b/test_official_patchtst.py
git commit -m "feat: adapt the official patchtst backbone"
```

---

### Task 7: Official iTransformer, CUDA Training, and Atomic Checkpoints

**Files:**
- Create: `src/tarca/stage1b/modeling/itransformer.py`
- Modify: `src/tarca/stage1b/modeling/__init__.py`
- Modify: `src/tarca/stage1b/neural.py`
- Modify: `src/tarca/stage1b/training.py`
- Create: `tests/stage1b/test_official_itransformer.py`
- Modify: `tests/stage1b/test_training_reproducibility.py`

**Interfaces:**
- Produces: `OfficialITransformerPredictor`
- Produces:
  `train_candidate(..., policy: TrainingPolicy, progress: ProgressSink) -> TrainingResult`

- [ ] **Step 1: Write failing parity/device/checkpoint tests**

```python
@pytest.mark.official_source
def test_itransformer_mean_is_upstream_mean() -> None:
    upstream, adapter = paired_itransformer(seed=104729)
    x = torch.randn(2, 64, 20)
    assert torch.allclose(adapter.forward_distribution(x).mean, upstream(x), atol=1e-6, rtol=0)


def test_training_receipt_binds_device_and_checkpoint(tmp_path: Path) -> None:
    result = run_tiny_training(device="cpu", checkpoint_root=tmp_path)
    assert result.receipt.device == "cpu"
    assert len(result.receipt.checkpoint_sha256) == 64
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_official_itransformer.py tests/stage1b/test_training_reproducibility.py -q
```

- [ ] **Step 3: Implement official iTransformer**

Load pinned `models/iTransformer.py::Model` with seq_len 64, pred_len 24,
d_model 512, three encoder layers, eight heads, d_ff 512, dropout 0.1 and
official normalization. Preserve upstream mean, add a conditional scale head,
and register layer × variable × subspace sites.

```python
class OfficialITransformerPredictor(nn.Module):
    def __init__(self, source: ModelSourceContext, shape: WindowShape) -> None:
        super().__init__()
        self._mean_backbone = load_pinned_itransformer(source, shape)
        self._scale_head = ConditionalScaleHead(shape)
        self._sites = build_itransformer_site_registry(self._mean_backbone, shape.variables)

    @property
    def operable_sites(self) -> tuple[OperableSite, ...]:
        return self._sites
```

- [ ] **Step 4: Implement CUDA/AMP training policy**

```python
@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    device: str
    precision: Literal["FP32", "AMP_FP16"]
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    dataloader_workers: int
    checkpoint_root: Path
    checkpoint_every_epochs: int = 1


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int
    batch: int
    completed_steps: int
    total_steps: int
    samples_per_second: float


class ProgressSink(Protocol):
    def report(self, progress: TrainingProgress) -> None:
        raise NotImplementedError
```

Move model/batches to the policy device, use pinned-memory DataLoader settings
on CUDA, autocast and GradScaler for AMP, and atomically checkpoint task/data/
config/precision hashes, epoch, optimizer/model state and RNG states.

- [ ] **Step 5: Test deterministic resume and operability**

Interrupt after epoch 1 and compare resumed versus uninterrupted final model
hash on CPU. Test capture, identity/full/subspace swaps, finite positive scale,
input device, and frozen weights.

```python
def test_resumed_training_matches_uninterrupted(tmp_path: Path) -> None:
    uninterrupted = run_tiny_training(checkpoint_root=tmp_path / "full")
    interrupted = run_tiny_training(checkpoint_root=tmp_path / "resume", stop_after_epoch=1)
    resumed = resume_tiny_training(interrupted.checkpoint)
    assert resumed.receipt.model_sha256 == uninterrupted.receipt.model_sha256
```

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_official_itransformer.py tests/stage1b/test_training_reproducibility.py tests/stage1b/test_neural_predictors.py -q
git add src/tarca/stage1b/modeling src/tarca/stage1b/neural.py src/tarca/stage1b/training.py tests/stage1b
git commit -m "feat: add official itransformer cuda training"
```

---

### Task 8: Execution Contracts and Staged Scientific Compilation

**Files:**
- Create: `src/tarca/execution/__init__.py`
- Create: `src/tarca/execution/contracts.py`
- Create: `src/tarca/stage1b/compiler.py`
- Create: `tests/execution/__init__.py`
- Create: `tests/execution/test_contracts.py`
- Create: `tests/stage1b/test_compiler.py`

**Interfaces:**
- Produces: `compile_stage1b_graph(...) -> Stage1BRunGraph`
- Produces:
  `compile_ready_manifest(graph, completed: Mapping[str, ArtifactRef]) -> TaskManifest`

- [ ] **Step 1: Write failing contract/compiler tests**

```python
def test_completed_policy_is_never_rerun() -> None:
    manifest = task_manifest()
    assert manifest.completed_task_policy == "NEVER_RERUN"


def test_primary_training_graph_contains_twelve_gpu_nodes() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs())
    assert sum(node.phase == "NEURAL_TRAIN" for node in graph.nodes) == 12
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_contracts.py tests/stage1b/test_compiler.py -q
```

- [ ] **Step 3: Implement authority-shaped execution types**

Implement immutable, extra-forbid `ScientificIdentity`, `ResourceRequest`,
`TaskSpec`, `TaskManifest`, `ResourceAllocation`, `PlannedTask`,
`ExecutionPlan`, `ExecutionContext`, `TaskState`, `TaskResult`, and
`MonitoringSnapshot` with the protocol fields. `COMPLETED` requires a verified
ArtifactRef.

```python
class ScientificIdentity(StrictContractModel):
    task_type: str
    world_id: str | None
    model_id: str | None
    seed: int | None
    config_sha256: Sha256Hash
    code_sha256: Sha256Hash
    data_sha256: Sha256Hash | None


class TaskSpec(StrictContractModel):
    schema_version: Literal["2.0.0"]
    task_id: str
    identity: ScientificIdentity
    inputs: tuple[ArtifactRef, ...]
    output_artifact_type: str
    executor_key: str
    resources: ResourceRequest


class TaskState(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALLED = "STALLED"
```

- [ ] **Step 4: Implement the staged run graph**

```python
@dataclass(frozen=True, slots=True)
class Stage1BJobNode:
    node_id: str
    identity: ScientificIdentity
    phase: str
    dependency_ids: tuple[str, ...]
    expected_input_types: tuple[str, ...]
    output_artifact_type: str
    resource_request: ResourceRequest
    executor_key: str


@dataclass(frozen=True, slots=True)
class Stage1BRunGraph:
    graph_id: str
    nodes: tuple[Stage1BJobNode, ...]
```

Materialize ready TaskSpecs only after dependency outputs exist as verified
ArtifactRefs. This preserves protocol input rules and enables a pre-hashed DAG.

- [ ] **Step 5: Verify scientific identity invariance and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_contracts.py tests/stage1b/test_compiler.py -q
git add src/tarca/execution src/tarca/stage1b/compiler.py tests/execution tests/stage1b/test_compiler.py
git commit -m "feat: compile stage1b into immutable execution tasks"
```

---

### Task 9: SQLite State, Allowlisted Workers, and Recovery

**Files:**
- Create: `src/tarca/execution/state.py`
- Create: `src/tarca/execution/registry.py`
- Create: `src/tarca/execution/worker.py`
- Create: `tests/execution/test_state.py`
- Create: `tests/execution/test_worker.py`

**Interfaces:**
- Produces: `ExecutionStateStore`
- Produces: `ExecutorRegistry`
- Produces: `run_worker(context, store, registry) -> TaskResult`

- [ ] **Step 1: Write failing state/recovery tests**

```python
def test_completed_job_is_never_claimed_again(tmp_path: Path) -> None:
    store = completed_store(tmp_path)
    assert store.claim_ready("worker-2", limit=1) == ()


def test_restart_marks_only_dead_worker_stalled(tmp_path: Path) -> None:
    result = running_store(tmp_path).reconcile_processes(process_probe())
    assert result.live_task_ids == ("live-task",)
    assert result.stalled_task_ids == ("dead-task",)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_state.py tests/execution/test_worker.py -q
```

- [ ] **Step 3: Implement WAL state and compare-and-set transitions**

Create `runs`, `job_nodes`, `task_specs`, `attempts`, `dependencies`,
`progress_events`, `resource_samples`, and `alerts`. Enable WAL, foreign keys
and a five-second busy timeout. Use parameterized SQL and explicit
transactions; state updates require the expected prior state.

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    state TEXT NOT NULL,
    worker_id TEXT,
    pid INTEGER,
    process_started_at_utc TEXT,
    heartbeat_at_utc TEXT,
    error_category TEXT,
    CHECK (state IN ('READY','RUNNING','COMPLETED','FAILED','STALLED'))
);
```

The compare-and-set write is parameterized and must affect exactly one row:

```python
cursor = connection.execute(
    "UPDATE attempts SET state = ? WHERE attempt_id = ? AND state = ?",
    (next_state.value, attempt_id, expected_state.value),
)
if cursor.rowcount != 1:
    raise StateTransitionConflict(attempt_id)
```

- [ ] **Step 4: Implement the immutable allowlist**

```python
Executor = Callable[[TaskSpec, ExecutionContext, ProgressSink], ArtifactRef]


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, Executor]) -> None:
        self._executors = MappingProxyType(dict(executors))

    def resolve(self, key: str) -> Executor:
        if key not in self._executors:
            raise ValueError("executor key is not allowlisted")
        return self._executors[key]
```

Never load a shell string or callable path from SQLite.

- [ ] **Step 5: Implement recovery/retry policy**

Match PID, process creation time, run ID and task ID before preserving a live
worker. Mark expired heartbeats STALLED. Retry transient IO/worker death once;
retry OOM once only after lower packing. Hash/truth/NaN/leakage/identity errors
remain terminal.

```python
class RetryDisposition(StrEnum):
    RETRY_ONCE = "RETRY_ONCE"
    RETRY_ONCE_WITH_LOWER_PACKING = "RETRY_ONCE_WITH_LOWER_PACKING"
    TERMINAL = "TERMINAL"


RETRY_POLICY = MappingProxyType({
    "TRANSIENT_IO": RetryDisposition.RETRY_ONCE,
    "WORKER_DIED": RetryDisposition.RETRY_ONCE,
    "CUDA_OOM": RetryDisposition.RETRY_ONCE_WITH_LOWER_PACKING,
    "HASH_DRIFT": RetryDisposition.TERMINAL,
    "TRUTH_LEAKAGE": RetryDisposition.TERMINAL,
    "NONFINITE": RetryDisposition.TERMINAL,
    "IDENTITY_DRIFT": RetryDisposition.TERMINAL,
})
```

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_state.py tests/execution/test_worker.py -q
git add src/tarca/execution/state.py src/tarca/execution/registry.py src/tarca/execution/worker.py tests/execution
git commit -m "feat: add recoverable stage1b execution state"
```

---

### Task 10: Resource Planning, Telemetry, and ETA

**Files:**
- Create: `src/tarca/execution/resources.py`
- Create: `src/tarca/execution/telemetry.py`
- Create: `src/tarca/execution/eta.py`
- Create: `tests/execution/test_resources.py`
- Create: `tests/execution/test_telemetry.py`
- Create: `tests/execution/test_eta.py`
- Modify: `src/tarca/stage1b/hardware.py`
- Modify: `tests/stage1b/test_hardware_gate.py`

**Interfaces:**
- Produces: `plan_resources(...) -> tuple[ResourceAllocation, ...]`
- Produces: `collect_resource_sample(...) -> ResourceSample`
- Produces: `estimate_run_eta(...) -> EtaEstimate`

- [ ] **Step 1: Write failing packing and ETA tests**

```python
def test_low_utilization_admits_second_job() -> None:
    decision = decide_gpu_packing(gpu_sample(utilization=62, used_gib=7), 181, 1)
    assert decision.target_jobs == 2


def test_eta_uses_two_gpu_critical_path() -> None:
    estimate = estimate_run_eta(two_equal_gpu_queues(hours_each=5))
    assert estimate.remaining_seconds == pytest.approx(5 * 3600)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_resources.py tests/execution/test_telemetry.py tests/execution/test_eta.py tests/stage1b/test_hardware_gate.py -q
```

- [ ] **Step 3: Implement resource samples and packing**

```python
@dataclass(frozen=True, slots=True)
class GpuSample:
    gpu_id: int
    utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    power_watts: float
    temperature_celsius: float
    compute_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResourceSample:
    sampled_at_utc: datetime
    host_cpu_percent: float
    effective_busy_cores: float
    process_rss_bytes: int
    process_pss_bytes: int | None
    host_memory_used_bytes: int
    gpu_samples: tuple[GpuSample, ...]
    disk_read_bytes_per_second: float
    disk_write_bytes_per_second: float
```

Track per-card utilization, VRAM, power, temperature and CUDA PIDs plus process
tree CPU, affinity, effective busy cores, RSS/PSS, host RAM and disk I/O.
Start one job per GPU; admit two below 70%/8GB after 180 seconds; admit three
below 80%/18GB; reduce above 20GB or on OOM/throttling/throughput loss/data
wait. Never alter TaskSpec.

- [ ] **Step 4: Implement CPU/RAM admission**

```python
@dataclass(frozen=True, slots=True)
class HostAdmissionPolicy:
    scheduler_monitor_cores: int = 1
    system_io_reserved_cores: int = 3
    maximum_data_cores: int = 24
    maximum_host_memory_bytes: int = 200 * 1024**3
    initial_loader_workers_per_gpu_job: int = 3
```

Reserve 1 CPU for monitoring and 1–4 for system/I/O; allow 20–26 for data
generation; start each GPU worker with 2–4 DataLoader workers; cap OMP/MKL and
set affinity. Cap admitted RAM at 200GB and fail preflight without suitable
local storage.

Sample runtime state every two seconds, persist at five-to-ten-second cadence,
and downsample long histories. Alert when monitoring exceeds one CPU core or
1GB RAM.

- [ ] **Step 5: Implement ETA and two-level time gate**

```python
@dataclass(frozen=True, slots=True)
class EtaEstimate:
    status: Literal["CALIBRATING", "READY"]
    remaining_seconds: float | None
    expected_completion_utc: datetime | None
    lower_seconds: float | None
    upper_seconds: float | None
    exceeds_24_hours: bool
```

Use per-world/model EWMA plus validation/checkpoint/wait overhead and the
longest remaining dependency path. Above 24 hours requires user authorization;
above 120 hours is infeasible. Neither condition shrinks workload.

Default to independent task-level GPU parallelism. Permit dual-GPU DDP only
when a bounded same-config probe proves at least a 30% wall-time reduction.
Choose FP32 or AMP during preflight, write the choice to a precision artifact,
and keep it fixed for the entire qualification.

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_resources.py tests/execution/test_telemetry.py tests/execution/test_eta.py tests/stage1b/test_hardware_gate.py -q
git add src/tarca/execution src/tarca/stage1b/hardware.py tests/execution tests/stage1b/test_hardware_gate.py
git commit -m "feat: plan and monitor dual gpu stage1b work"
```

---

### Task 11: Stage1B Executors and Science-Blind Scheduler

**Files:**
- Create: `src/tarca/stage1b/jobs.py`
- Create: `src/tarca/execution/scheduler.py`
- Create: `src/tarca/execution/worker_entry.py`
- Modify: `src/tarca/stage1b/runner.py`
- Modify: `scripts/run_stage1b_qualification.py`
- Create: `tests/execution/test_scheduler.py`
- Modify: `tests/stage1b/test_runner_integration.py`

**Interfaces:**
- Produces: `stage1b_executor_registry(repo_root) -> ExecutorRegistry`
- Produces: `Scheduler.run_until_terminal(run_id) -> RunTerminalStatus`

- [ ] **Step 1: Write failing science-blind/backfill tests**

```python
def test_scheduler_has_no_scientific_metric_columns() -> None:
    assert {"crps", "nll", "mae", "ranking"}.isdisjoint(Scheduler.visible_columns)


def test_idle_gpus_are_backfilled() -> None:
    launches = scheduler_with_two_gpus_and_three_ready_jobs().tick()
    assert {launch.allocation.gpu_ids for launch in launches} == {(0,), (1,)}
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_scheduler.py tests/stage1b/test_runner_integration.py -q
```

- [ ] **Step 3: Register exact Stage1B executors**

Register source materialization, reproduction, world generation, dataset
validation, VAR fit/score, neural training, model operability, score
aggregation, gate evaluation and receipt publication. Each consumes verified
ArtifactRefs and produces one verified ArtifactRef.

```python
def stage1b_executor_registry(repo_root: Path) -> ExecutorRegistry:
    return ExecutorRegistry({
        "stage1b.materialize_source": bind_repo(materialize_source_job, repo_root),
        "stage1b.reproduce_official": bind_repo(reproduce_official_job, repo_root),
        "stage1b.generate_world": bind_repo(generate_world_job, repo_root),
        "stage1b.validate_dataset": bind_repo(validate_dataset_job, repo_root),
        "stage1b.fit_var": bind_repo(fit_var_job, repo_root),
        "stage1b.train_neural": bind_repo(train_neural_job, repo_root),
        "stage1b.check_operability": bind_repo(check_operability_job, repo_root),
        "stage1b.score_model": bind_repo(score_model_job, repo_root),
        "stage1b.aggregate_scores": bind_repo(aggregate_scores_job, repo_root),
        "stage1b.evaluate_gate": bind_repo(evaluate_gate_job, repo_root),
        "stage1b.publish_receipt": bind_repo(publish_receipt_job, repo_root),
    })
```

- [ ] **Step 4: Launch workers without a shell**

```python
arguments = (
    sys.executable, "-m", "tarca.execution.worker_entry",
    "--database", str(database_path),
    "--run-id", run_id,
    "--task-id", task_id,
    "--attempt-id", attempt_id,
)
subprocess.Popen(arguments, shell=False, env=worker_environment, start_new_session=True)
```

Set CUDA visibility, OMP/MKL limits and affinity from allocation.

- [ ] **Step 5: Convert serial runner to graph execution**

Use a `SynchronousTestBackend` for tiny local integration and
`LocalMultiProcessBackend` on server. Require all comparisons before aggregate
gate evaluation. Assert identical scientific hashes across backends.

```python
class RunTerminalStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class WorkerBackend(Protocol):
    def launch(self, task: PlannedTask, database_path: Path) -> WorkerHandle:
        raise NotImplementedError

    def poll(self, handles: tuple[WorkerHandle, ...]) -> tuple[WorkerObservation, ...]:
        raise NotImplementedError
```

- [ ] **Step 6: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/execution/test_scheduler.py tests/stage1b/test_runner_integration.py tests/stage1b/test_stage1b_cli.py -q
git add src/tarca/stage1b/jobs.py src/tarca/execution/scheduler.py src/tarca/execution/worker_entry.py src/tarca/stage1b/runner.py scripts/run_stage1b_qualification.py tests
git commit -m "feat: schedule stage1b qualification as isolated jobs"
```

---

### Task 12: Read-Only FastAPI Monitoring

**Files:**
- Create: `src/tarca/monitoring/__init__.py`
- Create: `src/tarca/monitoring/schemas.py`
- Create: `src/tarca/monitoring/repository.py`
- Create: `src/tarca/monitoring/api.py`
- Create: `tests/monitoring/__init__.py`
- Create: `tests/monitoring/test_repository.py`
- Create: `tests/monitoring/test_api.py`

**Interfaces:**
- Produces: `create_monitoring_app(database_path, static_root) -> FastAPI`
- Provides GET `/api/v1/run`, `/jobs`, `/resources`, `/alerts`
- Provides WebSocket `/api/v1/stream`

- [ ] **Step 1: Write failing read-only/data-leak tests**

```python
def test_api_rejects_mutation(client: TestClient) -> None:
    assert client.post("/api/v1/jobs/task-1/restart").status_code == 405
    assert client.delete("/api/v1/jobs/task-1").status_code == 405


def test_api_excludes_scientific_fields(client: TestClient) -> None:
    payload = json.dumps(client.get("/api/v1/jobs").json()).lower()
    for forbidden in ("crps", "nll", "mae", "truth", "ranking", "best_seed"):
        assert forbidden not in payload
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/monitoring -q
```

- [ ] **Step 3: Implement explicit response models**

`JobStatusView` includes task/world/model/seed identity, state, PID/alive,
GPU IDs, expected/actual CPU cores/RAM/VRAM, epoch/batch, heartbeat, retry,
ETA and error category. `RunSummaryView`, `ResourceView` and `AlertView`
contain only approved runtime fields. Never serialize raw rows.

```python
class JobStatusView(BaseModel):
    task_id: str
    world_id: str | None
    model_id: str | None
    seed: int | None
    state: str
    pid: int | None
    alive: bool
    gpu_ids: tuple[int, ...]
    expected_cpu_cores: int
    actual_effective_busy_cores: float
    expected_ram_bytes: int
    actual_rss_bytes: int
    expected_vram_bytes: int
    actual_vram_bytes: int
    epoch: int | None
    batch: int | None
    heartbeat_at_utc: datetime | None
    retry_count: int
    eta_seconds: float | None
    error_category: str | None
```

- [ ] **Step 4: Implement safe queries/streaming**

Use read-only SQLite connections, select allowlisted columns, stream the same
Pydantic snapshots every two seconds, and serve static files only below
`static_root`. Do not add arbitrary file or log-path APIs.

```python
SAFE_JOB_COLUMNS = (
    "task_id", "world_id", "model_id", "seed", "state", "pid", "alive",
    "gpu_ids", "expected_cpu_cores", "actual_effective_busy_cores",
    "expected_ram_bytes", "actual_rss_bytes", "expected_vram_bytes",
    "actual_vram_bytes", "epoch", "batch", "heartbeat_at_utc",
    "retry_count", "eta_seconds", "error_category",
)


def open_readonly(database_path: Path) -> sqlite3.Connection:
    uri = f"file:{database_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)
```

- [ ] **Step 5: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/monitoring -q
git add src/tarca/monitoring tests/monitoring
git commit -m "feat: expose read-only stage1b monitoring api"
```

---

### Task 13: React Monitoring Dashboard

**Files:**
- Create: `frontend/stage1b-monitor/package.json`
- Create: `frontend/stage1b-monitor/package-lock.json`
- Create: `frontend/stage1b-monitor/tsconfig.json`
- Create: `frontend/stage1b-monitor/vite.config.ts`
- Create: `frontend/stage1b-monitor/src/api.ts`
- Create: `frontend/stage1b-monitor/src/types.ts`
- Create: `frontend/stage1b-monitor/src/App.tsx`
- Create: `frontend/stage1b-monitor/src/components/*.tsx`
- Create: `frontend/stage1b-monitor/src/*.test.tsx`
- Create: `frontend/stage1b-monitor/e2e/dashboard.spec.ts`

**Interfaces:**
- Consumes: Task 12 REST/WebSocket schemas
- Produces: static `frontend/stage1b-monitor/dist`

- [ ] **Step 1: Scaffold locked frontend dependencies**

Use React 19, TypeScript, Vite, ECharts, Vitest, Testing Library and
Playwright. Commit `package-lock.json`; load no CDN scripts.

- [ ] **Step 2: Write failing dashboard tests**

```tsx
it("shows expected and actual resources for both GPUs", async () => {
  render(<App api={fakeApi(twoGpuSnapshot)} />);
  expect(await screen.findByText("GPU 0")).toBeVisible();
  expect(screen.getByText("GPU 1")).toBeVisible();
  expect(screen.getByText("期望显存")).toBeVisible();
  expect(screen.getByText("实际显存")).toBeVisible();
  expect(screen.getByText("有效忙核")).toBeVisible();
});

it("contains no task mutation controls", () => {
  render(<App api={fakeApi(twoGpuSnapshot)} />);
  expect(screen.queryByRole("button", { name: /重启|停止|删除|修改/ })).toBeNull();
});
```

- [ ] **Step 3: Run RED**

```bash
cd frontend/stage1b-monitor
npm ci
npm test -- --run
```

- [ ] **Step 4: Implement the approved views**

Build run summary, resource cards, job table, telemetry charts and alert panel.
Show “校准中” without ETA. Use WebSocket with REST fallback and capped
exponential reconnect. Render server text normally; never use
`dangerouslySetInnerHTML`.

```tsx
export function App({ api }: { api: MonitoringApi }) {
  const snapshot = useRuntimeSnapshot(api);
  return (
    <main>
      <RunSummary summary={snapshot.run} />
      <ResourceGrid resources={snapshot.resources} />
      <JobTable jobs={snapshot.jobs} />
      <TelemetryCharts resources={snapshot.resources} />
      <AlertPanel alerts={snapshot.alerts} />
    </main>
  );
}
```

- [ ] **Step 5: Add E2E and 80% coverage**

```tsx
test("renders a running two-GPU job set", async ({ page }) => {
  await page.route("**/api/v1/**", route => route.fulfill(mockResponse(route.request().url())));
  await page.goto("/");
  await expect(page.getByText("Stage1B v2")).toBeVisible();
  await expect(page.getByText("GPU 0")).toBeVisible();
  await expect(page.getByText("GPU 1")).toBeVisible();
  await expect(page.getByText("预计剩余时间")).toBeVisible();
});
```

- [ ] **Step 6: Run GREEN and commit**

```bash
cd frontend/stage1b-monitor
npm test -- --run --coverage
npm run build
npx playwright test
git add frontend/stage1b-monitor
git commit -m "feat: add stage1b runtime monitoring dashboard"
```

---

### Task 14: Single-Container Runtime and Local CLI

**Files:**
- Create: `deploy/stage1b/Dockerfile`
- Create: `deploy/stage1b/entrypoint.sh`
- Create: `deploy/stage1b/compose.stage1b-v2.yaml`
- Create: `scripts/run_stage1b_runtime.py`
- Create: `tests/stage1b/test_runtime_cli.py`
- Create: `tests/stage1b/test_container_contract.py`

**Interfaces:**
- Commands: `preflight`, `launch`, `resume`, `status`

- [ ] **Step 1: Write failing CLI/container tests**

```python
def test_runtime_cli_has_no_formal_experiment_commands() -> None:
    help_text = run_cli("--help").stdout
    assert all(name in help_text for name in ("preflight", "launch", "resume", "status"))
    assert "E01" not in help_text and "E02" not in help_text


def test_monitor_port_is_host_loopback_only() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert compose["services"]["stage1b"]["ports"] == ["127.0.0.1:8765:8765"]
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_runtime_cli.py tests/stage1b/test_container_contract.py -q
```

- [ ] **Step 3: Build the multi-stage image**

```dockerfile
FROM node:20-bookworm-slim AS ui-build
WORKDIR /ui
COPY frontend/stage1b-monitor/package*.json ./
RUN npm ci
COPY frontend/stage1b-monitor/ ./
RUN npm run build

FROM pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04
WORKDIR /opt/tarca
COPY deploy/stage1b/requirements-server.lock /tmp/requirements-server.lock
RUN python -m pip install --require-hashes -r /tmp/requirements-server.lock
COPY . /opt/tarca
COPY --from=ui-build /ui/dist /opt/tarca/frontend/stage1b-monitor/dist
ENV PYTHONPATH=/opt/tarca/deploy/stage1b/py310:/opt/tarca/src
ENTRYPOINT ["/opt/tarca/deploy/stage1b/entrypoint.sh"]
```

Runtime has no Node, Docker socket or public host port. Configure shared memory,
two GPUs, read-only official sources and artifact/checkpoint volumes.

- [ ] **Step 4: Implement lifecycle and CLI**

Entrypoint validates environment/source/storage, starts monitoring/telemetry,
then launches or resumes scheduler and forwards SIGTERM. `preflight` performs
only bounded probes/calibration. `launch` requires passing receipts and blocks
ETA above 24 hours without an authorization receipt. `status` emits safe JSON.

```python
def dispatch_runtime_command(arguments: RuntimeArguments) -> int:
    handlers = MappingProxyType({
        "preflight": run_preflight,
        "launch": launch_runtime,
        "resume": resume_runtime,
        "status": emit_safe_status,
    })
    return handlers[arguments.command](arguments)
```

- [ ] **Step 5: Verify and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_runtime_cli.py tests/stage1b/test_container_contract.py -q
docker build -f deploy/stage1b/Dockerfile -t tarca-stage1b-v2:local .
git add deploy/stage1b scripts/run_stage1b_runtime.py tests/stage1b/test_runtime_cli.py tests/stage1b/test_container_contract.py
git commit -m "feat: package stage1b v2 server runtime"
```

---

### Task 15: Qualification Receipts, v2 Revisions, and Active Docs

**Files:**
- Modify: `src/tarca/stage1b/runner.py`
- Modify: `src/tarca/stage1b/freeze.py`
- Modify: `scripts/run_stage1b_qualification.py`
- Modify: `tests/stage1b/receipt_helpers.py`
- Modify: `tests/stage1b/test_freeze.py`
- Modify: `tests/stage1b/test_runner_integration.py`
- Modify: `docs/research/stage1b_world_qualification_spec.md`
- Modify: `docs/research/stage1b_v2_build_report.md`

**Interfaces:**
- Produces:
  `freeze_suite(receipt, artifact_root, series="v2", revision_id="v2-r1", authorization=None)`

- [ ] **Step 1: Write failing same-v2 revision test**

```python
def test_authorized_override_keeps_v2_series(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path, series="v2", revision_id="v2-r1")
    freeze_suite(
        passing_receipt(),
        tmp_path,
        series="v2",
        revision_id="v2-r2",
        authorization=OverrideAuthorization(
            authorized_by="user",
            reason="approved v2 override",
            prior_revision_id="v2-r1",
        ),
    )
    assert load_active_pointer(tmp_path)["revision_id"] == "v2-r2"
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_freeze.py tests/stage1b/test_runner_integration.py -q
```

- [ ] **Step 3: Extend receipts and freeze layout**

Require official-source, reproduction, environment, precision, graph,
TaskManifest, ExecutionPlan and hardware hashes. Preserve all comparison and
failure rows. Reject partial tasks, reserved seeds, E01/E02, source drift and
identity drift. Store immutable revisions under
`versions/v2/revisions/rN/`; failed runs are not frozen.

```python
class QualificationEvidence(FrozenModel):
    official_source_receipt_sha256: Sha256Hash
    reproduction_receipt_sha256: Sha256Hash
    environment_receipt_sha256: Sha256Hash
    precision_receipt_sha256: Sha256Hash
    run_graph_sha256: Sha256Hash
    task_manifest_sha256: Sha256Hash
    execution_plan_sha256: Sha256Hash
    hardware_receipt_sha256: Sha256Hash
    completed_task_count: int
    expected_task_count: int

    @model_validator(mode="after")
    def reject_partial_run(self) -> Self:
        if self.completed_task_count != self.expected_task_count:
            raise ValueError("partial qualification cannot be frozen")
        return self


def revision_root(artifact_root: Path, revision_id: str) -> Path:
    revision_number = parse_v2_revision_id(revision_id)
    return artifact_root / "versions" / "v2" / "revisions" / f"r{revision_number}"
```

- [ ] **Step 4: Update active documentation**

Record `BUILT_NOT_QUALIFIED`, direct official use, generator-owned truth,
server-only probes and dashboard. Remove the obsolete requirement that an
authorized v2 override must become v3. Keep only the v1 historical report.

```text
Implementation status: BUILT_NOT_QUALIFIED
Active scientific series: v2
Freeze default: immutable after qualification
Authorized override: creates the next immutable v2 revision and updates the active pointer
Executed during build: unit/integration/E2E and bounded probes only
Not executed during build: full Stage1B qualification, E01, E02
```

- [ ] **Step 5: Run GREEN and commit**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest tests/stage1b/test_freeze.py tests/stage1b/test_runner_integration.py tests/stage1b/test_stage1b_cli.py -q
git add src/tarca/stage1b/runner.py src/tarca/stage1b/freeze.py scripts/run_stage1b_qualification.py tests/stage1b docs/research/stage1b_world_qualification_spec.md docs/research/stage1b_v2_build_report.md
git commit -m "feat: freeze authorized stage1b v2 revisions"
```

---

### Task 16: Full Verification and Server Handoff

**Files:**
- Create: `docs/research/stage1b_v2_official_runtime_build_report.md`
- Modify: `README.md`

**Interfaces:**
- Produces: verified build report and server-only acceptance commands

- [ ] **Step 1: Run full Python verification**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pytest --cov=tarca --cov-branch --cov-report=term-missing --cov-fail-under=80
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m ruff check src tests scripts
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m ruff format --check src tests scripts
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m mypy src
```

- [ ] **Step 2: Run frontend verification**

```bash
cd frontend/stage1b-monitor
npm ci
npm test -- --run --coverage
npm run build
npx playwright test
npm audit --omit=dev
```

- [ ] **Step 3: Run security and integrity checks**

```powershell
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python -m pip_audit -r deploy/stage1b/requirements-server.lock
rg -n --hidden --glob '!*.lock' --glob '!docs/auth/**' 'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|api[_-]?key\s*[:=]|password\s*[:=]|token\s*[:=]' .
git diff --check
git diff -- docs/auth
```

- [ ] **Step 4: Run safe build/smoke checks**

```powershell
docker build -f deploy/stage1b/Dockerfile -t tarca-stage1b-v2:local .
docker run --rm tarca-stage1b-v2:local status --empty-ok
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python scripts/check_stage1a.py --json
& 'D:\software\MyAnaconda\Scripts\conda.exe' run -n tarca-stage1b-runtime-py310 python scripts/check_stage1b.py --allow-unfrozen --json
```

- [ ] **Step 5: Record but do not run server-only acceptance**

```bash
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml run --rm stage1b preflight
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml up -d stage1b
docker compose -f deploy/stage1b/compose.stage1b-v2.yaml exec stage1b status
```

Preflight must confirm 2×4090/24GB, 28 CPU cores, 224GB RAM, local NVMe, both
CUDA probes, frozen precision policy, useful packing and ETA at or below
24 hours. These commands do not run locally and do not authorize qualification.

- [ ] **Step 6: Inspect final scope and commit**

```powershell
git status --short
git diff --stat
git diff -- docs/auth
git diff --name-only | rg 'E01|E02|qualification_v3|worlds_v3'
git add README.md docs/research/stage1b_v2_official_runtime_build_report.md
git commit -m "docs: record stage1b v2 official runtime build"
```

## Execution Checkpoints

Stop for inline review after Tasks 0, 5, 7, 11, 14, and 16. At each checkpoint:

1. inspect commits and diff since the prior checkpoint;
2. rerun focused tests for completed tasks;
3. verify no `docs/auth` byte change;
4. verify E01/E02 and full qualification remain unexecuted;
5. report new risks before continuing.

The full Stage1B qualification is a separate, explicitly authorized server
operation after this implementation plan completes.
