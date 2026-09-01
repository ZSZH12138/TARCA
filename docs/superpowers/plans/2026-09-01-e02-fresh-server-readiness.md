# E02 Fresh-Server Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` inline. TARCA policy prohibits subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the already frozen Stage 2 result into a fail-closed E02 server handoff that can be restored, hardware-probed, and left ready for the user's separate formal-run acknowledgement on a fresh 2×RTX 4090 server.

**Architecture:** Keep the approved E02 science graph unchanged. Add an execution-only handoff contract, safe complete-archive restoration, a development-data/checkpoint-only two-GPU probe with a conservative critical-path estimate, launch-time evidence binding, and equivalent Docker-host/container-direct bootstrap paths. The scheduler continues to run one exclusive neural job per GPU and backfills dependency-ready CPU work under the frozen 24-core/200-GiB admission ceiling.

**Tech Stack:** Python 3.10/3.11, PyTorch 2.2.2 + CUDA 12.1 on server, Pydantic, SQLite, pytest, Docker Compose, Bash.

**Spec:** `docs/superpowers/specs/2026-08-31-stage2-e02-local-runtime-design.md`

## Global Constraints

- Do not change `configs/e02/e02_v1.yaml`, its scientific hash `9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c`, or any PASS/FAIL threshold.
- Bind the handoff to complete Stage 2 archive SHA-256 `7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a` and frozen receipt internal SHA-256 `37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166`.
- A preflight probe may read frozen Stage 2 artifacts and development data only; it must report `formal_tasks_executed: 0` and must not create a sealed E02 grant, formal bundle, prediction, score, decision, or E02 receipt.
- Require Python 3.10, PyTorch 2.2.2, CUDA 12.1, exactly two RTX 4090 GPUs with at least 23 GiB driver-reported memory each, at least 28 physical CPU cores, at least 224 GiB RAM, and at least 200 GiB free local storage.
- Use 24 work cores and 200 GiB host-memory admission, reserving one scheduler/monitor core and three system/I/O cores.
- Formal execution must remain blocked until the exact acknowledgement `I_ACKNOWLEDGE_E02_V1_FORMAL_RUN` is supplied in a later user-authorized action.
- Local Python verification uses `D:\software\MyAnaconda\envs\tarca-finalize-py311\python.exe` with process-local `PYTHONPATH=src`; do not modify that environment.

---

### Task 1: Safe Complete-Archive Restoration

**Files:**
- Create: `configs/e02/e02_server_handoff_v1.json`
- Create: `src/tarca/e02/server_handoff.py`
- Create: `tests/e02/test_server_handoff.py`

**Interfaces:**
- Produces: `E02ServerHandoff`, `load_e02_server_handoff(path: Path) -> E02ServerHandoff`
- Produces: `restore_stage2_complete_archive(repository_root: Path, archive_path: Path, handoff_path: Path) -> dict[str, object]`

- [x] **Step 1: Write tests that require exact archive/freeze identity, safe paths, regular files, atomic publication, and overwrite refusal.**
- [x] **Step 2: Run `pytest tests/e02/test_server_handoff.py -q` and confirm failure because the module is absent.**
- [x] **Step 3: Implement strict immutable handoff parsing and streaming safe extraction restricted to `artifacts/stage2/`; verify the frozen suite and create a sealed restore receipt only after all checks pass.**
- [x] **Step 4: Re-run the focused test and confirm all restore/tamper cases pass.**

### Task 2: E02 Hardware, Checkpoint, Throughput, and ETA Probe

**Files:**
- Create: `src/tarca/e02/server_probe.py`
- Create: `src/tarca/e02/server_preflight.py`
- Create: `tests/e02/test_server_probe.py`
- Create: `tests/e02/test_server_preflight.py`

**Interfaces:**
- Produces: `estimate_e02_critical_path_seconds(...) -> float`
- Produces: `run_e02_server_probe(...) -> dict[str, object]`
- Produces: `run_e02_server_preflight(...) -> dict[str, object]`

- [x] **Step 1: Write tests for two concurrent GPU probes plus a third wave, 120-trajectory/51,000-window scaling, 35% safety factor, one-hour reset margin, exact hardware gates, frozen-artifact verification, and zero formal access.**
- [x] **Step 2: Run the new probe/preflight tests and confirm missing-interface failures.**
- [x] **Step 3: Implement a spawn-safe probe that loads all three frozen iTransformer checkpoints, predicts only a fixed validation subset, records finite positive scales and unchanged hashes, and derives a conservative ETA without opening formal data.**
- [x] **Step 4: Implement preflight evidence with immutable archive/config/freeze bindings and re-run the focused tests.**

### Task 3: Bind Launch to Server Evidence and the Frozen Capacity Policy

**Files:**
- Modify: `src/tarca/e02/runtime.py`
- Modify: `src/tarca/e02/runner.py`
- Modify: `scripts/run_e02_v1.py`
- Modify: `tests/e02/test_runtime.py`
- Modify: `tests/e02/test_runner.py`
- Modify: `tests/e02/test_tasks.py`

**Interfaces:**
- Changes: `preflight_e02(..., evidence_path: Path) -> dict[str, Any]`
- Changes: `run_e02_formal(..., policy: HostAdmissionPolicy | None = None) -> E02RunResult`

- [x] **Step 1: Write failing tests proving preflight rejects missing/tampered/stale evidence, launch refuses a local-only receipt, the scheduler admits two GPU predictions plus the linear CPU prediction, and the third neural prediction starts in the next GPU wave.**
- [x] **Step 2: Run the focused tests and observe the current permissive preflight/policy failures.**
- [x] **Step 3: Validate and hash-bind the E02 evidence; pass the exact 24-core/200-GiB/200-GiB-storage policy into the formal scheduler and reserve the first four affinity IDs outside worker placement.**
- [x] **Step 4: Re-run E02 lifecycle, graph, scheduler, and legacy Stage 2 runner tests.**

### Task 4: Fresh-Server Bootstrap Paths

**Files:**
- Create: `deploy/stage2/e02_bootstrap.sh`
- Create: `deploy/stage2/e02_bootstrap_direct.sh`
- Modify: `deploy/stage2/entrypoint.sh`
- Create: `tests/e02/test_server_scripts.py`

**Interfaces:**
- Docker-host entry: `bash deploy/stage2/e02_bootstrap.sh --stage2-archive PATH --server-bundle PATH --remaining-rental-hours N`
- Container-direct entry: `bash deploy/stage2/e02_bootstrap_direct.sh --repository-root PATH --stage2-archive PATH --server-bundle PATH --remaining-rental-hours N`

- [x] **Step 1: Write static and subprocess contract tests for strict argument parsing, sidecar checks, offline dependency setup, restore → prepare → dry-run → hardware probe → bound preflight ordering, and the final `E02_READY_FOR_USER_LAUNCH` stop marker.**
- [x] **Step 2: Run the tests and confirm failure because the entrypoints are absent.**
- [x] **Step 3: Implement equivalent host/direct flows; neither path accepts or invokes the formal acknowledgement and neither creates an E02 execution database.**
- [x] **Step 4: Run Bash syntax, container contract, and entrypoint tests.**

### Task 5: Deterministic Handoff Bundle and Documentation

**Files:**
- Modify: `scripts/prepare_stage2_v1_server_bundle.py`
- Modify: `tests/stage2/test_bundle.py`
- Create: `docs/research/e02_fresh_server_handoff_v1.md`
- Modify: `docs/research/stage2_e02_server_handoff_v1.md`

**Interfaces:**
- Produces: refreshed `artifacts/stage2/server-bundles/tarca-stage2-v1-server.tar.gz`, `.sha256`, and `.receipt.json`

- [x] **Step 1: Write tests requiring the E02 handoff config, both entrypoints, probe/preflight modules, and new handoff guide in the deterministic bundle.**
- [x] **Step 2: Run the bundle test and confirm the old package lacks the new contract.**
- [x] **Step 3: Add the guide and package inputs, then build the bundle twice and require byte-identical SHA-256 values.**
- [x] **Step 4: Verify archive and bundle sidecars, scan for secrets/formal outputs, run focused coverage, full pytest, Ruff, mypy, Bash syntax, and `git diff --check`.**
