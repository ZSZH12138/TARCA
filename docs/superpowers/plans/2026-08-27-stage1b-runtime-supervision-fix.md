# Stage1B Runtime Supervision Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` in inline mode. TARCA project rules prohibit subagents, so every step must be executed by the current primary agent.

**Goal:** Make Stage1B scheduling account for already-running work and make the read-only dashboard display real runtime telemetry and ETA.

**Architecture:** Persist an immutable 74-node monitoring plan beside the execution state, expose active allocations as the source of truth for admission, and run a two-second supervisor loop that records psutil/NVML samples. The monitoring projection joins planned nodes, attempts, progress, and telemetry to produce honest nullable values and calibrated ETA; the React dashboard renders live, stale, and unavailable states explicitly.

**Tech Stack:** Python 3.10, Pydantic, SQLite WAL, psutil, nvidia-ml-py, PyTorch, FastAPI/WebSocket, React 19, TypeScript, Vitest, Playwright, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-27-stage1b-runtime-supervision-fix-design.md`

## Global Constraints

- Work only on `codex/stage1b-runtime-supervision-fix`.
- Execute inline; never create or use subagents.
- Use `D:\software\MyAnaconda\envs\tarca-stage1b-runtime-py310\python.exe` with `PYTHONPATH=deploy/stage1b/py310;src` for server-runtime Python tests.
- Use Python 3.11 only for the existing Stage0 doctor/CLI matrix.
- Do not change worlds, data, seeds, models, training budgets, gates, E01, E02, or freeze semantics.
- Do not modify or stage `docs/auth/**`.
- Do not run full Stage1B, E01, E02, or create a frozen revision.
- Every production behavior change follows RED → GREEN → REFACTOR.
- Branch coverage must remain at least 80%.

---

### Task 1: Enforce an Active Resource Ledger Across Scheduler Ticks

**Files:**
- Modify: `src/tarca/execution/state.py`
- Modify: `src/tarca/execution/resources.py`
- Modify: `src/tarca/execution/scheduler.py`
- Modify: `src/tarca/execution/__init__.py`
- Test: `tests/execution/test_state.py`
- Test: `tests/execution/test_resources.py`
- Test: `tests/execution/test_scheduler.py`

**Interfaces:**
- Produces: immutable `RunningAttempt` with `run_id`, `attempt_id`, `task`, `allocation`, `pid`, and process timestamps.
- Produces: `ExecutionStateStore.running_attempts(run_id) -> tuple[RunningAttempt, ...]`.
- Changes: `plan_resources(tasks, capacity, policy=None, *, active=())` subtracts every active request/allocation before admitting queued work.
- Consumes: `Scheduler.tick()` passes state-backed active attempts into `plan_resources()` on every tick.

- [ ] **Step 1: Add the repeated-tick regression test**

Add a scheduler test that enqueues eight independent 20 GiB single-GPU tasks on a two-card host, calls `tick()` five times while the backend keeps earlier workers running, and asserts:

```python
assert launches_per_tick == (2, 0, 0, 0, 0)
assert store.run_attempt_counts("run-a") == {"READY": 6, "RUNNING": 2}
assert {launch.task.allocation.gpu_ids for launch in first} == {(0,), (1,)}
```

Name the test `test_repeated_ticks_never_reuse_resources_held_by_running_attempts`.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
$env:PYTHONPATH='deploy/stage1b/py310;src'
& 'D:\software\MyAnaconda\envs\tarca-stage1b-runtime-py310\python.exe' -m pytest tests/execution/test_scheduler.py::test_repeated_ticks_never_reuse_resources_held_by_running_attempts -q
```

Expected: FAIL because later ticks launch the remaining six tasks and reuse GPU 0/1.

- [ ] **Step 3: Add state and resource-ledger tests**

Add tests proving:

```python
running = store.running_attempts("run-a")
assert running[0].allocation.gpu_ids == (0,)
assert running[0].task.resource_request.gpu_memory_gib == 20.0
```

and:

```python
allocations = plan_resources(queued, capacity, active=active)
assert allocations == ()
```

Cover a 24-core active CPU task, active host-memory consumption, a missing allocation that fails closed, and release of one GPU after an attempt completes.

- [ ] **Step 4: Run the new state/resource tests and verify RED**

Run the named new tests only. Expected: FAIL because `RunningAttempt`, `running_attempts()`, and the `active` argument do not exist.

- [ ] **Step 5: Implement the minimal active ledger**

Add:

```python
@dataclass(frozen=True, slots=True)
class RunningAttempt:
    run_id: str
    attempt_id: str
    task: TaskSpec
    allocation: ResourceAllocation
    pid: int | None
    process_started_at_utc: datetime | None
    heartbeat_at_utc: datetime | None
```

`running_attempts()` must query only the latest `RUNNING` attempt for the requested run and reject a running row without a valid allocation. `plan_resources()` must start from remaining CPU, memory, and GPU IDs after active allocations; any GPU with an active task is unavailable for a new task. The queued-task greedy order remains deterministic.

- [ ] **Step 6: Verify GREEN and refactor**

Run all execution state/resource/scheduler tests. Confirm repeated ticks remain at two active GPU tasks and one released GPU immediately accepts exactly one queued task.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/tarca/execution tests/execution
git commit -m "fix: account for active stage1b resource allocations"
```

---

### Task 2: Persist the Full 74-Node Runtime Monitoring Plan

**Files:**
- Modify: `src/tarca/execution/contracts.py`
- Modify: `src/tarca/execution/state.py`
- Modify: `src/tarca/execution/__init__.py`
- Modify: `src/tarca/stage1b/runner.py`
- Test: `tests/execution/test_contracts.py`
- Test: `tests/execution/test_state.py`
- Test: `tests/stage1b/test_runner_integration.py`

**Interfaces:**
- Produces: frozen `RunPlanNode(task_id, phase, identity, resource_request, dependency_task_ids)`.
- Produces: `ExecutionStateStore.register_run_plan(run_id, nodes)` with immutable re-registration semantics.
- Produces: `ExecutionStateStore.planned_task_count(run_id) -> int`.
- Consumes: `run_scheduled_qualification()` registers all compiled graph nodes before the first ready manifest is enqueued.

- [ ] **Step 1: Write plan-contract and state tests**

Tests must reject duplicate task IDs, unknown/self dependencies, dependency cycles, and re-registering the same run with drifted node content. A valid two-node plan must round-trip with count two.

- [ ] **Step 2: Run the plan tests and verify RED**

Expected: FAIL because the plan contract/table/API do not exist.

- [ ] **Step 3: Implement immutable run-plan storage**

Add a `run_plan_nodes` table keyed by `(run_id, task_id)` containing canonical identity, resource request, phase, and dependency IDs. Register in one transaction; exact re-registration is a no-op and any drift raises `ValueError`.

- [ ] **Step 4: Register the compiled Stage1B graph**

Immediately after `state.create_run(...)`, map every `Stage1BJobNode` to `RunPlanNode` and call `register_run_plan()`. Assert the registered count equals `len(graph.nodes)` and add an integration assertion for exactly 74 nodes.

- [ ] **Step 5: Verify GREEN**

Run contract/state/runner integration tests. Ensure existing enqueue/dependency behavior remains unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/tarca/execution src/tarca/stage1b/runner.py tests/execution tests/stage1b/test_runner_integration.py
git commit -m "feat: persist the complete stage1b runtime plan"
```

---

### Task 3: Record Real Host, Process, and GPU Telemetry

**Files:**
- Create: `src/tarca/execution/supervision.py`
- Modify: `src/tarca/execution/state.py`
- Modify: `src/tarca/execution/scheduler.py`
- Modify: `src/tarca/execution/__init__.py`
- Test: `tests/execution/test_supervision.py`
- Test: `tests/execution/test_scheduler.py`

**Interfaces:**
- Produces: `RuntimeSupervisor(store, probe, policy, clock=time.monotonic)`.
- Produces: `RuntimeSupervisor.sample_if_due(run_id, supervisor_pid) -> bool`.
- Produces: `ExecutionStateStore.add_alert_once(...)`.
- Changes: `Scheduler(..., supervisor=None)` invokes supervision without granting it access to scientific scores or gates.

- [ ] **Step 1: Write failing telemetry-supervisor tests**

With a fake probe returning non-zero values, assert one due sample creates:

```python
assert len(store.resource_samples("run-a", attempt_id=None)) == 1
assert len(store.resource_samples("run-a", attempt_id=attempt_id)) == 1
```

Assert a second call before two seconds writes nothing, a call after two seconds writes another sample, and a probe exception creates one deduplicated `TELEMETRY_UNAVAILABLE` alert without raising.

- [ ] **Step 2: Run and verify RED**

Expected: import failure for `tarca.execution.supervision`.

- [ ] **Step 3: Implement the supervisor**

The run-level sample uses the scheduler PID and contains global host/NVML values. Each running attempt with a live PID gets a process sample. Sampling is best-effort; all exceptions are converted to alerts. Store the `ResourceSample.sampled_at_utc` value, not a fabricated zero record.

- [ ] **Step 4: Integrate supervision with scheduler polling**

Call `sample_if_due()` once per scheduler loop after backend polling and after launches so the first sample observes newly started workers. Preserve the existing 0.2-second scheduler poll while the supervisor independently throttles itself to two seconds.

- [ ] **Step 5: Verify GREEN and no scientific data access**

Run supervision, scheduler, telemetry, and state tests. Assert the supervisor constructor accepts only execution state/probe/policy/clock and cannot inspect metrics, truth, or gate artifacts.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/tarca/execution tests/execution
git commit -m "feat: collect live stage1b runtime telemetry"
```

---

### Task 4: Project Honest ETA and Nullable Telemetry Through the API

**Files:**
- Modify: `src/tarca/monitoring/schemas.py`
- Modify: `src/tarca/monitoring/repository.py`
- Modify: `tests/monitoring/conftest.py`
- Modify: `tests/monitoring/test_repository.py`
- Modify: `tests/monitoring/test_api.py`

**Interfaces:**
- Changes: actual CPU/RAM/VRAM/utilization fields become nullable when no real sample exists.
- Adds: `ResourceView.telemetry_status: "LIVE" | "STALE" | "UNAVAILABLE"`.
- Adds: `RunSummaryView.last_sampled_at_utc: datetime | None`.
- Uses: running training progress `completed_steps`, `total_steps`, process start time, and progress timestamp for job ETA.
- Uses: full `run_plan_nodes` count for total/pending progress.

- [ ] **Step 1: Write failing monitoring projection tests**

Create three fixtures:

1. no resource sample → nullable actual values and `UNAVAILABLE`;
2. sample age under10 seconds → exact non-zero values and `LIVE`;
3. sample age over10 seconds → preserved last values and `STALE`.

Add a running training fixture with 20/100 steps and 20 seconds elapsed; assert job ETA is 80 seconds. Add a 0/100 fixture and assert ETA is `None` with run status `CALIBRATING`.

- [ ] **Step 2: Run and verify RED**

Expected: current repository returns zeros, no telemetry status, total count reflects only enqueued tasks, and ETA is always `None`.

- [ ] **Step 3: Implement plan-aware job projection**

Query every `run_plan_nodes` row and left-join its latest attempt, progress event, and attempt resource sample. A never-enqueued plan node is `PENDING`. Use `planned_task_count` for total and derive pending as total minus completed/running/failed.

- [ ] **Step 4: Implement telemetry freshness**

Use the latest run-level sample timestamp. Missing means `UNAVAILABLE`; at most10 seconds old means `LIVE`; older means `STALE`. Never convert missing actual values to numeric zero.

- [ ] **Step 5: Implement ETA**

For a running task with valid progress:

```python
fraction = completed_steps / total_steps
remaining = elapsed_seconds * (1.0 - fraction) / fraction
```

Reject non-finite, negative, zero-total, or completed-greater-than-total inputs. Completed tasks have ETA0. A run is `AVAILABLE` only when every currently running critical task has an ETA; otherwise it remains `CALIBRATING`. Run ETA is the maximum active task ETA plus known same-phase pending estimates; unknown future phases keep the run calibrating rather than inventing a number.

- [ ] **Step 6: Verify API read-only behavior and GREEN**

Run all monitoring tests. Confirm GET/WebSocket expose the new fields and POST/PUT/PATCH/DELETE still return405.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/tarca/monitoring tests/monitoring
git commit -m "feat: expose honest stage1b telemetry and eta"
```

---

### Task 5: Render Live, Stale, and Unavailable Data in the Frontend

**Files:**
- Modify: `frontend/stage1b-monitor/src/types.ts`
- Modify: `frontend/stage1b-monitor/src/components/ResourceGrid.tsx`
- Modify: `frontend/stage1b-monitor/src/components/RunSummary.tsx`
- Modify: `frontend/stage1b-monitor/src/components/JobTable.tsx`
- Modify: `frontend/stage1b-monitor/src/components/TelemetryCharts.tsx`
- Modify: `frontend/stage1b-monitor/src/format.ts`
- Modify: `frontend/stage1b-monitor/src/test/fixtures.ts`
- Modify: `frontend/stage1b-monitor/src/App.test.tsx`
- Modify: `frontend/stage1b-monitor/src/format.test.ts`
- Modify: `frontend/stage1b-monitor/e2e/snapshot.json`
- Modify: `frontend/stage1b-monitor/e2e/dashboard.spec.ts`

**Interfaces:**
- Consumes: nullable actual values, `telemetry_status`, `last_sampled_at_utc`, and calibrated ETA fields from Task4.
- Produces: visible Chinese labels `数据正常`, `数据过期`, `遥测不可用`, and last sample time.

- [ ] **Step 1: Write failing component tests**

Assert:

- real0% displays `0%` and `数据正常`;
- missing utilization displays `—` and `遥测不可用`;
- stale data preserves the last number but displays `数据过期`;
- charts use `null` for missing bars rather than0;
- run summary shows last sample time and ETA only when available.

- [ ] **Step 2: Run and verify RED**

Run `npm test -- --run`. Expected: type/test failures because current API fields are non-nullable and no status labels exist.

- [ ] **Step 3: Implement minimal UI changes**

Update types first, then render nullable values with `—`. Add status badges and last-sample text. Pass `null` to ECharts for unavailable metrics. Do not add any write controls.

- [ ] **Step 4: Update E2E evidence wording**

Keep the mocked UI E2E but rename the test to make the boundary explicit: `renders a live telemetry snapshot supplied by the monitoring API contract`. Backend state-to-API integration remains covered by Task4.

- [ ] **Step 5: Verify GREEN**

Run frontend unit tests, coverage, production build, and Playwright. Confirm branch coverage remains at least80%.

- [ ] **Step 6: Commit Task 5**

```bash
git add frontend/stage1b-monitor
git commit -m "feat: show truthful stage1b runtime telemetry"
```

---

### Task 6: Wire the Supervisor Into the Formal Stage1B Runtime

**Files:**
- Modify: `src/tarca/stage1b/runner.py`
- Modify: `scripts/run_stage1b_runtime.py`
- Modify: `tests/stage1b/test_runner_integration.py`
- Modify: `tests/stage1b/test_runtime_cli.py`
- Modify: `tests/stage1b/test_container_contract.py`

**Interfaces:**
- Consumes: `RuntimeSupervisor`, full run plan, and active-ledger scheduler.
- Produces: formal `launch` and `resume` runtime with real telemetry sampling and no duplicate GPU allocation.

- [ ] **Step 1: Write failing formal-runtime integration test**

Use the existing small compiled graph and fake telemetry probe/backend. Assert the formal runner registers the full plan, passes a supervisor into `Scheduler`, records at least one run sample, and never exceeds the synthetic two-GPU capacity across repeated ticks.

- [ ] **Step 2: Run and verify RED**

Expected: runner constructs `Scheduler` without a supervisor and has no run-plan registration.

- [ ] **Step 3: Implement formal wiring**

Construct `PsutilNvmlTelemetryProbe` once, create `RuntimeSupervisor` with a two-second policy, and pass it to `Scheduler`. The monitoring uvicorn process remains read-only and separate. Do not alter qualification graph identity or task inputs.

- [ ] **Step 4: Verify GREEN**

Run runtime CLI, runner integration, container contract, scheduler, supervision, and monitoring tests together.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/tarca/stage1b/runner.py scripts/run_stage1b_runtime.py tests/stage1b
git commit -m "fix: supervise the formal stage1b runtime"
```

---

### Task 7: Update Server Handoff and Complete Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/research/stage1b_v2_official_runtime_build_report.md`

**Interfaces:**
- Documents: exact concurrency behavior, live/stale/unavailable semantics, SSH tunnel, safe launch, and the fact that full Stage1B remains unexecuted.

- [ ] **Step 1: Update documentation**

State that neural tasks run one per GPU, active allocations are deducted across ticks, CPU admission is capped at24 data cores, telemetry samples every2 seconds, and ETA calibrates from real progress. Remove any statement implying aggressive GPU overpacking.

- [ ] **Step 2: Run focused regression suites**

Run all execution, monitoring, runtime CLI, runner integration, and frontend tests. Read full output and fix failures before broader verification.

- [ ] **Step 3: Run complete Python verification**

Run the Python3.10 server-compatible matrix with branch coverage at80%, Ruff format/lint, and mypy strict. Then run the Python3.11 Stage0 doctor/CLI matrix separately.

- [ ] **Step 4: Run complete frontend verification**

Run `npm ci`, unit tests, coverage, build, Playwright, and `npm audit --omit=dev`.

- [ ] **Step 5: Run container and security verification**

Run Compose config validation, rebuild the image, verify nonroot user/entrypoint, run `status --empty-ok`, audit the exact Python lock, scan for secret patterns, run `git diff --check`, and confirm `git diff -- docs/auth` is empty.

- [ ] **Step 6: Review the complete diff**

Confirm every design requirement maps to code/tests, no scientific config changed, the pre-existing untracked auth snapshot remains untouched, and no full Stage1B/E01/E02/freeze artifact was created.

- [ ] **Step 7: Commit Task 7**

```bash
git add README.md docs/research/stage1b_v2_official_runtime_build_report.md
git commit -m "docs: hand off the supervised stage1b runtime"
```

- [ ] **Step 8: Report the evidence**

Report exact test counts, coverage, container result, branch/commit, remaining server-only GPU preflight, and the truthful project state `BUILT_NOT_QUALIFIED / UNFROZEN`.
