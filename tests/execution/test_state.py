from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tarca.contracts import ArtifactRef
from tarca.execution import ResourceAllocation, ResourceRequest, ScientificIdentity, TaskSpec
from tarca.execution.state import (
    AttemptState,
    ExecutionStateStore,
    ProcessIdentity,
    RetryDisposition,
    StateTransitionConflict,
)


def _task(
    task_id: str = "task-a",
    output_type: str = "test_output",
    *,
    inputs: tuple[ArtifactRef, ...] = (),
) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
            experiment_id="stage1b-qualification-v2",
            task_id=task_id,
            model_id="model-none",
            data_id="world-a",
            seed=104729,
        ),
        phase="DATA_GENERATE",
        inputs=inputs,
        output_artifact_type=output_type,
        resource_request=ResourceRequest(
            cpu_threads=2,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=4.0,
        ),
    )


def _artifact(marker: str = "a", artifact_type: str = "test_output") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{marker}",
        artifact_type=artifact_type,
        content_hash=marker * 64,
        schema_version="2.0.0",
        relative_path=f"outputs/{marker}.json",
    )


def _store(tmp_path: Path) -> ExecutionStateStore:
    store = ExecutionStateStore(tmp_path / "execution.sqlite3", artifact_verifier=lambda _: True)
    store.create_run("run-a", "graph-a")
    return store


def test_store_uses_wal_foreign_keys_and_busy_timeout(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.pragma("journal_mode").lower() == "wal"
    assert store.pragma("foreign_keys") == 1
    assert store.pragma("busy_timeout") == 5000
    assert set(store.table_names()) >= {
        "runs",
        "job_nodes",
        "task_specs",
        "attempts",
        "dependencies",
        "progress_events",
        "resource_samples",
        "alerts",
    }


def test_completed_job_is_never_claimed_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.enqueue_task("run-a", _task(), "stage1b.generate_dataset")
    claim = store.claim_ready("worker-1", limit=1)[0]
    store.complete_attempt(claim.attempt_id, _artifact())
    assert store.claim_ready("worker-2", limit=1) == ()
    assert store.attempt_state(claim.attempt_id) is AttemptState.COMPLETED


def test_task_is_not_claimed_before_its_dependency_completes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent_attempt = store.enqueue_task(
        "run-a",
        _task("parent-task", "parent_output"),
        "stage1b.parent",
    )
    parent_artifact = _artifact("b", "parent_output")
    with pytest.raises(ValueError, match="dependency is not completed"):
        store.enqueue_task(
            "run-a",
            _task("child-task", inputs=(parent_artifact,)),
            "stage1b.child",
            dependency_task_ids=("parent-task",),
        )
    parent_claim = store.claim_ready("worker-1", limit=1)[0]
    assert parent_claim.attempt_id == parent_attempt
    store.complete_attempt(parent_attempt, parent_artifact)
    store.enqueue_task(
        "run-a",
        _task("child-task", inputs=(parent_artifact,)),
        "stage1b.child",
        dependency_task_ids=("parent-task",),
    )
    child_claim = store.claim_ready("worker-2", limit=1)[0]
    assert child_claim.task.task_id == "child-task"


def test_state_transition_is_compare_and_set(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt_id = store.enqueue_task("run-a", _task(), "stage1b.generate_dataset")
    store.transition(attempt_id, AttemptState.READY, AttemptState.RUNNING)
    with pytest.raises(StateTransitionConflict):
        store.transition(attempt_id, AttemptState.READY, AttemptState.RUNNING)


def test_concurrent_workers_cannot_claim_the_same_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.enqueue_task("run-a", _task(), "stage1b.generate_dataset")

    def claim(worker_id: str) -> tuple[str, ...]:
        return tuple(item.attempt_id for item in store.claim_ready(worker_id, limit=1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = tuple(pool.map(claim, ("worker-1", "worker-2")))
    claimed_ids = tuple(attempt_id for claim_ids in claims for attempt_id in claim_ids)
    assert claimed_ids == ("task-a-attempt-1",)


def test_ready_tasks_can_be_inspected_then_claimed_exactly_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt_id = store.enqueue_task("run-a", _task(), "test.execute")

    queued = store.ready_tasks("run-a")
    claim = store.claim_attempt(attempt_id, "worker-a")

    assert tuple(item.attempt_id for item in queued) == (attempt_id,)
    assert claim is not None and claim.task.task_id == "task-a"
    assert store.claim_attempt(attempt_id, "worker-b") is None


def test_run_status_and_completed_artifacts_are_read_only_views(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task()
    attempt_id = store.enqueue_task("run-a", task, "test.execute")
    assert store.claim_attempt(attempt_id, "worker-a") is not None
    artifact = _artifact()
    store.complete_attempt(attempt_id, artifact)

    assert store.run_attempt_counts("run-a") == {"COMPLETED": 1}
    assert store.completed_artifacts("run-a") == {"task-a": artifact}


def test_running_attempts_expose_frozen_task_and_allocation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task()
    attempt_id = store.enqueue_task("run-a", task, "test.execute")
    allocation = ResourceAllocation(
        cpu_threads=2,
        gpu_ids=(),
        host_memory_gib_limit=4.0,
        worker_id="worker-a",
    )
    assert store.claim_attempt(attempt_id, "worker-a", allocation) is not None

    running = store.running_attempts("run-a")

    assert len(running) == 1
    assert running[0].attempt_id == attempt_id
    assert running[0].task == task
    assert running[0].allocation == allocation


def test_running_attempt_without_allocation_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt_id = store.enqueue_task("run-a", _task(), "test.execute")
    assert store.claim_attempt(attempt_id, "worker-a") is not None

    with pytest.raises(RuntimeError, match="no committed resource allocation"):
        store.running_attempts("run-a")


class _Probe:
    def __init__(self, identities: dict[int, ProcessIdentity]) -> None:
        self.identities = identities

    def inspect(self, pid: int) -> ProcessIdentity | None:
        return self.identities.get(pid)


def test_restart_marks_only_dead_or_mismatched_workers_stalled(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    store = _store(tmp_path)
    live_attempt = store.enqueue_task("run-a", _task("live-task"), "stage1b.live")
    dead_attempt = store.enqueue_task("run-a", _task("dead-task"), "stage1b.dead")
    live_start = now - timedelta(minutes=5)
    dead_start = now - timedelta(minutes=4)
    store.bind_running_process(live_attempt, "worker-live", 101, live_start, now=now)
    store.bind_running_process(dead_attempt, "worker-dead", 202, dead_start, now=now)
    probe = _Probe(
        {
            101: ProcessIdentity(
                pid=101,
                process_started_at_utc=live_start,
                run_id="run-a",
                task_id="live-task",
            ),
            202: ProcessIdentity(
                pid=202,
                process_started_at_utc=dead_start,
                run_id="different-run",
                task_id="dead-task",
            ),
        }
    )
    result = store.reconcile_processes(probe, now=now)
    assert result.live_task_ids == ("live-task",)
    assert result.stalled_task_ids == ("dead-task",)


def test_expired_heartbeat_is_stalled_even_when_pid_exists(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    store = _store(tmp_path)
    attempt = store.enqueue_task("run-a", _task(), "stage1b.task")
    started = now - timedelta(minutes=5)
    store.bind_running_process(
        attempt,
        "worker-a",
        101,
        started,
        now=now - timedelta(seconds=20),
    )
    probe = _Probe(
        {
            101: ProcessIdentity(
                pid=101,
                process_started_at_utc=started,
                run_id="run-a",
                task_id="task-a",
            )
        }
    )
    result = store.reconcile_processes(probe, now=now, heartbeat_timeout_seconds=10.0)
    assert result.stalled_task_ids == ("task-a",)


def test_retry_policy_is_bounded_and_scientifically_safe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempt = store.enqueue_task("run-a", _task(), "stage1b.task")
    store.transition(attempt, AttemptState.READY, AttemptState.RUNNING)
    store.fail_attempt(attempt, "TRANSIENT_IO")
    retried = store.retry_attempt(attempt, "TRANSIENT_IO")
    assert retried is not None
    assert store.retry_disposition("TRANSIENT_IO") is RetryDisposition.RETRY_ONCE
    store.transition(retried, AttemptState.READY, AttemptState.RUNNING)
    store.fail_attempt(retried, "TRANSIENT_IO")
    assert store.retry_attempt(retried, "TRANSIENT_IO") is None

    terminal = store.enqueue_task("run-a", _task("terminal-task"), "stage1b.task")
    store.transition(terminal, AttemptState.READY, AttemptState.RUNNING)
    store.fail_attempt(terminal, "HASH_DRIFT")
    assert store.retry_attempt(terminal, "HASH_DRIFT") is None
    assert store.retry_disposition("HASH_DRIFT") is RetryDisposition.TERMINAL

    oom = store.enqueue_task("run-a", _task("oom-task"), "stage1b.task")
    store.transition(oom, AttemptState.READY, AttemptState.RUNNING)
    store.fail_attempt(oom, "CUDA_OOM")
    assert store.retry_attempt(oom, "CUDA_OOM", lower_packing_applied=False) is None
    assert store.retry_attempt(oom, "CUDA_OOM", lower_packing_applied=True) is not None
