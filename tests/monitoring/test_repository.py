from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tarca.execution import ResourceAllocation, ResourceRequest, RunPlanNode, ScientificIdentity
from tarca.execution.contracts import TaskSpec
from tarca.execution.state import ExecutionStateStore
from tarca.monitoring.repository import MonitoringRepository, open_readonly

SAMPLED_AT = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def test_repository_builds_only_explicit_runtime_views(monitoring_database: Path) -> None:
    snapshot = MonitoringRepository(
        monitoring_database,
        now=lambda: SAMPLED_AT,
    ).snapshot()

    assert snapshot.run.run_id == "run-a"
    assert snapshot.run.total_tasks == 2
    assert snapshot.run.pending_tasks == 1
    assert snapshot.run.last_sampled_at_utc == SAMPLED_AT
    assert snapshot.jobs[0].world_id == "lorenz96_f10_v2"
    assert snapshot.jobs[0].expected_cpu_cores == 4
    assert snapshot.jobs[0].actual_effective_busy_cores == 18.5
    assert snapshot.jobs[0].actual_vram_bytes == 18 * 1024**3
    assert snapshot.jobs[0].eta_seconds == 80.0
    assert snapshot.jobs[0].epoch == 3
    assert snapshot.jobs[1].state == "PENDING"
    assert {resource.label for resource in snapshot.resources} == {"主机", "GPU 0", "GPU 1"}
    assert {resource.telemetry_status for resource in snapshot.resources} == {"LIVE"}
    assert snapshot.alerts[0].category == "GPU_PRESSURE"


def _running_database(tmp_path: Path, *, completed_steps: int = 0) -> Path:
    path = tmp_path / "execution.sqlite3"
    store = ExecutionStateStore(path, artifact_verifier=lambda _: True)
    store.create_run("run-a", "graph-a")
    task = TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id="task-a",
            model_id="itransformer",
            data_id="world-a",
            seed=104729,
        ),
        phase="NEURAL_TRAIN",
        inputs=(),
        output_artifact_type="TEST_ARTIFACT",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=1,
            gpu_memory_gib=20.0,
            host_memory_gib=16.0,
        ),
    )
    store.register_run_plan(
        "run-a",
        (
            RunPlanNode(
                identity=task.identity,
                phase=task.phase,
                resource_request=task.resource_request,
                dependency_task_ids=(),
            ),
        ),
    )
    attempt_id = store.enqueue_task("run-a", task, "test.execute")
    allocation = ResourceAllocation(
        cpu_threads=4,
        gpu_ids=(0,),
        host_memory_gib_limit=16.0,
        worker_id="worker-a",
    )
    assert store.claim_attempt(attempt_id, "worker-a", allocation) is not None
    store.bind_running_process(
        attempt_id,
        "worker-a",
        987654,
        datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
        now=datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
    )
    store.record_progress(
        attempt_id,
        {"completed_steps": completed_steps, "total_steps": 100},
        now=SAMPLED_AT,
    )
    return path


def test_missing_telemetry_is_nullable_and_unavailable(tmp_path: Path) -> None:
    snapshot = MonitoringRepository(
        _running_database(tmp_path),
        now=lambda: SAMPLED_AT,
    ).snapshot()

    assert snapshot.run.last_sampled_at_utc is None
    assert snapshot.jobs[0].actual_effective_busy_cores is None
    assert snapshot.jobs[0].actual_rss_bytes is None
    assert snapshot.jobs[0].actual_vram_bytes is None
    assert {resource.telemetry_status for resource in snapshot.resources} == {"UNAVAILABLE"}
    assert all(resource.utilization_percent is None for resource in snapshot.resources)


def test_telemetry_freshness_preserves_real_values(monitoring_database: Path) -> None:
    live = MonitoringRepository(
        monitoring_database,
        now=lambda: datetime(2026, 8, 26, 12, 0, 9, tzinfo=UTC),
    ).snapshot()
    stale = MonitoringRepository(
        monitoring_database,
        now=lambda: datetime(2026, 8, 26, 12, 0, 11, tzinfo=UTC),
    ).snapshot()

    assert live.resources[0].actual_effective_busy_cores == 18.5
    assert {resource.telemetry_status for resource in live.resources} == {"LIVE"}
    assert stale.resources[0].actual_effective_busy_cores == 18.5
    assert {resource.telemetry_status for resource in stale.resources} == {"STALE"}


def test_zero_progress_keeps_eta_calibrating(tmp_path: Path) -> None:
    snapshot = MonitoringRepository(
        _running_database(tmp_path, completed_steps=0),
        now=lambda: SAMPLED_AT,
    ).snapshot()

    assert snapshot.jobs[0].eta_seconds is None
    assert snapshot.run.eta_seconds is None
    assert snapshot.run.eta_status == "CALIBRATING"


@pytest.mark.parametrize(
    ("completed_key", "total_key"),
    (
        ("completed_conditions", "total_conditions"),
        ("completed_base_groups", "total_base_groups"),
        ("completed_seed_worlds", "total_seed_worlds"),
    ),
)
def test_e01_progress_pairs_provide_running_eta(
    tmp_path: Path,
    completed_key: str,
    total_key: str,
) -> None:
    path = _running_database(tmp_path)
    store = ExecutionStateStore(path, artifact_verifier=lambda _: True)
    attempt_id = store.running_attempts("run-a")[0].attempt_id
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM progress_events WHERE attempt_id = ?", (attempt_id,))
    store.record_progress(
        attempt_id,
        {completed_key: 20, total_key: 100},
        now=SAMPLED_AT,
    )

    snapshot = MonitoringRepository(path, now=lambda: SAMPLED_AT).snapshot()

    assert snapshot.jobs[0].eta_seconds == 80.0
    assert snapshot.run.eta_status == "AVAILABLE"


def test_run_eta_uses_phase_and_resource_history_for_unseen_tail_tasks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "execution.sqlite3"
    store = ExecutionStateStore(path, artifact_verifier=lambda _: True)
    store.create_run("run-a", "graph-a")

    gpu_request = ResourceRequest(
        cpu_threads=2,
        gpu_count=1,
        gpu_memory_gib=8.0,
        host_memory_gib=16.0,
    )
    cpu_request = ResourceRequest(
        cpu_threads=2,
        gpu_count=0,
        gpu_memory_gib=0.0,
        host_memory_gib=8.0,
    )
    train_request = ResourceRequest(
        cpu_threads=4,
        gpu_count=1,
        gpu_memory_gib=20.0,
        host_memory_gib=32.0,
    )

    def identity(task_id: str, model_id: str) -> ScientificIdentity:
        return ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id=task_id,
            model_id=model_id,
            data_id="world-a",
            seed=104729,
        )

    plan = (
        RunPlanNode(
            identity=identity("freeze-history", "itransformer_reference"),
            phase="MODEL_FREEZE_CHECK",
            resource_request=gpu_request,
            dependency_task_ids=(),
        ),
        RunPlanNode(
            identity=identity("cpu-history", "model-none"),
            phase="OFFICIAL_REPRODUCTION",
            resource_request=cpu_request,
            dependency_task_ids=(),
        ),
        RunPlanNode(
            identity=identity("training", "itransformer_reference"),
            phase="NEURAL_TRAIN",
            resource_request=train_request,
            dependency_task_ids=(),
        ),
        RunPlanNode(
            identity=identity("freeze-pending", "patchtst_reference"),
            phase="MODEL_FREEZE_CHECK",
            resource_request=gpu_request,
            dependency_task_ids=("training",),
        ),
        RunPlanNode(
            identity=identity("receipt-pending", "model-none"),
            phase="QUALIFICATION_RECEIPT",
            resource_request=cpu_request,
            dependency_task_ids=("freeze-pending",),
        ),
    )
    store.register_run_plan("run-a", plan)

    def start(node: RunPlanNode, allocation: ResourceAllocation) -> str:
        task = TaskSpec(
            identity=node.identity,
            phase=node.phase,
            inputs=(),
            output_artifact_type="TEST_ARTIFACT",
            resource_request=node.resource_request,
        )
        attempt_id = store.enqueue_task("run-a", task, "test.execute")
        assert store.claim_attempt(attempt_id, allocation.worker_id, allocation) is not None
        store.bind_running_process(
            attempt_id,
            allocation.worker_id,
            987650 + len(attempt_id),
            datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
            now=datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
        )
        return attempt_id

    freeze_attempt = start(
        plan[0],
        ResourceAllocation(
            cpu_threads=2,
            gpu_ids=(0,),
            host_memory_gib_limit=16.0,
            worker_id="freeze-worker",
        ),
    )
    cpu_attempt = start(
        plan[1],
        ResourceAllocation(
            cpu_threads=2,
            gpu_ids=(),
            host_memory_gib_limit=8.0,
            worker_id="cpu-worker",
        ),
    )
    training_attempt = start(
        plan[2],
        ResourceAllocation(
            cpu_threads=4,
            gpu_ids=(1,),
            host_memory_gib_limit=32.0,
            worker_id="training-worker",
        ),
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE attempts SET state = 'COMPLETED', "
            "process_started_at_utc = ?, updated_at_utc = ? WHERE attempt_id = ?",
            (
                "2026-08-26T11:59:30+00:00",
                "2026-08-26T12:00:00+00:00",
                freeze_attempt,
            ),
        )
        connection.execute(
            "UPDATE attempts SET state = 'COMPLETED', "
            "process_started_at_utc = ?, updated_at_utc = ? WHERE attempt_id = ?",
            (
                "2026-08-26T11:59:40+00:00",
                "2026-08-26T12:00:00+00:00",
                cpu_attempt,
            ),
        )
    store.record_progress(
        training_attempt,
        {"completed_steps": 20, "total_steps": 100},
        now=SAMPLED_AT,
    )

    snapshot = MonitoringRepository(path, now=lambda: SAMPLED_AT).snapshot()
    jobs = {job.task_id: job for job in snapshot.jobs}

    assert jobs["freeze-pending"].eta_seconds == 30.0
    assert jobs["receipt-pending"].eta_seconds == 20.0
    assert snapshot.run.eta_status == "AVAILABLE"
    assert snapshot.run.eta_seconds == 130.0


def test_monitoring_connection_is_sqlite_read_only(monitoring_database: Path) -> None:
    with (
        open_readonly(monitoring_database) as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("DELETE FROM alerts")
