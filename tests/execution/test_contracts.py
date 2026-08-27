from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tarca.contracts import ArtifactRef
from tarca.execution import (
    ExecutionContext,
    ExecutionPlan,
    MonitoringSnapshot,
    PlannedTask,
    ResourceAllocation,
    ResourceRequest,
    RunPlanNode,
    ScientificIdentity,
    TaskManifest,
    TaskResult,
    TaskSpec,
    TaskState,
)


def _artifact(marker: str = "a") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{marker}",
        artifact_type="test_output",
        content_hash=marker * 64,
        schema_version="2.0.0",
        relative_path=f"outputs/{marker}.json",
    )


def _identity(marker: str = "a") -> ScientificIdentity:
    return ScientificIdentity(
        protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
        experiment_id="stage1b-qualification-v2",
        task_id=f"task-{marker}",
        model_id="model-none",
        data_id="world-a",
        seed=104729,
    )


def _task(marker: str = "a") -> TaskSpec:
    return TaskSpec(
        identity=_identity(marker),
        phase="DATA_GENERATE",
        inputs=(),
        output_artifact_type="test_output",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=8.0,
        ),
    )


def test_completed_policy_is_never_rerun() -> None:
    manifest = TaskManifest(
        manifest_id="manifest-a",
        tasks=(_task(),),
        completed_task_policy="NEVER_RERUN",
    )
    assert manifest.completed_task_policy == "NEVER_RERUN"
    with pytest.raises(ValidationError):
        TaskManifest(
            manifest_id="manifest-b",
            tasks=(_task(),),
            completed_task_policy="RERUN",
        )


def test_completed_result_requires_artifact_and_other_states_reject_one() -> None:
    with pytest.raises(ValidationError, match="COMPLETED"):
        TaskResult(
            task_id="task-a",
            attempt_id="attempt-a",
            state=TaskState.COMPLETED,
            artifact=None,
        )
    completed = TaskResult(
        task_id="task-a",
        attempt_id="attempt-a",
        state=TaskState.COMPLETED,
        artifact=_artifact(),
    )
    assert completed.artifact is not None
    with pytest.raises(ValidationError, match="only COMPLETED"):
        TaskResult(
            task_id="task-a",
            attempt_id="attempt-a",
            state=TaskState.FAILED,
            artifact=_artifact(),
        )


def test_runtime_allocation_does_not_change_scientific_identity() -> None:
    identity = _identity()
    first = PlannedTask(
        task_id=identity.task_id,
        attempt_id="attempt-1",
        executor_key="stage1b.generate_data",
        allocation=ResourceAllocation(
            cpu_threads=4,
            gpu_ids=(),
            host_memory_gib_limit=8.0,
            worker_id="worker-1",
        ),
        input_refs=(),
        expected_output_artifact_type="test_output",
    )
    second = first.model_copy(
        update={
            "allocation": ResourceAllocation(
                cpu_threads=8,
                gpu_ids=(),
                host_memory_gib_limit=16.0,
                worker_id="worker-2",
            )
        }
    )
    assert first.task_id == second.task_id == identity.task_id


def test_execution_contracts_are_strict_and_monitoring_maps_are_immutable() -> None:
    with pytest.raises(ValidationError):
        ExecutionContext(
            run_id="run-a",
            task_id="task-a",
            attempt_id="attempt-a",
            runtime_identity="runtime-a",
            worker_identity="worker-a",
            scientific_override="forbidden",
        )
    snapshot = MonitoringSnapshot(
        phase="NEURAL_TRAIN",
        terminal_status=None,
        task_counts={"RUNNING": 2},
        resource_summary={"cpu_percent": 75.0},
        heartbeat_age_seconds=1.5,
        eta_status="CALIBRATING",
    )
    with pytest.raises(TypeError):
        snapshot.task_counts["FAILED"] = 1  # type: ignore[index]
    assert snapshot.model_dump(mode="json")["task_counts"] == {"RUNNING": 2}


def test_execution_plan_rejects_duplicate_gpu_ids_and_task_ids() -> None:
    with pytest.raises(ValidationError, match="GPU IDs"):
        ResourceAllocation(
            cpu_threads=4,
            gpu_ids=(0, 0),
            host_memory_gib_limit=8.0,
            worker_id="worker-a",
        )
    planned = PlannedTask(
        task_id="task-a",
        attempt_id="attempt-a",
        executor_key="stage1b.generate_data",
        allocation=ResourceAllocation(
            cpu_threads=4,
            gpu_ids=(0,),
            host_memory_gib_limit=8.0,
            worker_id="worker-a",
        ),
        input_refs=(),
        expected_output_artifact_type="test_output",
    )
    with pytest.raises(ValidationError, match="task IDs"):
        ExecutionPlan(
            plan_id="plan-a",
            task_manifest_id="manifest-a",
            backend_id="local",
            planned_tasks=(planned, planned.model_copy(update={"attempt_id": "attempt-b"})),
            max_concurrency=2,
            resource_snapshot_hash="b" * 64,
            created_at=datetime(2026, 8, 26, tzinfo=UTC),
        )


def test_run_plan_node_rejects_self_or_duplicate_dependencies() -> None:
    with pytest.raises(ValidationError, match="depend on itself"):
        RunPlanNode(
            identity=_identity(),
            phase="DATA_GENERATE",
            resource_request=_task().resource_request,
            dependency_task_ids=("task-a",),
        )
    with pytest.raises(ValidationError, match="unique"):
        RunPlanNode(
            identity=_identity(),
            phase="DATA_GENERATE",
            resource_request=_task().resource_request,
            dependency_task_ids=("parent", "parent"),
        )
