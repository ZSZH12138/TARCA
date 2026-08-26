from __future__ import annotations

from pathlib import Path

import pytest

from tarca.contracts import ArtifactRef
from tarca.execution import (
    ExecutionContext,
    ResourceRequest,
    ScientificIdentity,
    TaskSpec,
    TaskState,
)
from tarca.execution.registry import ExecutorRegistry
from tarca.execution.state import AttemptState, ExecutionStateStore
from tarca.execution.worker import run_worker


def _task(executor_task_id: str = "task-a") -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
            experiment_id="stage1b-qualification-v2",
            task_id=executor_task_id,
            model_id="model-none",
            data_id="world-a",
            seed=104729,
        ),
        phase="DATA_GENERATE",
        inputs=(),
        output_artifact_type="test_output",
        resource_request=ResourceRequest(
            cpu_threads=2,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=4.0,
        ),
    )


def _artifact(artifact_type: str = "test_output") -> ArtifactRef:
    return ArtifactRef(
        artifact_id="artifact-a",
        artifact_type=artifact_type,
        content_hash="a" * 64,
        schema_version="2.0.0",
        relative_path="outputs/a.json",
    )


def _running_store(
    tmp_path: Path,
    *,
    executor_key: str = "stage1b.test",
    verifier_result: bool = True,
) -> tuple[ExecutionStateStore, ExecutionContext]:
    store = ExecutionStateStore(
        tmp_path / "execution.sqlite3",
        artifact_verifier=lambda _: verifier_result,
    )
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(), executor_key)
    claim = store.claim_ready("worker-a", limit=1)[0]
    context = ExecutionContext(
        run_id="run-a",
        task_id="task-a",
        attempt_id=claim.attempt_id,
        runtime_identity="runtime-a",
        worker_identity="worker-a",
    )
    return store, context


def test_registry_is_an_immutable_explicit_allowlist() -> None:
    registry = ExecutorRegistry({"stage1b.test": lambda task, context, progress: _artifact()})
    assert callable(registry.resolve("stage1b.test"))
    with pytest.raises(ValueError, match="allowlisted"):
        registry.resolve("subprocess:rm")
    with pytest.raises(ValueError, match="registry identifier"):
        ExecutorRegistry({"python -c bad": lambda task, context, progress: _artifact()})


def test_worker_completes_only_after_artifact_verification(tmp_path: Path) -> None:
    store, context = _running_store(tmp_path)

    def executor(task: TaskSpec, _context: ExecutionContext, progress: object) -> ArtifactRef:
        progress.report({"completed_steps": 1, "total_steps": 1})  # type: ignore[attr-defined]
        return _artifact(task.output_artifact_type)

    result = run_worker(context, store, ExecutorRegistry({"stage1b.test": executor}))
    assert result.state is TaskState.COMPLETED
    assert result.artifact == _artifact()
    assert store.attempt_state(context.attempt_id) is AttemptState.COMPLETED
    assert len(store.progress_events(context.attempt_id)) == 1


def test_worker_fails_closed_on_unverified_or_wrong_artifact(tmp_path: Path) -> None:
    unverified_store, unverified_context = _running_store(
        tmp_path / "unverified",
        verifier_result=False,
    )
    unverified = run_worker(
        unverified_context,
        unverified_store,
        ExecutorRegistry({"stage1b.test": lambda task, context, progress: _artifact()}),
    )
    assert unverified.state is TaskState.FAILED
    assert unverified_store.attempt_error(unverified_context.attempt_id) == "HASH_DRIFT"

    wrong_store, wrong_context = _running_store(tmp_path / "wrong")
    wrong = run_worker(
        wrong_context,
        wrong_store,
        ExecutorRegistry({"stage1b.test": lambda task, context, progress: _artifact("wrong_type")}),
    )
    assert wrong.state is TaskState.FAILED
    assert wrong_store.attempt_error(wrong_context.attempt_id) == "IDENTITY_DRIFT"


def test_worker_never_resolves_an_executor_from_a_shell_string(tmp_path: Path) -> None:
    store, context = _running_store(tmp_path, executor_key="not.allowlisted")
    result = run_worker(
        context,
        store,
        ExecutorRegistry({"stage1b.test": lambda *args: _artifact()}),
    )
    assert result.state is TaskState.FAILED
    assert store.attempt_error(context.attempt_id) == "IDENTITY_DRIFT"
