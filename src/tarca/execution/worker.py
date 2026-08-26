from __future__ import annotations

from contextlib import suppress
from typing import NoReturn

import torch

from tarca.contracts import ArtifactRef
from tarca.execution.contracts import ExecutionContext, TaskResult, TaskState
from tarca.execution.registry import ExecutorRegistry, ProgressSink
from tarca.execution.state import ExecutionStateStore, StateTransitionConflict


class _StateProgressSink:
    def __init__(self, store: ExecutionStateStore, attempt_id: str) -> None:
        self._store = store
        self._attempt_id = attempt_id

    def report(self, progress: object) -> None:
        self._store.record_progress(self._attempt_id, progress)


def _failure_category(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "CUDA_OOM"
    if isinstance(error, OSError):
        return "TRANSIENT_IO"
    if "truth" in message or "leakage" in message:
        return "TRUTH_LEAKAGE"
    if "nan" in message or "nonfinite" in message or "non-finite" in message:
        return "NONFINITE"
    if "hash" in message or "verification" in message:
        return "HASH_DRIFT"
    if "identity" in message or "allowlisted" in message or "artifact type" in message:
        return "IDENTITY_DRIFT"
    return "WORKER_ERROR"


def _context_error(message: str) -> NoReturn:
    raise ValueError(f"execution context identity mismatch: {message}")


def run_worker(
    context: ExecutionContext,
    store: ExecutionStateStore,
    registry: ExecutorRegistry,
) -> TaskResult:
    claim = store.claimed_task(context.attempt_id)
    if claim.run_id != context.run_id:
        _context_error("run_id")
    if claim.task.task_id != context.task_id:
        _context_error("task_id")
    if claim.worker_id != context.worker_identity:
        _context_error("worker_identity")
    progress: ProgressSink = _StateProgressSink(store, context.attempt_id)
    try:
        executor = registry.resolve(claim.executor_key)
        artifact: ArtifactRef = executor(claim.task, context, progress)
        if artifact.artifact_type != claim.task.output_artifact_type:
            raise ValueError("executor artifact type violates task identity")
        store.complete_attempt(context.attempt_id, artifact)
        return TaskResult(
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            state=TaskState.COMPLETED,
            artifact=artifact,
        )
    except Exception as error:
        category = _failure_category(error)
        with suppress(StateTransitionConflict):
            store.fail_attempt(context.attempt_id, category)
        return TaskResult(
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            state=TaskState.FAILED,
            artifact=None,
        )
