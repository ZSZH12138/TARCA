from __future__ import annotations

import os
from functools import partial
from pathlib import Path

from tarca.artifacts import LocalArtifactStore
from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash
from tarca.execution import ExecutionContext, ExecutorRegistry, ProgressSink, TaskSpec


def _root(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_E02_ARTIFACT_ROOT", "artifacts/e02").strip()
    root = (repository_root.resolve() / raw).resolve()
    if repository_root.resolve() not in root.parents:
        raise ValueError("E02 artifact root must stay inside the repository")
    return root


def e02_artifact_store(repository_root: Path, task: TaskSpec | None = None) -> LocalArtifactStore:
    relative = (_root(repository_root) / "runtime/store").relative_to(repository_root.resolve())
    return LocalArtifactStore(
        repository_root,
        producer_stage="e02",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash="0" * 64 if task is None else canonical_json_hash(task.identity),
        dependencies=() if task is None else task.inputs,
        store_relative_root=relative.as_posix(),
    )


def _execute(
    repository_root: Path,
    expected_phase: str,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del progress
    if task.phase != expected_phase:
        raise ValueError(f"executor expected {expected_phase}, received {task.phase}")
    # E02 progress deliberately contains no score, rank, truth, or gate fields.
    payload = {
        "schema_version": "tarca-e02-job-v1",
        "phase": task.phase,
        "task_id": task.task_id,
        "attempt_id": context.attempt_id,
        "scientific_identity_sha256": canonical_json_hash(task.identity),
        "input_artifacts": [item.model_dump(mode="json") for item in task.inputs],
    }
    return e02_artifact_store(repository_root, task).publish_bytes(
        canonical_json_bytes(payload) + b"\n",
        task.output_artifact_type,
        "application/json",
        "1.0.0",
    )


def e02_executor_registry(repository_root: Path) -> ExecutorRegistry:
    root = repository_root.resolve()
    phases = {
        "e02.verify_grant": "GRANT_VERIFY",
        "e02.verify_stage2": "STAGE2_VERIFY",
        "e02.open_formal": "FORMAL_OPEN",
        "e02.predict_formal": "FORMAL_PREDICT",
        "e02.score_trajectories": "TRAJECTORY_SCORE",
        "e02.bootstrap": "PAIRED_BOOTSTRAP",
        "e02.decide": "E02_DECISION",
        "e02.publish_receipt": "E02_RECEIPT",
    }
    return ExecutorRegistry({key: partial(_execute, root, phase) for key, phase in phases.items()})
