from __future__ import annotations

import os
from functools import partial
from pathlib import Path

from tarca.artifacts import LocalArtifactStore
from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash
from tarca.execution import ExecutionContext, ExecutorRegistry, ProgressSink, TaskSpec


def _artifact_root(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2").strip()
    root = (repository_root.resolve() / raw).resolve()
    if repository_root.resolve() not in root.parents:
        raise ValueError("Stage 2 artifact root must stay inside the repository")
    return root


def stage2_artifact_store(
    repository_root: Path, task: TaskSpec | None = None
) -> LocalArtifactStore:
    relative = (_artifact_root(repository_root) / "runtime/store").relative_to(
        repository_root.resolve()
    )
    return LocalArtifactStore(
        repository_root,
        producer_stage="stage2",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash="0" * 64 if task is None else canonical_json_hash(task.identity),
        dependencies=() if task is None else task.inputs,
        store_relative_root=relative.as_posix(),
    )


def _execute_phase(
    repository_root: Path,
    expected_phase: str,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del progress
    if task.phase != expected_phase:
        raise ValueError(f"executor expected {expected_phase}, received {task.phase}")
    payload = {
        "schema_version": "tarca-stage2-job-v1",
        "phase": task.phase,
        "task_id": task.task_id,
        "attempt_id": context.attempt_id,
        "scientific_identity_sha256": canonical_json_hash(task.identity),
        "input_artifacts": [item.model_dump(mode="json") for item in task.inputs],
        "formal_access_event_count": 0,
    }
    return stage2_artifact_store(repository_root, task).publish_bytes(
        canonical_json_bytes(payload) + b"\n",
        task.output_artifact_type,
        "application/json",
        "1.0.0",
    )


def stage2_executor_registry(repository_root: Path) -> ExecutorRegistry:
    root = repository_root.resolve()
    phases = {
        "stage2.verify_upstream": "UPSTREAM_VERIFY",
        "stage2.verify_source": "SOURCE_VERIFY",
        "stage2.generate_development_data": "DEV_DATA",
        "stage2.fit_baseline": "BASELINE_FIT",
        "stage2.train_neural": "NEURAL_TRAIN",
        "stage2.validate_checkpoint": "CHECKPOINT_VALIDATE",
        "stage2.predict_validation": "VALIDATION_PREDICT",
        "stage2.select_model": "MODEL_SELECT",
        "stage2.freeze_candidate": "FREEZE_CANDIDATE",
        "stage2.publish_receipt": "STAGE2_RECEIPT",
    }
    return ExecutorRegistry(
        {key: partial(_execute_phase, root, phase) for key, phase in phases.items()}
    )
