from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import torch

from tarca.contracts import canonical_json_hash, sha256_file
from tarca.execution import (
    ExecutionStateStore,
    ResourceRequest,
    RunPlanNode,
    ScientificIdentity,
    TaskSpec,
)
from tarca.execution.state import AttemptState

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SPEC = ROOT / "configs/stage2/stage2_device_mismatch_recovery_v1.json"
SEEDS = (1797287582, 883082243, 1933050005)


def _seal(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "receipt_sha256": canonical_json_hash(payload)}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository_root = tmp_path / "repository"
    artifact_root = repository_root / "artifacts/stage2"
    checkpoint_root = artifact_root / "runtime/checkpoints"
    checkpoint_root.mkdir(parents=True)
    database = artifact_root / "runtime/execution.sqlite3"
    store = ExecutionStateStore(database)
    run_id = "run-" + "a" * 64
    graph_id = "stage2-graph-" + "b" * 64
    store.create_run(run_id, graph_id)
    tasks: list[TaskSpec] = []
    task_entries: list[dict[str, object]] = []
    for model_id in ("PATCHTST", "ITRANSFORMER"):
        for seed in SEEDS:
            task_digest = canonical_json_hash({"model_id": model_id, "seed": seed})
            task_id = f"stage2-neural-train-{task_digest}"
            identity = ScientificIdentity(
                protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
                experiment_id="stage2_probabilistic_forecasting_v1",
                task_id=task_id,
                model_id=model_id,
                data_id="lorenz96_twoscale_v2",
                seed=seed,
            )
            task = TaskSpec(
                identity=identity,
                phase="NEURAL_TRAIN",
                inputs=(),
                output_artifact_type="STAGE2_NEURAL_CHECKPOINT",
                resource_request=ResourceRequest(
                    cpu_threads=4,
                    gpu_count=1,
                    gpu_memory_gib=20.0,
                    host_memory_gib=32.0,
                ),
            )
            tasks.append(task)
            attempt_id = store.enqueue_task(run_id, task, "stage2.train_neural")
            store.transition(attempt_id, AttemptState.READY, AttemptState.RUNNING)
            store.fail_attempt(attempt_id, "WORKER_ERROR")
            checkpoint_task_sha256 = canonical_json_hash({"task_id": task_id})
            checkpoint_relative = (
                "artifacts/stage2/runtime/checkpoints/"
                f"training-{checkpoint_task_sha256}.pt"
            )
            checkpoint = repository_root / checkpoint_relative
            torch.save(
                {
                    "schema_version": "1.0.0",
                    "status": "COMPLETE",
                    "seed": seed,
                    "task_sha256": checkpoint_task_sha256,
                    "epoch": 41,
                },
                checkpoint,
            )
            task_entries.append(
                {
                    "task_id": task_id,
                    "source_attempt_id": attempt_id,
                    "model_id": model_id,
                    "seed": seed,
                    "checkpoint_relative_path": checkpoint_relative,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_task_sha256": checkpoint_task_sha256,
                    "checkpoint_epoch": 41,
                }
            )
    store.register_run_plan(
        run_id,
        tuple(
            RunPlanNode(
                identity=task.identity,
                phase=task.phase,
                resource_request=task.resource_request,
                dependency_task_ids=(),
            )
            for task in tasks
        ),
    )
    source_database_sha256 = sha256_file(database)
    archive_sha256 = "c" * 64
    source_manifest_sha256 = "d" * 64
    server_bundle_sha256 = "e" * 64
    spec = {
        "schema_version": "tarca-stage2-device-mismatch-recovery-spec-v1",
        "recovery_id": "stage2-device-mismatch-recovery-v1",
        "reason": "DEVICE_MISMATCH_V1",
        "source_archive_filename": "recovery.tar.gz",
        "source_archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_database_sha256": source_database_sha256,
        "run_id": run_id,
        "graph_id": graph_id,
        "scientific_config_sha256": "f" * 64,
        "planned_task_count": 6,
        "completed_attempt_count": 0,
        "failed_attempt_count": 6,
        "source_attempt_number": 1,
        "source_attempt_state": "FAILED",
        "source_error_category": "WORKER_ERROR",
        "target_attempt_number": 2,
        "checkpoint_policy": "COMPLETE_ZERO_TRAINING_STEPS_NO_REWRITE",
        "tasks": task_entries,
    }
    spec_path = repository_root / "configs/stage2/recovery.json"
    _write_json(spec_path, spec)
    input_payload = {
        "schema_version": "tarca-stage2-recovery-input-v1",
        "status": "RESTORED",
        "source_archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_database_sha256": source_database_sha256,
        "server_bundle_sha256": server_bundle_sha256,
    }
    input_receipt = artifact_root / "runtime/recovery_input_receipt.json"
    _write_json(input_receipt, _seal(input_payload))
    return repository_root, artifact_root, spec_path, input_receipt


def test_repository_recovery_spec_is_exactly_narrow() -> None:
    from tarca.stage2.recovery import load_stage2_recovery_spec

    spec = load_stage2_recovery_spec(RECOVERY_SPEC)

    assert spec.source_archive_sha256 == (
        "79c6cb2c0f8fd8a1801d378fb779212b66f3774d8372df7b360b1721b3f9b126"
    )
    assert spec.run_id == (
        "run-acff24d96653a25d4aac54b9389c605d8c35293cc930f9fa8a560947306401fb"
    )
    assert spec.planned_task_count == 37
    assert spec.completed_attempt_count == 16
    assert len(spec.tasks) == 6
    assert {(task.model_id, task.seed) for task in spec.tasks} == {
        (model_id, seed)
        for model_id in ("PATCHTST", "ITRANSFORMER")
        for seed in SEEDS
    }


def test_authorized_recovery_preserves_failures_and_appends_attempt_two(
    tmp_path: Path,
) -> None:
    from tarca.stage2.recovery import authorize_stage2_device_mismatch_recovery

    root, artifacts, spec_path, input_receipt = _recovery_fixture(tmp_path)
    first = authorize_stage2_device_mismatch_recovery(
        root,
        artifacts,
        spec_path=spec_path,
        recovery_input_receipt_path=input_receipt,
    )
    second = authorize_stage2_device_mismatch_recovery(
        root,
        artifacts,
        spec_path=spec_path,
        recovery_input_receipt_path=input_receipt,
    )

    assert first == second
    assert first["status"] == "AUTHORIZED"
    database = artifacts / "runtime/execution.sqlite3"
    with sqlite3.connect(database) as connection:
        source_rows = connection.execute(
            "SELECT state, error_category FROM attempts WHERE attempt_number = 1"
        ).fetchall()
        target_rows = connection.execute(
            "SELECT state, error_category FROM attempts WHERE attempt_number = 2"
        ).fetchall()
        events = connection.execute("SELECT COUNT(*) FROM recovery_events").fetchone()
        alerts = connection.execute(
            "SELECT category FROM alerts ORDER BY alert_id"
        ).fetchall()
    assert source_rows == [("FAILED", "WORKER_ERROR")] * 6
    assert target_rows == [("READY", None)] * 6
    assert events == (6,)
    assert alerts == [("CONTROLLED_RECOVERY_AUTHORIZED",)]


def test_recovery_rejects_checkpoint_drift_before_mutating_database(tmp_path: Path) -> None:
    from tarca.stage2.recovery import (
        Stage2RecoveryRejected,
        authorize_stage2_device_mismatch_recovery,
        load_stage2_recovery_spec,
    )

    root, artifacts, spec_path, input_receipt = _recovery_fixture(tmp_path)
    spec = load_stage2_recovery_spec(spec_path)
    checkpoint = root / spec.tasks[0].checkpoint_relative_path
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(Stage2RecoveryRejected, match="checkpoint SHA-256"):
        authorize_stage2_device_mismatch_recovery(
            root,
            artifacts,
            spec_path=spec_path,
            recovery_input_receipt_path=input_receipt,
        )

    with sqlite3.connect(artifacts / "runtime/execution.sqlite3") as connection:
        attempts = connection.execute(
            "SELECT attempt_number, state FROM attempts ORDER BY attempt_id"
        ).fetchall()
    assert attempts == [(1, "FAILED")] * 6
