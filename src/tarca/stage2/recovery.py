from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

import torch
from pydantic import field_validator, model_validator

from tarca.contracts import (
    Sha256Hash,
    StrictContractModel,
    canonical_json_bytes,
    canonical_json_hash,
    sha256_file,
)
from tarca.stage2.config import load_stage2_config


class Stage2RecoveryRejected(RuntimeError):
    """Raised before an unsafe or ambiguous Stage 2 recovery mutation."""


class Stage2RecoveryTask(StrictContractModel):
    task_id: str
    source_attempt_id: str
    model_id: Literal["PATCHTST", "ITRANSFORMER"]
    seed: int
    checkpoint_relative_path: str
    checkpoint_sha256: Sha256Hash
    checkpoint_task_sha256: Sha256Hash
    checkpoint_epoch: int

    @field_validator("task_id", "source_attempt_id")
    @classmethod
    def _identifier_is_safe(cls, value: str) -> str:
        if not value or any(marker in value for marker in ("/", "\\", "\x00")):
            raise ValueError("recovery task identifiers must be safe")
        return value

    @field_validator("checkpoint_relative_path")
    @classmethod
    def _checkpoint_path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("recovery checkpoint path must be repository-relative")
        if path.parts[:4] != ("artifacts", "stage2", "runtime", "checkpoints"):
            raise ValueError("recovery checkpoint path is outside the checkpoint root")
        return value

    @model_validator(mode="after")
    def _task_binding_is_exact(self) -> Self:
        if self.source_attempt_id != f"{self.task_id}-attempt-1":
            raise ValueError("recovery source attempt does not match its task")
        expected_name = f"training-{self.checkpoint_task_sha256}.pt"
        if PurePosixPath(self.checkpoint_relative_path).name != expected_name:
            raise ValueError("recovery checkpoint filename does not match its task hash")
        if self.checkpoint_epoch < 1:
            raise ValueError("recovery checkpoint epoch must be positive")
        return self


class Stage2RecoverySpec(StrictContractModel):
    schema_version: Literal["tarca-stage2-device-mismatch-recovery-spec-v1"]
    recovery_id: Literal["stage2-device-mismatch-recovery-v1"]
    reason: Literal["DEVICE_MISMATCH_V1"]
    source_archive_filename: str
    source_archive_sha256: Sha256Hash
    source_manifest_sha256: Sha256Hash
    source_database_sha256: Sha256Hash
    run_id: str
    graph_id: str
    scientific_config_sha256: Sha256Hash
    planned_task_count: int
    completed_attempt_count: int
    failed_attempt_count: Literal[6]
    source_attempt_number: Literal[1]
    source_attempt_state: Literal["FAILED"]
    source_error_category: Literal["WORKER_ERROR"]
    target_attempt_number: Literal[2]
    checkpoint_policy: Literal["COMPLETE_ZERO_TRAINING_STEPS_NO_REWRITE"]
    tasks: tuple[Stage2RecoveryTask, ...]

    @model_validator(mode="after")
    def _scope_is_exact(self) -> Self:
        if len(self.tasks) != 6:
            raise ValueError("device-mismatch recovery requires exactly six tasks")
        task_ids = tuple(task.task_id for task in self.tasks)
        attempts = tuple(task.source_attempt_id for task in self.tasks)
        if len(set(task_ids)) != 6 or len(set(attempts)) != 6:
            raise ValueError("device-mismatch recovery tasks must be unique")
        expected_pairs = {
            (model_id, seed)
            for model_id in ("PATCHTST", "ITRANSFORMER")
            for seed in (1797287582, 883082243, 1933050005)
        }
        if {(task.model_id, task.seed) for task in self.tasks} != expected_pairs:
            raise ValueError("device-mismatch recovery model and seed scope drifted")
        if self.planned_task_count < 6 or self.completed_attempt_count < 0:
            raise ValueError("device-mismatch recovery task counts are invalid")
        return self


class Stage2RecoveryInputReceipt(StrictContractModel):
    schema_version: Literal["tarca-stage2-recovery-input-v1"]
    status: Literal["RESTORED"]
    source_archive_sha256: Sha256Hash
    source_manifest_sha256: Sha256Hash
    source_database_sha256: Sha256Hash
    server_bundle_sha256: Sha256Hash
    restored_file_count: int | None = None
    receipt_sha256: Sha256Hash

    @model_validator(mode="after")
    def _receipt_is_sealed(self) -> Self:
        payload = self.model_dump(
            mode="json", exclude={"receipt_sha256"}, exclude_none=True
        )
        if self.receipt_sha256 != canonical_json_hash(payload):
            raise ValueError("recovery input receipt SHA-256 does not match")
        return self


class Stage2RecoveryAuthorizationReceipt(StrictContractModel):
    schema_version: Literal["tarca-stage2-recovery-authorization-v1"]
    status: Literal["AUTHORIZED"]
    recovery_id: str
    reason: Literal["DEVICE_MISMATCH_V1"]
    run_id: str
    graph_id: str
    spec_sha256: Sha256Hash
    source_database_sha256: Sha256Hash
    server_bundle_sha256: Sha256Hash
    source_attempt_ids: tuple[str, ...]
    new_attempt_ids: tuple[str, ...]
    checkpoint_sha256: tuple[Sha256Hash, ...]
    created_at_utc: str
    receipt_sha256: Sha256Hash

    @model_validator(mode="after")
    def _receipt_is_sealed(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != canonical_json_hash(payload):
            raise ValueError("recovery authorization receipt SHA-256 does not match")
        if not (
            len(self.source_attempt_ids)
            == len(self.new_attempt_ids)
            == len(self.checkpoint_sha256)
            == 6
        ):
            raise ValueError("recovery authorization receipt must bind six tasks")
        return self


def load_stage2_recovery_spec(path: Path) -> Stage2RecoverySpec:
    try:
        return Stage2RecoverySpec.model_validate_json(path.resolve().read_text(encoding="utf-8"))
    except Exception as error:
        raise Stage2RecoveryRejected(f"Stage 2 recovery spec is invalid: {error}") from error


def load_stage2_recovery_input_receipt(path: Path) -> Stage2RecoveryInputReceipt:
    try:
        return Stage2RecoveryInputReceipt.model_validate_json(
            path.resolve().read_text(encoding="utf-8")
        )
    except Exception as error:
        raise Stage2RecoveryRejected(f"recovery input receipt is invalid: {error}") from error


def _load_input_receipt(path: Path, spec: Stage2RecoverySpec) -> Stage2RecoveryInputReceipt:
    receipt = load_stage2_recovery_input_receipt(path)
    if (
        receipt.source_archive_sha256 != spec.source_archive_sha256
        or receipt.source_manifest_sha256 != spec.source_manifest_sha256
        or receipt.source_database_sha256 != spec.source_database_sha256
    ):
        raise Stage2RecoveryRejected("recovery input receipt does not match the frozen spec")
    return receipt


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".stage2-recovery-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_checkpoint(root: Path, task: Stage2RecoveryTask) -> None:
    checkpoint = (root / task.checkpoint_relative_path).resolve()
    expected_root = (root / "artifacts/stage2/runtime/checkpoints").resolve()
    try:
        checkpoint.relative_to(expected_root)
    except ValueError as error:
        raise Stage2RecoveryRejected("checkpoint escapes the Stage 2 checkpoint root") from error
    if not checkpoint.is_file() or sha256_file(checkpoint) != task.checkpoint_sha256:
        raise Stage2RecoveryRejected(f"checkpoint SHA-256 mismatch for {task.task_id}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2RecoveryRejected(f"checkpoint cannot be loaded for {task.task_id}") from error
    if not isinstance(payload, dict) or any(
        (
            payload.get("schema_version") != "1.0.0",
            payload.get("status") != "COMPLETE",
            payload.get("seed") != task.seed,
            payload.get("task_sha256") != task.checkpoint_task_sha256,
            payload.get("epoch") != task.checkpoint_epoch,
        )
    ):
        raise Stage2RecoveryRejected(f"checkpoint identity mismatch for {task.task_id}")


def _existing_receipt(
    receipt_path: Path,
    database: Path,
    spec: Stage2RecoverySpec,
) -> dict[str, object] | None:
    if not receipt_path.is_file():
        return None
    try:
        receipt = Stage2RecoveryAuthorizationReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise Stage2RecoveryRejected(f"existing recovery receipt is invalid: {error}") from error
    if receipt.recovery_id != spec.recovery_id or receipt.run_id != spec.run_id:
        raise Stage2RecoveryRejected("existing recovery receipt identity drifted")
    with sqlite3.connect(database) as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM recovery_events WHERE recovery_id = ?",
            (spec.recovery_id,),
        ).fetchone()
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE attempt_id IN ({}) AND state = 'READY'".format(
                ",".join("?" for _ in receipt.new_attempt_ids)
            ),
            receipt.new_attempt_ids,
        ).fetchone()
    if event_count != (6,) or attempt_count != (6,):
        raise Stage2RecoveryRejected("existing recovery receipt does not match the ledger")
    return cast(dict[str, object], receipt.model_dump(mode="json"))


def authorize_stage2_device_mismatch_recovery(
    repository_root: Path,
    artifact_root: Path,
    *,
    spec_path: Path,
    recovery_input_receipt_path: Path,
) -> dict[str, object]:
    root = repository_root.resolve()
    artifacts = artifact_root.resolve()
    database = artifacts / "runtime/execution.sqlite3"
    receipt_path = artifacts / "runtime/device_mismatch_recovery_receipt.json"
    spec = load_stage2_recovery_spec(spec_path)
    spec_sha256 = canonical_json_hash(spec.model_dump(mode="json"))
    existing = _existing_receipt(receipt_path, database, spec)
    if existing is not None:
        return existing
    input_receipt = _load_input_receipt(recovery_input_receipt_path, spec)
    if not database.is_file() or sha256_file(database) != spec.source_database_sha256:
        raise Stage2RecoveryRejected("source execution database SHA-256 does not match")
    repository_config = root / "configs/stage2/stage2_v1.yaml"
    if repository_config.is_file():
        config = load_stage2_config(repository_config)
        if config.scientific_hash() != spec.scientific_config_sha256:
            raise Stage2RecoveryRejected("Stage 2 scientific configuration drifted")
    for task in spec.tasks:
        _validated_checkpoint(root, task)

    created_at = datetime.now(UTC).isoformat(timespec="microseconds")
    source_attempt_ids = tuple(task.source_attempt_id for task in spec.tasks)
    new_attempt_ids = tuple(f"{task.task_id}-attempt-2" for task in spec.tasks)
    receipt_payload: dict[str, object] = {
        "schema_version": "tarca-stage2-recovery-authorization-v1",
        "status": "AUTHORIZED",
        "recovery_id": spec.recovery_id,
        "reason": spec.reason,
        "run_id": spec.run_id,
        "graph_id": spec.graph_id,
        "spec_sha256": spec_sha256,
        "source_database_sha256": spec.source_database_sha256,
        "server_bundle_sha256": input_receipt.server_bundle_sha256,
        "source_attempt_ids": source_attempt_ids,
        "new_attempt_ids": new_attempt_ids,
        "checkpoint_sha256": tuple(task.checkpoint_sha256 for task in spec.tasks),
        "created_at_utc": created_at,
    }
    sealed = {**receipt_payload, "receipt_sha256": canonical_json_hash(receipt_payload)}
    receipt = Stage2RecoveryAuthorizationReceipt.model_validate(sealed)

    connection = sqlite3.connect(database, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recovery_events (
                recovery_id TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                source_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                new_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                reason TEXT NOT NULL,
                spec_sha256 TEXT NOT NULL,
                checkpoint_sha256 TEXT NOT NULL,
                code_bundle_sha256 TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                PRIMARY KEY (recovery_id, source_attempt_id),
                UNIQUE (new_attempt_id)
            )
            """
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        run = connection.execute(
            "SELECT graph_id, status FROM runs WHERE run_id = ?", (spec.run_id,)
        ).fetchone()
        if run is None or run["graph_id"] != spec.graph_id or run["status"] != "ACTIVE":
            raise Stage2RecoveryRejected("source run identity or state does not match")
        planned = connection.execute(
            "SELECT COUNT(*) FROM run_plan_nodes WHERE run_id = ?", (spec.run_id,)
        ).fetchone()
        if planned is None or int(planned[0]) != spec.planned_task_count:
            raise Stage2RecoveryRejected("source run plan count does not match")
        counts = {
            str(row["state"]): int(row["count"])
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM attempts GROUP BY state"
            ).fetchall()
        }
        if (
            counts.get("COMPLETED", 0) != spec.completed_attempt_count
            or counts.get("FAILED", 0) != spec.failed_attempt_count
            or set(counts) - {"COMPLETED", "FAILED"}
        ):
            raise Stage2RecoveryRejected("source attempt counts do not match")
        rows = connection.execute(
            """
            SELECT a.attempt_id, a.task_id, a.attempt_number, a.state,
                   a.error_category, a.packing_level, j.run_id, j.phase,
                   j.scientific_identity_json
            FROM attempts a JOIN job_nodes j USING(task_id)
            WHERE a.attempt_id IN ({})
            ORDER BY a.attempt_id
            """.format(",".join("?" for _ in source_attempt_ids)),
            source_attempt_ids,
        ).fetchall()
        by_attempt = {str(row["attempt_id"]): row for row in rows}
        if set(by_attempt) != set(source_attempt_ids):
            raise Stage2RecoveryRejected("one or more source attempts are missing")
        for task in spec.tasks:
            row = by_attempt[task.source_attempt_id]
            identity = json.loads(str(row["scientific_identity_json"]))
            if any(
                (
                    row["task_id"] != task.task_id,
                    row["attempt_number"] != 1,
                    row["state"] != "FAILED",
                    row["error_category"] != "WORKER_ERROR",
                    row["run_id"] != spec.run_id,
                    row["phase"] != "NEURAL_TRAIN",
                    identity.get("model_id") != task.model_id,
                    identity.get("seed") != task.seed,
                )
            ):
                raise Stage2RecoveryRejected(f"source attempt identity mismatch: {task.task_id}")
        for task, new_attempt_id in zip(spec.tasks, new_attempt_ids, strict=True):
            packing_level = int(by_attempt[task.source_attempt_id]["packing_level"])
            connection.execute(
                """
                INSERT INTO attempts(
                    attempt_id, task_id, attempt_number, state, packing_level,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, 2, 'READY', ?, ?, ?)
                """,
                (new_attempt_id, task.task_id, packing_level, created_at, created_at),
            )
            connection.execute(
                """
                INSERT INTO recovery_events(
                    recovery_id, run_id, source_attempt_id, new_attempt_id,
                    reason, spec_sha256, checkpoint_sha256,
                    code_bundle_sha256, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.recovery_id,
                    spec.run_id,
                    task.source_attempt_id,
                    new_attempt_id,
                    spec.reason,
                    spec_sha256,
                    task.checkpoint_sha256,
                    input_receipt.server_bundle_sha256,
                    created_at,
                ),
            )
        connection.execute(
            """
            INSERT INTO alerts(run_id, attempt_id, created_at_utc, category, message)
            VALUES (?, NULL, ?, ?, ?)
            """,
            (
                spec.run_id,
                created_at,
                "CONTROLLED_RECOVERY_AUTHORIZED",
                "Six COMPLETE neural checkpoints were authorized for zero-training-step recovery.",
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    payload = cast(dict[str, object], receipt.model_dump(mode="json"))
    _atomic_json(receipt_path, payload)
    return payload
