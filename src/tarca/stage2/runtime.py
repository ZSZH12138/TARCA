from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.stage2.config import load_stage2_config

STAGE2_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN"


class Stage2RuntimeAuthorizationError(RuntimeError):
    """Raised before any Stage 2 state mutation when authorization is incomplete."""


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return payload


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _unsigned(value)
    return {**payload, "receipt_sha256": canonical_json_hash(payload)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".stage2-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_receipt(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2RuntimeAuthorizationError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("receipt_sha256") != canonical_json_hash(
        _unsigned(value)
    ):
        raise Stage2RuntimeAuthorizationError(f"{label} is invalid or tampered")
    return value


def prepare_stage2(repository_root: Path, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    del repository_root
    config = load_stage2_config(config_path.resolve())
    receipt = _seal(
        {
            "schema_version": "tarca-stage2-prepared-v1",
            "status": "PREPARED",
            "config_file_sha256": sha256_file(config_path.resolve()),
            "scientific_config_sha256": config.scientific_hash(),
            "runtime_profile_sha256": config.runtime_hash(),
            "expected_gpu_training_tasks": 6,
            "completed_task_policy": "NEVER_RERUN",
            "formal_tasks_executed": 0,
            "scientific_results_visible": False,
        }
    )
    path = artifact_root.resolve() / "runtime/prepared_receipt.json"
    if path.is_file():
        existing = _read_receipt(path, "prepared receipt")
        if existing != receipt:
            raise Stage2RuntimeAuthorizationError("prepared receipt identity drifted")
        return existing
    _atomic_json(path, receipt)
    return receipt


def dry_run_stage2(repository_root: Path, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    del repository_root, config_path
    prepared = _read_receipt(
        artifact_root.resolve() / "runtime/prepared_receipt.json", "prepared receipt"
    )
    return {
        "status": "DRY_RUN_OK",
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "gpu_training_tasks": 6,
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }


def record_stage2_preflight(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    evidence_path: Path,
) -> dict[str, Any]:
    del repository_root, config_path
    prepared = _read_receipt(
        artifact_root.resolve() / "runtime/prepared_receipt.json", "prepared receipt"
    )
    evidence = json.loads(evidence_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("status") != "PREFLIGHT_PASS":
        raise Stage2RuntimeAuthorizationError("preflight evidence did not pass")
    receipt = _seal(
        {
            "schema_version": "tarca-stage2-preflight-v1",
            "status": "PREFLIGHT_PASS",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "evidence_sha256": sha256_file(evidence_path.resolve()),
            "formal_tasks_executed": 0,
            "scientific_results_visible": False,
        }
    )
    _atomic_json(artifact_root.resolve() / "runtime/preflight_receipt.json", receipt)
    return receipt


def _authorize(value: str) -> None:
    if value != STAGE2_ACKNOWLEDGEMENT:
        raise Stage2RuntimeAuthorizationError("exact Stage 2 training acknowledgement required")


def launch_stage2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    del repository_root, config_path
    _authorize(acknowledgement)
    runtime = artifact_root.resolve() / "runtime"
    prepared = _read_receipt(runtime / "prepared_receipt.json", "prepared receipt")
    preflight = _read_receipt(runtime / "preflight_receipt.json", "preflight receipt")
    if preflight.get("prepared_receipt_sha256") != prepared.get("receipt_sha256"):
        raise Stage2RuntimeAuthorizationError("preflight does not bind the prepared receipt")
    database = runtime / "execution.sqlite3"
    if database.exists():
        raise Stage2RuntimeAuthorizationError("execution database exists; use resume")
    receipt = _seal(
        {
            "schema_version": "tarca-stage2-launch-v1",
            "status": "AUTHORIZED",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "preflight_receipt_sha256": preflight["receipt_sha256"],
            "run_id": f"stage2-run-{prepared['scientific_config_sha256']}",
        }
    )
    _atomic_json(runtime / "launch_authorization_receipt.json", receipt)
    return receipt


def resume_stage2(
    repository_root: Path, config_path: Path, artifact_root: Path, *, acknowledgement: str
) -> dict[str, Any]:
    del repository_root, config_path
    _authorize(acknowledgement)
    runtime = artifact_root.resolve() / "runtime"
    if not (runtime / "execution.sqlite3").is_file():
        raise Stage2RuntimeAuthorizationError("execution database is required before resume")
    launch = _read_receipt(runtime / "launch_authorization_receipt.json", "launch receipt")
    return {"status": "RESUME_AUTHORIZED", "run_id": launch["run_id"]}


def status_stage2(artifact_root: Path) -> dict[str, Any]:
    database = artifact_root.resolve() / "runtime/execution.sqlite3"
    if not database.is_file():
        return {"status": "NOT_STARTED", "scientific_results_visible": False}
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT state, COUNT(*) FROM attempts GROUP BY state").fetchall()
    return {
        "status": "RUNNING_OR_WAITING",
        "attempt_counts": {str(state): int(count) for state, count in rows},
        "scientific_results_visible": False,
    }


def recover_stage2(artifact_root: Path) -> dict[str, Any]:
    runtime = artifact_root.resolve() / "runtime"
    candidates = tuple(
        path
        for path in sorted(runtime.glob("*"))
        if path.is_file() and path.suffix in {".json", ".sqlite3"}
    )
    if not candidates:
        raise Stage2RuntimeAuthorizationError(
            "no consistent runtime files are available to recover"
        )
    digest = canonical_json_hash([(path.name, sha256_file(path)) for path in candidates])
    capsule = runtime / "recovery" / digest
    capsule.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.copy2(path, capsule / path.name)
    receipt = _seal(
        {
            "schema_version": "tarca-stage2-recovery-v1",
            "status": "RECOVERED",
            "capsule_sha256": digest,
            "files": [path.name for path in candidates],
        }
    )
    _atomic_json(capsule / "recovery_receipt.json", receipt)
    return receipt


def dispatch_stage2_runtime_command(
    command: str, repository_root: Path, config_path: Path, artifact_root: Path, **arguments: Any
) -> dict[str, Any]:
    commands: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": prepare_stage2,
        "dry-run": dry_run_stage2,
        "preflight": record_stage2_preflight,
        "launch": launch_stage2,
        "resume": resume_stage2,
    }
    if command == "status":
        return status_stage2(artifact_root)
    if command == "recover":
        return recover_stage2(artifact_root)
    if command == "freeze":
        return {"status": "FREEZE_REQUIRES_COMPLETED_MANIFEST"}
    if command not in commands:
        raise ValueError("Stage 2 runtime command is not allowlisted")
    return commands[command](repository_root, config_path, artifact_root, **arguments)
