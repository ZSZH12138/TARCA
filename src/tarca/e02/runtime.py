from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e02.config import load_e02_config
from tarca.e02.grant import create_e02_grant

E02_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN"


class E02RuntimeAuthorizationError(RuntimeError):
    """Raised before sealed formal access when authorization is incomplete."""


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("receipt_sha256", None)
    return result


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _unsigned(value)
    return {**payload, "receipt_sha256": canonical_json_hash(payload)}


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".e02-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise E02RuntimeAuthorizationError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("receipt_sha256") != canonical_json_hash(
        _unsigned(value)
    ):
        raise E02RuntimeAuthorizationError(f"{label} is invalid or tampered")
    return value


def prepare_e02(repository_root: Path, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    del repository_root
    config = load_e02_config(config_path.resolve())
    receipt = _seal(
        {
            "schema_version": "tarca-e02-prepared-v1",
            "status": "PREPARED",
            "config_file_sha256": sha256_file(config_path.resolve()),
            "scientific_config_sha256": config.scientific_hash(),
            "expected_formal_trajectories": config.gate.required_completed_trajectories,
            "formal_tasks_executed": 0,
            "scientific_results_visible": False,
        }
    )
    path = artifact_root.resolve() / "runtime/prepared_receipt.json"
    if path.is_file():
        existing = _read(path, "prepared receipt")
        if existing != receipt:
            raise E02RuntimeAuthorizationError("prepared receipt identity drifted")
        return existing
    _atomic(path, receipt)
    return receipt


def dry_run_e02(repository_root: Path, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    del repository_root, config_path
    prepared = _read(artifact_root.resolve() / "runtime/prepared_receipt.json", "prepared receipt")
    return {
        "status": "DRY_RUN_OK",
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }


def preflight_e02(repository_root: Path, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    del repository_root, config_path
    prepared = _read(artifact_root.resolve() / "runtime/prepared_receipt.json", "prepared receipt")
    receipt = _seal(
        {
            "schema_version": "tarca-e02-preflight-v1",
            "status": "PREFLIGHT_PASS",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "formal_tasks_executed": 0,
        }
    )
    _atomic(artifact_root.resolve() / "runtime/preflight_receipt.json", receipt)
    return receipt


def launch_e02(
    repository_root: Path, config_path: Path, artifact_root: Path, *, acknowledgement: str
) -> dict[str, Any]:
    del repository_root, config_path
    if acknowledgement != E02_ACKNOWLEDGEMENT:
        raise E02RuntimeAuthorizationError("exact E02 formal-run acknowledgement required")
    runtime = artifact_root.resolve() / "runtime"
    prepared = _read(runtime / "prepared_receipt.json", "prepared receipt")
    preflight = _read(runtime / "preflight_receipt.json", "preflight receipt")
    if preflight.get("prepared_receipt_sha256") != prepared.get("receipt_sha256"):
        raise E02RuntimeAuthorizationError("preflight does not bind the prepared receipt")
    authorization = ArtifactRef(
        artifact_id=f"e02-authorization-{prepared['scientific_config_sha256']}",
        artifact_type="SEALED_ACCESS_AUTHORIZATION",
        content_hash=canonical_json_hash(
            {"prepared": prepared["receipt_sha256"], "preflight": preflight["receipt_sha256"]}
        ),
        schema_version="1.0.0",
        relative_path="artifacts/e02/runtime/launch_authorization.json",
    )
    grant = create_e02_grant(authorization, issued_at=datetime.now(UTC))
    _atomic(runtime / "sealed_access_grant.json", grant.model_dump(mode="json"))
    return _seal(
        {
            "schema_version": "tarca-e02-launch-v1",
            "status": "AUTHORIZED",
            "grant_id": grant.grant_id,
            "run_id": f"e02-run-{prepared['scientific_config_sha256']}",
        }
    )


def dispatch_e02_runtime_command(
    command: str, repository_root: Path, config_path: Path, artifact_root: Path, **arguments: Any
) -> dict[str, Any]:
    commands: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": prepare_e02,
        "dry-run": dry_run_e02,
        "preflight": preflight_e02,
        "launch": launch_e02,
    }
    if command == "status":
        return {"status": "NOT_STARTED", "scientific_results_visible": False}
    if command == "resume":
        if arguments.get("acknowledgement") != E02_ACKNOWLEDGEMENT:
            raise E02RuntimeAuthorizationError("exact E02 formal-run acknowledgement required")
        return {"status": "RESUME_AUTHORIZED"}
    if command == "finalize":
        return {"status": "FINALIZE_REQUIRES_COMPLETE_EVIDENCE"}
    if command == "recover":
        return {"status": "RECOVERY_REQUIRES_RUNTIME_STATE"}
    if command not in commands:
        raise ValueError("E02 runtime command is not allowlisted")
    return commands[command](repository_root, config_path, artifact_root, **arguments)
