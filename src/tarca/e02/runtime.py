from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e02.config import load_e02_config
from tarca.e02.grant import create_e02_grant
from tarca.e02.runner import run_e02_formal
from tarca.e02.tasks import E02Graph, FrozenStage2Input, compile_e02_graph
from tarca.execution import LocalMultiProcessBackend, ResourceCapacity, RunTerminalStatus
from tarca.stage2.freeze import verify_frozen_stage2_suite

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
    del config_path
    prepared = _read(artifact_root.resolve() / "runtime/prepared_receipt.json", "prepared receipt")
    freeze = verify_frozen_stage2_suite(repository_root.resolve() / "artifacts/stage2")
    receipt = _seal(
        {
            "schema_version": "tarca-e02-preflight-v1",
            "status": "PREFLIGHT_PASS",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "stage2_freeze_receipt_sha256": freeze.receipt_sha256,
            "formal_tasks_executed": 0,
        }
    )
    _atomic(artifact_root.resolve() / "runtime/preflight_receipt.json", receipt)
    return receipt


def _file_ref(
    root: Path,
    relative: str,
    artifact_id: str,
    artifact_type: str,
) -> ArtifactRef:
    path = root / relative
    if not path.is_file():
        raise E02RuntimeAuthorizationError(f"required E02 input is missing: {relative}")
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        content_hash=sha256_file(path),
        schema_version="1.0.0",
        relative_path=relative,
    )


def _compiled_graph(root: Path, config_path: Path) -> E02Graph:
    verify_frozen_stage2_suite(root / "artifacts/stage2")
    return compile_e02_graph(
        load_e02_config(config_path.resolve()),
        FrozenStage2Input(
            freeze_receipt=_file_ref(
                root,
                "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json",
                "stage2-v1-freeze-receipt",
                "STAGE2_FREEZE_RECEIPT",
            ),
            sealed_access_grant=_file_ref(
                root,
                "artifacts/e02/runtime/sealed_access_grant.json",
                "e02-v1-sealed-access-grant",
                "SEALED_ACCESS_GRANT",
            ),
            frozen=True,
        ),
    )


def _runtime_capacity(artifact_root: Path) -> ResourceCapacity:
    import psutil
    import torch

    return ResourceCapacity(
        logical_cpu_count=psutil.cpu_count(logical=True) or 0,
        physical_cpu_count=psutil.cpu_count(logical=False) or 0,
        available_memory_bytes=psutil.virtual_memory().available,
        gpu_memory_bytes=tuple(
            int(torch.cuda.get_device_properties(index).total_memory)
            for index in range(torch.cuda.device_count())
        ),
        local_storage_available=True,
        local_storage_free_bytes=shutil.disk_usage(artifact_root.resolve()).free,
    )


def launch_e02(
    repository_root: Path, config_path: Path, artifact_root: Path, *, acknowledgement: str
) -> dict[str, Any]:
    if acknowledgement != E02_ACKNOWLEDGEMENT:
        raise E02RuntimeAuthorizationError("exact E02 formal-run acknowledgement required")
    root = repository_root.resolve()
    runtime = artifact_root.resolve() / "runtime"
    prepared = _read(runtime / "prepared_receipt.json", "prepared receipt")
    preflight = _read(runtime / "preflight_receipt.json", "preflight receipt")
    if preflight.get("prepared_receipt_sha256") != prepared.get("receipt_sha256"):
        raise E02RuntimeAuthorizationError("preflight does not bind the prepared receipt")
    freeze = verify_frozen_stage2_suite(root / "artifacts/stage2")
    if preflight.get("stage2_freeze_receipt_sha256") != freeze.receipt_sha256:
        raise E02RuntimeAuthorizationError("preflight does not bind the frozen Stage 2 suite")
    database = runtime / "execution.sqlite3"
    if database.exists():
        raise E02RuntimeAuthorizationError("E02 execution database exists; use resume")
    authorization_payload = _seal(
        {
            "schema_version": "tarca-e02-authorization-v1",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "preflight_receipt_sha256": preflight["receipt_sha256"],
            "stage2_freeze_receipt_sha256": freeze.receipt_sha256,
        }
    )
    _atomic(runtime / "launch_authorization.json", authorization_payload)
    authorization = ArtifactRef(
        artifact_id=f"e02-authorization-{prepared['scientific_config_sha256']}",
        artifact_type="SEALED_ACCESS_AUTHORIZATION",
        content_hash=sha256_file(runtime / "launch_authorization.json"),
        schema_version="1.0.0",
        relative_path="artifacts/e02/runtime/launch_authorization.json",
    )
    grant = create_e02_grant(authorization, issued_at=datetime.now(UTC))
    _atomic(runtime / "sealed_access_grant.json", grant.model_dump(mode="json"))
    graph = _compiled_graph(root, config_path)
    graph_run_id = f"run-{graph.graph_id.removeprefix('e02-graph-')}"
    launch = _seal(
        {
            "schema_version": "tarca-e02-launch-v1",
            "status": "AUTHORIZED",
            "grant_id": grant.grant_id,
            "run_id": graph_run_id,
            "graph_id": graph.graph_id,
        }
    )
    _atomic(runtime / "launch_receipt.json", launch)
    result = run_e02_formal(
        graph,
        _runtime_capacity(artifact_root),
        repository_root=root,
        database_path=database,
        backend=LocalMultiProcessBackend(
            root, environment_overrides={"TARCA_EXECUTION_KIND": "e02-v1"}
        ),
        maximum_wait_seconds=2.0,
    )
    if result.status is not RunTerminalStatus.COMPLETED:
        raise RuntimeError(f"E02 execution stopped with {result.status.value}")
    receipt_path = artifact_root.resolve() / "frozen/v1/e02_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        **launch,
        "status": "COMPLETED",
        "completed_tasks": len(result.completed),
        "outcome": receipt["outcome"],
        "e02_receipt_sha256": receipt["receipt_sha256"],
    }


def resume_e02(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    if acknowledgement != E02_ACKNOWLEDGEMENT:
        raise E02RuntimeAuthorizationError("exact E02 formal-run acknowledgement required")
    root = repository_root.resolve()
    runtime = artifact_root.resolve() / "runtime"
    launch = _read(runtime / "launch_receipt.json", "launch receipt")
    if not (runtime / "execution.sqlite3").is_file():
        raise E02RuntimeAuthorizationError("execution database is required before resume")
    graph = _compiled_graph(root, config_path)
    if graph.graph_id != launch.get("graph_id"):
        raise E02RuntimeAuthorizationError("E02 resume graph identity drifted")
    result = run_e02_formal(
        graph,
        _runtime_capacity(artifact_root),
        repository_root=root,
        database_path=runtime / "execution.sqlite3",
        backend=LocalMultiProcessBackend(
            root, environment_overrides={"TARCA_EXECUTION_KIND": "e02-v1"}
        ),
        maximum_wait_seconds=2.0,
    )
    return {
        "status": result.status.value,
        "run_id": launch["run_id"],
        "completed_tasks": len(result.completed),
    }


def status_e02(artifact_root: Path) -> dict[str, Any]:
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


def recover_e02(artifact_root: Path) -> dict[str, Any]:
    runtime = artifact_root.resolve() / "runtime"
    candidates = tuple(
        path
        for path in sorted(runtime.glob("*"))
        if path.is_file() and path.suffix in {".json", ".sqlite3"}
    )
    if not candidates:
        raise E02RuntimeAuthorizationError("no consistent E02 runtime files are available")
    digest = canonical_json_hash([(path.name, sha256_file(path)) for path in candidates])
    capsule = runtime / "recovery" / digest
    capsule.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.copy2(path, capsule / path.name)
    receipt = _seal(
        {
            "schema_version": "tarca-e02-recovery-v1",
            "status": "RECOVERED",
            "capsule_sha256": digest,
            "files": [path.name for path in candidates],
        }
    )
    _atomic(capsule / "recovery_receipt.json", receipt)
    return receipt


def dispatch_e02_runtime_command(
    command: str, repository_root: Path, config_path: Path, artifact_root: Path, **arguments: Any
) -> dict[str, Any]:
    commands: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": prepare_e02,
        "dry-run": dry_run_e02,
        "preflight": preflight_e02,
        "launch": launch_e02,
        "resume": resume_e02,
    }
    if command == "status":
        return status_e02(artifact_root)
    if command == "finalize":
        path = artifact_root.resolve() / "frozen/v1/e02_receipt.json"
        if not path.is_file():
            return {"status": "FINALIZE_REQUIRES_COMPLETE_EVIDENCE"}
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if command == "recover":
        return recover_e02(artifact_root)
    if command not in commands:
        raise ValueError("E02 runtime command is not allowlisted")
    return commands[command](repository_root, config_path, artifact_root, **arguments)
