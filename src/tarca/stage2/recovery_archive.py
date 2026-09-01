from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import IO, cast

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.stage2.recovery import (
    Stage2RecoveryRejected,
    Stage2RecoverySpec,
    load_stage2_recovery_spec,
)


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\x00" in name:
        raise Stage2RecoveryRejected("recovery archive contains an unsafe path")
    return path


def _atomic_stream(destination: Path, source: IO[bytes]) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".stage2-restore-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _atomic_json(destination: Path, payload: dict[str, object]) -> None:
    data = canonical_json_bytes(payload) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".stage2-restore-receipt-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _archive_manifest(archive: tarfile.TarFile) -> tuple[dict[str, object], str]:
    candidates = tuple(
        member
        for member in archive.getmembers()
        if member.isfile()
        and PurePosixPath(member.name).parent == PurePosixPath("transfer")
        and PurePosixPath(member.name).name.startswith("stage2-recovery-manifest-")
        and PurePosixPath(member.name).suffix == ".json"
    )
    if len(candidates) != 1:
        raise Stage2RecoveryRejected("recovery archive must contain one server manifest")
    handle = archive.extractfile(candidates[0])
    if handle is None:
        raise Stage2RecoveryRejected("recovery archive manifest cannot be read")
    try:
        value = json.load(handle)
    except Exception as error:
        raise Stage2RecoveryRejected("recovery archive manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise Stage2RecoveryRejected("recovery archive manifest must be an object")
    expected = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if expected != canonical_json_hash(unsigned):
        raise Stage2RecoveryRejected("recovery archive manifest SHA-256 does not match")
    return cast(dict[str, object], value), candidates[0].name


def _validate_manifest(manifest: dict[str, object], spec: Stage2RecoverySpec) -> str:
    run = manifest.get("run")
    counts = manifest.get("attempt_counts")
    if not isinstance(run, dict) or not isinstance(counts, dict):
        raise Stage2RecoveryRejected("recovery archive run evidence is missing")
    if any(
        (
            manifest.get("schema_version")
            != "tarca-stage2-server-recovery-archive-v1",
            manifest.get("manifest_sha256") != spec.source_manifest_sha256,
            run.get("run_id") != spec.run_id,
            run.get("graph_id") != spec.graph_id,
            run.get("status") != "ACTIVE",
            manifest.get("planned_tasks") != spec.planned_task_count,
            manifest.get("failed_neural_tasks") != 6,
            counts.get("COMPLETED") != spec.completed_attempt_count,
            counts.get("FAILED") != 6,
            manifest.get("database_snapshot_sha256") != spec.source_database_sha256,
        )
    ):
        raise Stage2RecoveryRejected("recovery archive manifest does not match the spec")
    checkpoint_rows = manifest.get("checkpoints")
    if not isinstance(checkpoint_rows, list):
        raise Stage2RecoveryRejected("recovery archive checkpoint evidence is missing")
    observed = {
        (
            row.get("relative_path"),
            row.get("sha256"),
            row.get("status"),
            row.get("epoch"),
            row.get("seed"),
        )
        for row in checkpoint_rows
        if isinstance(row, dict)
    }
    expected = {
        (
            task.checkpoint_relative_path,
            task.checkpoint_sha256,
            "COMPLETE",
            task.checkpoint_epoch,
            task.seed,
        )
        for task in spec.tasks
    }
    if observed != expected:
        raise Stage2RecoveryRejected("recovery archive checkpoint set does not match")
    snapshot = manifest.get("database_snapshot_relative_path")
    if not isinstance(snapshot, str):
        raise Stage2RecoveryRejected("recovery archive database snapshot path is missing")
    _safe_member_name(snapshot)
    return snapshot


def _is_restorable_artifact(path: PurePosixPath) -> bool:
    prefixes = (
        ("artifacts", "stage2"),
        ("artifacts", "stage1b", "frozen", "v2"),
        ("artifacts", "e01", "frozen", "v2"),
    )
    if not any(path.parts[: len(prefix)] == prefix for prefix in prefixes):
        return False
    return path.as_posix() != "artifacts/stage2/runtime/execution.sqlite3"


def _destination(root: Path, member: PurePosixPath) -> Path:
    destination = (root / Path(*member.parts)).resolve()
    if destination == root or root not in destination.parents:
        raise Stage2RecoveryRejected("recovery destination escapes the repository")
    return destination


def restore_stage2_recovery_archive(
    repository_root: Path,
    *,
    recovery_archive: Path,
    server_bundle: Path,
    spec_path: Path,
) -> dict[str, object]:
    root = repository_root.resolve()
    archive_path = recovery_archive.resolve()
    bundle_path = server_bundle.resolve()
    spec = load_stage2_recovery_spec(spec_path)
    if archive_path.name != spec.source_archive_filename:
        raise Stage2RecoveryRejected("recovery archive filename does not match the spec")
    if not archive_path.is_file() or sha256_file(archive_path) != spec.source_archive_sha256:
        raise Stage2RecoveryRejected("recovery archive SHA-256 does not match the spec")
    if not bundle_path.is_file():
        raise Stage2RecoveryRejected("server bundle is missing")
    authorization = root / "artifacts/stage2/runtime/device_mismatch_recovery_receipt.json"
    if authorization.exists():
        raise Stage2RecoveryRejected("recovery input cannot overwrite an authorized ledger")

    restored_files = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _safe_member_name(member.name)
            if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise Stage2RecoveryRejected("recovery archive contains a non-regular member")
        manifest, _ = _archive_manifest(archive)
        snapshot_name = _validate_manifest(manifest, spec)
        by_name = {member.name: member for member in members}
        for member in members:
            member_path = _safe_member_name(member.name)
            if not member.isfile() or not _is_restorable_artifact(member_path):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise Stage2RecoveryRejected(f"recovery artifact cannot be read: {member.name}")
            _atomic_stream(_destination(root, member_path), handle)
            restored_files += 1
        snapshot_member = by_name.get(snapshot_name)
        if snapshot_member is None or not snapshot_member.isfile():
            raise Stage2RecoveryRejected("recovery database snapshot is missing")
        snapshot_handle = archive.extractfile(snapshot_member)
        if snapshot_handle is None:
            raise Stage2RecoveryRejected("recovery database snapshot cannot be read")
        database = root / "artifacts/stage2/runtime/execution.sqlite3"
        database_sha256 = _atomic_stream(database, snapshot_handle)
        if database_sha256 != spec.source_database_sha256:
            raise Stage2RecoveryRejected("restored execution database SHA-256 does not match")

    for task in spec.tasks:
        checkpoint = root / task.checkpoint_relative_path
        if not checkpoint.is_file() or sha256_file(checkpoint) != task.checkpoint_sha256:
            raise Stage2RecoveryRejected("restored checkpoint SHA-256 does not match")
    frozen_inputs = manifest.get("frozen_input_sha256")
    if not isinstance(frozen_inputs, dict):
        raise Stage2RecoveryRejected("recovery frozen-input evidence is missing")
    for relative, expected_sha256 in frozen_inputs.items():
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise Stage2RecoveryRejected("recovery frozen-input evidence is invalid")
        path = _destination(root, _safe_member_name(relative))
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise Stage2RecoveryRejected(f"restored frozen input SHA-256 mismatch: {relative}")

    receipt_payload: dict[str, object] = {
        "schema_version": "tarca-stage2-recovery-input-v1",
        "status": "RESTORED",
        "source_archive_sha256": spec.source_archive_sha256,
        "source_manifest_sha256": spec.source_manifest_sha256,
        "source_database_sha256": spec.source_database_sha256,
        "server_bundle_sha256": sha256_file(bundle_path),
        "restored_file_count": restored_files + 1,
    }
    receipt = {**receipt_payload, "receipt_sha256": canonical_json_hash(receipt_payload)}
    _atomic_json(root / "artifacts/stage2/runtime/recovery_input_receipt.json", receipt)
    return receipt
