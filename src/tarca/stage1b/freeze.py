from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tarca.contracts import canonical_json_bytes


class FreezeRejected(RuntimeError):
    """Raised when qualification evidence cannot be promoted to a frozen suite."""


@dataclass(frozen=True, slots=True)
class OverrideAuthorization:
    authorized_by: str
    reason: str
    prior_version: str

    def __post_init__(self) -> None:
        if not self.authorized_by.strip() or not self.reason.strip():
            raise ValueError("override authorization identity and reason must not be blank")
        if re.fullmatch(r"v[1-9][0-9]*", self.prior_version) is None:
            raise ValueError("override prior version must use vN format")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any], *, replace: bool) -> bytes:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FreezeRejected(f"frozen path already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreezeRejected(f"{label} must be a JSON object")
    return value


def _validate_receipt(receipt: dict[str, Any]) -> None:
    suite_decision = _mapping(receipt.get("suite_decision"), "suite decision")
    if suite_decision.get("status") != "PASS":
        raise FreezeRejected("suite gate did not pass")
    if receipt.get("source_evidence_verified") is not True:
        raise FreezeRejected("source evidence was not verified")
    required_hashes = (
        "source_manifest_sha256",
        "world_config_sha256",
        "qualification_config_sha256",
        "hardware_receipt_sha256",
    )
    for field in required_hashes:
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))) is None:
            raise FreezeRejected(f"receipt {field} is missing or invalid")
    source_commits = receipt.get("source_commits")
    if not isinstance(source_commits, dict) or not source_commits:
        raise FreezeRejected("receipt source commits are missing")
    if any(
        not isinstance(source_id, str)
        or not source_id.strip()
        or re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None
        for source_id, commit in source_commits.items()
    ):
        raise FreezeRejected("receipt source commits are invalid")
    serialized = canonical_json_bytes(receipt).decode("utf-8")
    if any(identifier in serialized for identifier in ('"E01"', '"E02"', '"TEST"')):
        raise FreezeRejected("receipt contains a formal experiment identifier")


def load_active_pointer(artifact_root: Path) -> dict[str, Any]:
    active_path = artifact_root.resolve() / "active.json"
    if not active_path.is_file():
        raise FreezeRejected("Stage1B active pointer does not exist")
    return _mapping(json.loads(active_path.read_text(encoding="utf-8")), "active pointer")


def freeze_suite(
    receipt: dict[str, Any],
    artifact_root: Path,
    *,
    version: str,
    authorization: OverrideAuthorization | None = None,
) -> dict[str, Any]:
    if re.fullmatch(r"v[1-9][0-9]*", version) is None:
        raise FreezeRejected("freeze version must use vN format")
    _validate_receipt(receipt)
    root = artifact_root.resolve()
    version_root = root / "versions" / version
    if version_root.exists():
        raise FreezeRejected("frozen versions are immutable and cannot be overwritten")
    active_path = root / "active.json"
    prior_pointer = load_active_pointer(root) if active_path.is_file() else None
    if prior_pointer is not None:
        if authorization is None:
            raise FreezeRejected("moving the active pointer requires user authorization")
        if authorization.prior_version != prior_pointer.get("version"):
            raise FreezeRejected("authorization prior version does not match the active pointer")
    elif authorization is not None:
        raise FreezeRejected("initial freeze must not claim an override authorization")

    receipt_payload = canonical_json_bytes(receipt)
    world_decisions = receipt.get("world_decisions")
    if not isinstance(world_decisions, list):
        raise FreezeRejected("receipt world decisions are missing")
    selected_models = {
        str(item["world_id"]): str(item["selected_neural_adapter"])
        for raw_item in world_decisions
        for item in [_mapping(raw_item, "world decision")]
        if item.get("role") == "PRIMARY_MECHANISTIC" and item.get("status") == "PASS"
    }
    manifest: dict[str, Any] = {
        "schema_version": "2.0.0",
        "version": version,
        "qualification_id": receipt.get("qualification_id"),
        "suite_id": receipt.get("suite_id"),
        "suite_gate_status": "PASS",
        "receipt_sha256": _sha256(receipt_payload),
        "source_manifest_sha256": receipt.get("source_manifest_sha256"),
        "source_commits": receipt.get("source_commits"),
        "world_config_sha256": receipt.get("world_config_sha256"),
        "qualification_config_sha256": receipt.get("qualification_config_sha256"),
        "hardware_receipt_sha256": receipt.get("hardware_receipt_sha256"),
        "selected_models": selected_models,
        "override_authorization": asdict(authorization) if authorization is not None else None,
    }
    manifest_path = version_root / "manifest.json"
    manifest_payload = _write_json(manifest_path, manifest, replace=False)
    manifest_hash = _sha256(manifest_payload)
    hash_path = version_root / "manifest.sha256"
    hash_path.write_text(f"{manifest_hash}\n", encoding="ascii")
    manifest_path.chmod(0o444)
    hash_path.chmod(0o444)
    active = {
        "schema_version": "2.0.0",
        "version": version,
        "manifest_sha256": manifest_hash,
    }
    _write_json(active_path, active, replace=True)
    return manifest


def verify_frozen_suite(
    artifact_root: Path,
    *,
    version: str | None = None,
    allow_unfrozen: bool = False,
) -> dict[str, Any]:
    root = artifact_root.resolve()
    active_path = root / "active.json"
    if not active_path.is_file():
        if allow_unfrozen:
            return {"status": "UNFROZEN", "active_version": None}
        raise FreezeRejected("Stage1B is not frozen")
    active = load_active_pointer(root)
    selected_version = version or str(active.get("version", ""))
    if re.fullmatch(r"v[1-9][0-9]*", selected_version) is None:
        raise FreezeRejected("active pointer version is invalid")
    version_root = root / "versions" / selected_version
    manifest_path = version_root / "manifest.json"
    hash_path = version_root / "manifest.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise FreezeRejected("frozen manifest files are missing")
    manifest_payload = manifest_path.read_bytes()
    expected_hash = hash_path.read_text(encoding="ascii").strip()
    actual_hash = _sha256(manifest_payload)
    if actual_hash != expected_hash:
        raise FreezeRejected("frozen manifest hash does not match")
    if selected_version == active.get("version") and active.get("manifest_sha256") != actual_hash:
        raise FreezeRejected("active pointer manifest hash does not match")
    manifest = _mapping(json.loads(manifest_payload), "frozen manifest")
    if manifest.get("suite_gate_status") != "PASS" or manifest.get("version") != selected_version:
        raise FreezeRejected("frozen manifest gate or version is invalid")
    return {
        "status": "PASS",
        "active_version": active.get("version"),
        "verified_version": selected_version,
        "manifest_sha256": actual_hash,
    }
