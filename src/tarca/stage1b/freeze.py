from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import Field, ValidationError, model_validator

from tarca.contracts import Sha256Hash, canonical_json_bytes
from tarca.stage1b.config import FrozenModel
from tarca.stage1b.evidence_io import sha256_bytes, write_canonical_json


class FreezeRejected(RuntimeError):
    """Raised when qualification evidence cannot be promoted to a frozen suite."""


class QualificationEvidence(FrozenModel):
    official_source_receipt_sha256: Sha256Hash
    reproduction_receipt_sha256: Sha256Hash
    environment_receipt_sha256: Sha256Hash
    precision_receipt_sha256: Sha256Hash
    run_graph_sha256: Sha256Hash
    task_manifest_sha256: Sha256Hash
    execution_plan_sha256: Sha256Hash
    hardware_receipt_sha256: Sha256Hash
    completed_task_count: int = Field(gt=0)
    expected_task_count: int = Field(gt=0)
    source_drift_detected: bool = False
    identity_drift_detected: bool = False

    @model_validator(mode="after")
    def reject_incomplete_or_drifted_run(self) -> Self:
        if self.completed_task_count != self.expected_task_count:
            raise ValueError("partial qualification cannot be frozen")
        if self.source_drift_detected:
            raise ValueError("source drift prevents qualification freeze")
        if self.identity_drift_detected:
            raise ValueError("scientific identity drift prevents qualification freeze")
        return self


@dataclass(frozen=True, slots=True)
class OverrideAuthorization:
    authorized_by: str
    reason: str
    prior_manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.authorized_by.strip() or not self.reason.strip():
            raise ValueError("override authorization identity and reason must not be blank")
        if re.fullmatch(r"[0-9a-f]{64}", self.prior_manifest_sha256) is None:
            raise ValueError("prior manifest hash must be a lowercase SHA-256")


def _write_json(path: Path, value: dict[str, Any], *, replace: bool) -> bytes:
    try:
        return write_canonical_json(path, value, replace=replace)
    except FileExistsError as error:
        raise FreezeRejected(f"frozen path already exists: {path}") from error


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreezeRejected(f"{label} must be a JSON object")
    return value


def freeze_root(artifact_root: Path) -> Path:
    return artifact_root.resolve() / "frozen" / "v2"


def _integer_seed_set(value: object, label: str, *, allow_empty: bool) -> set[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise FreezeRejected(f"receipt {label} are missing")
    if any(type(seed) is not int or seed < 0 for seed in value):
        raise FreezeRejected(f"receipt {label} are invalid")
    seeds = set(value)
    if len(seeds) != len(value):
        raise FreezeRejected(f"receipt {label} contain duplicates")
    return seeds


def _validate_seed_boundary(receipt: dict[str, Any]) -> None:
    qualification_seeds = _integer_seed_set(
        receipt.get("qualification_seeds"), "qualification seeds", allow_empty=False
    )
    reserved_seeds = _integer_seed_set(
        receipt.get("reserved_formal_seeds"), "reserved formal seeds", allow_empty=True
    )
    if qualification_seeds & reserved_seeds:
        raise FreezeRejected("qualification uses a reserved formal seed")
    for collection_name in ("comparisons", "training_receipts"):
        collection = receipt.get(collection_name)
        if not isinstance(collection, list):
            raise FreezeRejected(f"receipt {collection_name} are missing")
        for raw_row in collection:
            row = _mapping(raw_row, collection_name[:-1])
            raw_seed = (
                row.get("qualification_seed")
                if collection_name == "training_receipts"
                else row.get("seed")
            )
            if type(raw_seed) is not int or raw_seed not in qualification_seeds:
                raise FreezeRejected(
                    f"receipt {collection_name} contain an undeclared or reserved seed"
                )


def _validate_receipt(receipt: dict[str, Any]) -> QualificationEvidence:
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
    try:
        evidence = QualificationEvidence.model_validate(receipt.get("qualification_evidence"))
    except ValidationError as error:
        raise FreezeRejected(f"qualification evidence is invalid: {error}") from error
    if evidence.hardware_receipt_sha256 != receipt["hardware_receipt_sha256"]:
        raise FreezeRejected("hardware receipt identity drift prevents qualification freeze")
    _validate_seed_boundary(receipt)
    for required_rows in ("world_decisions", "failure_ledger"):
        if not isinstance(receipt.get(required_rows), list):
            raise FreezeRejected(f"receipt {required_rows} are missing")
    serialized = canonical_json_bytes(receipt).decode("utf-8")
    if any(identifier in serialized for identifier in ('"E01"', '"E02"', '"TEST"')):
        raise FreezeRejected("receipt contains a formal experiment identifier")
    return evidence


def load_active_pointer(artifact_root: Path) -> dict[str, Any]:
    active_path = artifact_root.resolve() / "active.json"
    if not active_path.is_file():
        raise FreezeRejected("Stage1B active pointer does not exist")
    return _mapping(json.loads(active_path.read_text(encoding="utf-8")), "active pointer")


def _remove_internal_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file():
            child.chmod(0o644)
    path.chmod(0o755)
    shutil.rmtree(path)


def _publish_freeze_directory(
    staging_root: Path,
    selected_root: Path,
    active_path: Path,
    active: dict[str, Any],
) -> None:
    backup_root = selected_root.with_name(f".{selected_root.name}.{uuid.uuid4().hex}.backup")
    had_prior = selected_root.exists()
    if had_prior:
        os.replace(selected_root, backup_root)
    try:
        os.replace(staging_root, selected_root)
        _write_json(active_path, active, replace=True)
    except Exception:
        _remove_internal_tree(selected_root)
        if had_prior and backup_root.exists():
            os.replace(backup_root, selected_root)
        raise
    else:
        _remove_internal_tree(backup_root)


def freeze_suite(
    receipt: dict[str, Any],
    artifact_root: Path,
    *,
    series: str = "v2",
    authorization: OverrideAuthorization | None = None,
) -> dict[str, Any]:
    if series != "v2":
        raise FreezeRejected("the active scientific series is fixed to v2")
    evidence = _validate_receipt(receipt)
    root = artifact_root.resolve()
    selected_root = freeze_root(root)
    active_path = root / "active.json"
    prior_pointer = load_active_pointer(root) if active_path.is_file() else None
    if prior_pointer is not None:
        verified = verify_frozen_suite(root)
        if authorization is None:
            raise FreezeRejected("replacing frozen v2 requires user authorization")
        if authorization.prior_manifest_sha256 != verified["manifest_sha256"]:
            raise FreezeRejected("authorization prior manifest does not match the active pointer")
        if prior_pointer.get("series") != series:
            raise FreezeRejected("authorization cannot change the active scientific series")
    elif authorization is not None:
        raise FreezeRejected("initial freeze must not claim an override authorization")
    elif selected_root.exists():
        raise FreezeRejected("frozen v2 exists without an active pointer")

    receipt_payload = canonical_json_bytes(receipt) + b"\n"
    selected_models = {
        str(item["world_id"]): str(item["selected_neural_adapter"])
        for raw_item in receipt["world_decisions"]
        for item in [_mapping(raw_item, "world decision")]
        if item.get("role") == "PRIMARY_MECHANISTIC" and item.get("status") == "PASS"
    }
    manifest: dict[str, Any] = {
        "schema_version": "2.0.0",
        "series": series,
        "qualification_id": receipt.get("qualification_id"),
        "suite_id": receipt.get("suite_id"),
        "suite_gate_status": "PASS",
        "receipt_sha256": sha256_bytes(receipt_payload),
        "source_manifest_sha256": receipt.get("source_manifest_sha256"),
        "source_commits": receipt.get("source_commits"),
        "world_config_sha256": receipt.get("world_config_sha256"),
        "qualification_config_sha256": receipt.get("qualification_config_sha256"),
        "qualification_evidence": evidence.model_dump(mode="json"),
        "selected_models": selected_models,
        "override_authorization": asdict(authorization) if authorization is not None else None,
    }
    selected_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = selected_root.with_name(f".{selected_root.name}.{uuid.uuid4().hex}.tmp")
    try:
        receipt_path = staging_root / "qualification_receipt.json"
        written_receipt = _write_json(receipt_path, receipt, replace=False)
        if sha256_bytes(written_receipt) != manifest["receipt_sha256"]:
            raise FreezeRejected("qualification receipt changed while freezing")
        manifest_path = staging_root / "manifest.json"
        manifest_payload = _write_json(manifest_path, manifest, replace=False)
        manifest_hash = sha256_bytes(manifest_payload)
        hash_path = staging_root / "manifest.sha256"
        hash_path.write_text(f"{manifest_hash}\n", encoding="ascii")
        if sha256_bytes(manifest_path.read_bytes()) != hash_path.read_text(
            encoding="ascii"
        ).strip():
            raise FreezeRejected("staged manifest hash does not match")
        for frozen_path in (receipt_path, manifest_path, hash_path):
            frozen_path.chmod(0o444)
        active = {
            "schema_version": "2.0.0",
            "series": series,
            "manifest_sha256": manifest_hash,
        }
        _publish_freeze_directory(staging_root, selected_root, active_path, active)
    except Exception:
        _remove_internal_tree(staging_root)
        raise
    return manifest


def verify_frozen_suite(
    artifact_root: Path,
    *,
    series: str | None = None,
    allow_unfrozen: bool = False,
) -> dict[str, Any]:
    root = artifact_root.resolve()
    active_path = root / "active.json"
    if not active_path.is_file():
        if allow_unfrozen:
            return {"status": "UNFROZEN", "active_series": None}
        raise FreezeRejected("Stage1B is not frozen")
    active = load_active_pointer(root)
    selected_series = series or str(active.get("series", ""))
    if selected_series != "v2":
        raise FreezeRejected("active scientific series is invalid")
    if active.get("series") != selected_series:
        raise FreezeRejected("active pointer scientific series does not match")
    selected_root = freeze_root(root)
    manifest_path = selected_root / "manifest.json"
    receipt_path = selected_root / "qualification_receipt.json"
    hash_path = selected_root / "manifest.sha256"
    if not manifest_path.is_file() or not receipt_path.is_file() or not hash_path.is_file():
        raise FreezeRejected("frozen revision files are missing")
    manifest_payload = manifest_path.read_bytes()
    expected_hash = hash_path.read_text(encoding="ascii").strip()
    actual_hash = sha256_bytes(manifest_payload)
    if actual_hash != expected_hash:
        raise FreezeRejected("frozen manifest hash does not match")
    if active.get("manifest_sha256") != actual_hash:
        raise FreezeRejected("active pointer manifest hash does not match")
    manifest = _mapping(json.loads(manifest_payload), "frozen manifest")
    if (
        manifest.get("suite_gate_status") != "PASS"
        or manifest.get("series") != selected_series
    ):
        raise FreezeRejected("frozen manifest gate or series is invalid")
    if sha256_bytes(receipt_path.read_bytes()) != manifest.get("receipt_sha256"):
        raise FreezeRejected("frozen qualification receipt hash does not match")
    return {
        "status": "PASS",
        "active_series": active.get("series"),
        "manifest_sha256": actual_hash,
    }
