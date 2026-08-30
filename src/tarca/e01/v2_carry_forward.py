from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import field_validator, model_validator

from tarca.contracts import (
    Sha256Hash,
    StrictContractModel,
    canonical_json_bytes,
    canonical_json_hash,
)
from tarca.e01.v2_config import E01V2Config


class CarryForwardVerificationError(RuntimeError):
    """Raised when immutable E01-v1 evidence cannot prove E01-B passed."""


class E01BControlCount(StrictContractModel):
    control: Literal["RANDOM_CONCEPT", "WRONG_LAG", "WRONG_SCM"]
    seed_count: Literal[5]


class E01BCarryForwardReceipt(StrictContractModel):
    schema_version: Literal["tarca-e01-b-carry-forward-v2"]
    status: Literal["PASS"]
    source_report_sha256: Sha256Hash
    source_recovery_validation_sha256: Sha256Hash
    source_archive_sha256: Sha256Hash
    source_scientific_config_sha256: Sha256Hash
    stage1b_manifest_sha256: Sha256Hash
    v1_overall_gate_status: Literal["FAIL"]
    e01_b_convergence_seed_count: Literal[5]
    directional_seed_counts: tuple[E01BControlCount, ...]
    e01_b_formal_seed_count: Literal[5]
    runtime_alert_count: Literal[0]
    receipt_sha256: Sha256Hash

    @field_validator("directional_seed_counts", mode="before")
    @classmethod
    def _list_becomes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _receipt_is_complete_and_sealed(self) -> Self:
        if tuple(item.control for item in self.directional_seed_counts) != (
            "RANDOM_CONCEPT",
            "WRONG_LAG",
            "WRONG_SCM",
        ):
            raise ValueError("carry-forward controls must be complete and ordered")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != canonical_json_hash(payload):
            raise ValueError("carry-forward receipt SHA-256 is invalid")
        return self


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CarryForwardVerificationError(
            "carry-forward evidence escaped repository root"
        ) from error
    if not path.is_file():
        raise CarryForwardVerificationError("carry-forward evidence file is missing")
    return path


def _mapping_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CarryForwardVerificationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CarryForwardVerificationError(f"{label} must contain a JSON object")
    return value


def _history_record(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CarryForwardVerificationError("E01-v1 history record could not be read") from error
    record = _mapping_bytes(raw, "E01-v1 history record")
    if record.get("schema_version") != "tarca-e01-v1-history-record-v1":
        raise CarryForwardVerificationError("E01-v1 history record schema is invalid")
    expected = record.get("record_sha256")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    canonical = canonical_json_bytes({**unsigned, "record_sha256": expected}) + b"\n"
    if expected != canonical_json_hash(unsigned) or raw != canonical:
        raise CarryForwardVerificationError("E01-v1 history record SHA-256 is invalid")
    if record.get("overall_gate_status") != "FAIL":
        raise CarryForwardVerificationError("E01-v1 overall failure history was not preserved")
    return record


def _embedded_mapping(
    record: dict[str, object],
    name: str,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    evidence = _require_dict(record.get("original_evidence"), "original E01-v1 evidence")
    entry = _require_dict(evidence.get(name), label)
    if entry.get("encoding") != "base64" or entry.get("media_type") != "application/json":
        raise CarryForwardVerificationError(f"{label} encoding metadata is invalid")
    try:
        raw = base64.b64decode(str(entry.get("content_base64", "")), validate=True)
    except (ValueError, binascii.Error) as error:
        raise CarryForwardVerificationError(f"{label} Base64 payload is invalid") from error
    observed = hashlib.sha256(raw).hexdigest()
    if entry.get("sha256") != observed or observed != expected_sha256:
        raise CarryForwardVerificationError(f"{label} SHA-256 drifted")
    return _mapping_bytes(raw, label)


def _require_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CarryForwardVerificationError(f"{label} is missing or invalid")
    return value


def verify_e01_b_carry_forward(
    repository_root: Path,
    config: E01V2Config,
) -> E01BCarryForwardReceipt:
    root = repository_root.resolve()
    frozen = config.carry_forward
    report_path = _inside(root, frozen.report_path)
    recovery_path = _inside(root, frozen.recovery_validation_path)
    if report_path != recovery_path:
        raise CarryForwardVerificationError("E01-v1 evidence must use one history record")
    history = _history_record(report_path)
    report = _embedded_mapping(
        history,
        "final_report",
        frozen.report_sha256,
        "E01-v1 final report",
    )
    recovery = _embedded_mapping(
        history,
        "recovery_validation",
        frozen.recovery_validation_sha256,
        "E01-v1 recovery validation",
    )
    gate = _require_dict(report.get("gate"), "E01-v1 gate")
    convergence = _require_dict(gate.get("convergence_seed_counts"), "convergence counts")
    directional = _require_dict(gate.get("directional_seed_counts"), "directional counts")
    l96_directional = _require_dict(directional.get("lorenz96_twoscale_v2"), "E01-B controls")
    convergence_count = convergence.get("lorenz96_twoscale_v2")
    if convergence_count != frozen.required_convergence_seed_count:
        raise CarryForwardVerificationError("E01-B convergence evidence did not pass five seeds")
    controls = ("RANDOM_CONCEPT", "WRONG_LAG", "WRONG_SCM")
    if any(
        l96_directional.get(control) != frozen.required_directional_seed_count
        for control in controls
    ):
        raise CarryForwardVerificationError("E01-B directional control evidence is incomplete")

    seed_world_reports = report.get("seed_world_reports")
    if not isinstance(seed_world_reports, list):
        raise CarryForwardVerificationError("E01-v1 seed-world reports are missing")
    l96_reports = tuple(
        item
        for item in seed_world_reports
        if isinstance(item, dict) and item.get("world_id") == "lorenz96_twoscale_v2"
    )
    l96_seeds = tuple(item.get("seed") for item in l96_reports)
    if len(l96_reports) != 5 or len(set(l96_seeds)) != 5:
        raise CarryForwardVerificationError("E01-B must contain five unique formal seed reports")

    stage1b = _require_dict(recovery.get("stage1b_freeze"), "Stage1B freeze evidence")
    attempts = _require_dict(recovery.get("attempt_counts"), "runtime attempt counts")
    checks = (
        recovery.get("aggregate_artifact_sha256") == frozen.report_sha256,
        recovery.get("archive_sha256") == frozen.expected_archive_sha256,
        recovery.get("scientific_config_sha256") == frozen.expected_scientific_config_sha256,
        report.get("scientific_config_sha256") == frozen.expected_scientific_config_sha256,
        stage1b.get("manifest_sha256") == frozen.expected_stage1b_manifest_sha256,
        stage1b.get("status") == "PASS",
        recovery.get("sqlite_integrity_ok") is True,
        recovery.get("planned_tasks") == 166,
        attempts == {"COMPLETED": 166},
        recovery.get("seed_world_report_count") == 10,
    )
    if not all(checks):
        raise CarryForwardVerificationError("E01-v1 recovery identity or completeness drifted")
    if recovery.get("alert_count") != 0:
        raise CarryForwardVerificationError("E01-v1 recovery contains runtime alerts")
    if gate.get("status") != "FAIL":
        raise CarryForwardVerificationError("E01-v1 overall failure history was not preserved")

    payload = {
        "schema_version": "tarca-e01-b-carry-forward-v2",
        "status": "PASS",
        "source_report_sha256": frozen.report_sha256,
        "source_recovery_validation_sha256": frozen.recovery_validation_sha256,
        "source_archive_sha256": frozen.expected_archive_sha256,
        "source_scientific_config_sha256": frozen.expected_scientific_config_sha256,
        "stage1b_manifest_sha256": frozen.expected_stage1b_manifest_sha256,
        "v1_overall_gate_status": "FAIL",
        "e01_b_convergence_seed_count": convergence_count,
        "directional_seed_counts": tuple(
            {"control": control, "seed_count": l96_directional[control]} for control in controls
        ),
        "e01_b_formal_seed_count": len(l96_reports),
        "runtime_alert_count": 0,
    }
    return E01BCarryForwardReceipt.model_validate(
        {**payload, "receipt_sha256": canonical_json_hash(payload)}
    )
