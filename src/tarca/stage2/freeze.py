from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from tarca.contracts import Sha256Hash, StrictContractModel, canonical_json_hash
from tarca.stage1b.evidence_io import write_canonical_json
from tarca.stage2.manifest import Stage2Manifest, stage2_manifest_from_payload


class Stage2FreezeRejected(RuntimeError):
    """Raised when Stage 2 evidence cannot be frozen or verified."""


class Stage2FreezeReceipt(StrictContractModel):
    schema_version: Literal["tarca-stage2-freeze-v1"]
    status: Literal["FROZEN"]
    manifest_sha256: Sha256Hash
    scientific_sha256: Sha256Hash
    strongest_linear_model_id: Literal["VAR", "DLINEAR"]
    primary_itransformer_seed: int
    formal_access_event_count: Literal[0]
    receipt_sha256: Sha256Hash

    @model_validator(mode="after")
    def _receipt_hash_matches(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != canonical_json_hash(payload):
            raise ValueError("Stage 2 freeze receipt SHA-256 does not match")
        return self


def _freeze_root(artifact_root: Path) -> Path:
    return artifact_root.resolve() / "frozen" / "v1"


def freeze_stage2_suite(
    artifact_root: Path,
    manifest: Stage2Manifest,
) -> Stage2FreezeReceipt:
    if manifest.runtime_failure_ids:
        raise Stage2FreezeRejected("Stage 2 runtime failures prevent freeze")
    if manifest.formal_access_event_count != 0:
        raise Stage2FreezeRejected("formal access events prevent Stage 2 freeze")
    root = _freeze_root(artifact_root)
    if root.exists():
        raise Stage2FreezeRejected("Stage 2 frozen suite already exists")
    manifest_payload = manifest.payload()
    manifest_sha256 = canonical_json_hash(manifest_payload)
    receipt_payload = {
        "schema_version": "tarca-stage2-freeze-v1",
        "status": "FROZEN",
        "manifest_sha256": manifest_sha256,
        "scientific_sha256": manifest.scientific_sha256,
        "strongest_linear_model_id": manifest.strongest_linear.model_id,
        "primary_itransformer_seed": manifest.primary_itransformer.seed,
        "formal_access_event_count": 0,
    }
    receipt = Stage2FreezeReceipt.model_validate(
        {**receipt_payload, "receipt_sha256": canonical_json_hash(receipt_payload)}
    )
    root.mkdir(parents=True)
    try:
        write_canonical_json(root / "stage2_manifest.json", manifest_payload, replace=False)
        write_canonical_json(
            root / "stage2_freeze_receipt.json",
            receipt.model_dump(mode="json"),
            replace=False,
        )
    except Exception as error:
        raise Stage2FreezeRejected(f"cannot write Stage 2 freeze: {error}") from error
    return receipt


def verify_frozen_stage2_suite(artifact_root: Path) -> Stage2FreezeReceipt:
    root = _freeze_root(artifact_root)
    manifest_path = root / "stage2_manifest.json"
    receipt_path = root / "stage2_freeze_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise Stage2FreezeRejected("frozen Stage 2 files are missing")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = Stage2FreezeReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        manifest = stage2_manifest_from_payload(manifest_payload)
    except Exception as error:
        raise Stage2FreezeRejected(f"frozen Stage 2 manifest is invalid: {error}") from error
    if canonical_json_hash(manifest_payload) != receipt.manifest_sha256:
        raise Stage2FreezeRejected("frozen Stage 2 manifest SHA-256 does not match")
    if manifest.scientific_sha256 != receipt.scientific_sha256:
        raise Stage2FreezeRejected("frozen Stage 2 scientific identity does not match")
    if (
        manifest.strongest_linear.model_id != receipt.strongest_linear_model_id
        or manifest.primary_itransformer.seed != receipt.primary_itransformer_seed
    ):
        raise Stage2FreezeRejected("frozen Stage 2 selection identity does not match")
    return receipt

