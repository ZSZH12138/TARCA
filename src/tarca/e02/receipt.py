from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from tarca.contracts import Sha256Hash, StrictContractModel, canonical_json_hash
from tarca.e02.decision import E02Decision, E02Evidence, E02Outcome


class E02Receipt(StrictContractModel):
    schema_version: Literal["tarca-e02-receipt-v1"]
    outcome: E02Outcome
    e02_config_sha256: Sha256Hash
    stage2_freeze_receipt_sha256: Sha256Hash
    evidence_sha256: Sha256Hash
    decision_sha256: Sha256Hash
    receipt_sha256: Sha256Hash

    @model_validator(mode="after")
    def _receipt_hash_matches(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != canonical_json_hash(payload):
            raise ValueError("E02 receipt SHA-256 does not match")
        return self


def build_e02_receipt(decision: E02Decision, evidence: E02Evidence) -> E02Receipt:
    payload = {
        "schema_version": "tarca-e02-receipt-v1",
        "outcome": decision.outcome,
        "e02_config_sha256": evidence.e02_config_sha256,
        "stage2_freeze_receipt_sha256": evidence.stage2_freeze_receipt_sha256,
        "evidence_sha256": evidence.evidence_sha256(),
        "decision_sha256": decision.decision_sha256(),
    }
    return E02Receipt.model_validate(
        {**payload, "receipt_sha256": canonical_json_hash(payload)}
    )


def verify_e02_receipt(
    receipt: E02Receipt,
    decision: E02Decision,
    evidence: E02Evidence,
) -> E02Receipt:
    expected = build_e02_receipt(decision, evidence)
    if receipt != expected:
        raise ValueError("E02 receipt does not match the supplied evidence and decision")
    return receipt

