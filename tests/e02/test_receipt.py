from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from tarca.e02.decision import evaluate_e02
from tarca.e02.receipt import E02Receipt, build_e02_receipt, verify_e02_receipt
from tests.e02.test_decision import E02_CONFIG, passing_evidence


def test_e02_receipt_binds_decision_and_complete_evidence() -> None:
    evidence = passing_evidence()
    decision = evaluate_e02(evidence, E02_CONFIG)

    receipt = build_e02_receipt(decision, evidence)

    assert receipt.outcome == "PASS"
    assert verify_e02_receipt(receipt, decision, evidence) == receipt


def test_e02_receipt_hash_changes_when_evidence_changes() -> None:
    evidence = passing_evidence()
    first_decision = evaluate_e02(evidence, E02_CONFIG)
    first = build_e02_receipt(first_decision, evidence)
    changed = replace(evidence, positive_initializations=3)
    changed_decision = evaluate_e02(changed, E02_CONFIG)
    second = build_e02_receipt(changed_decision, changed)

    assert first.evidence_sha256 != second.evidence_sha256
    assert first.receipt_sha256 != second.receipt_sha256


def test_e02_receipt_rejects_tampered_receipt_hash() -> None:
    evidence = passing_evidence()
    decision = evaluate_e02(evidence, E02_CONFIG)
    receipt = build_e02_receipt(decision, evidence)
    payload = receipt.model_dump(mode="json")
    payload["receipt_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="receipt SHA-256"):
        E02Receipt.model_validate(payload)
