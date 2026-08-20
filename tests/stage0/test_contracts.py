from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tarca.contracts import (
    ArtifactRef,
    GateDecision,
    GateStatus,
    ResearchContractManifest,
    build_gate_spec,
    validate_gate_decision,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_artifact_identity_ignores_path() -> None:
    first = ArtifactRef(
        artifact_id="artifact-a",
        artifact_type="PREREGISTRATION",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path="docs/preregistration_v0.md",
    )
    second = ArtifactRef(
        artifact_id="artifact-b",
        artifact_type="PREREGISTRATION",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path="archive/preregistration_v0.md",
    )

    assert first.identity_key() == second.identity_key()


@pytest.mark.parametrize("path", ["../secret", "/absolute/path", "C:/absolute/path"])
def test_artifact_ref_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_id="unsafe",
            artifact_type="TEST",
            content_hash=HASH_A,
            schema_version="1.0.0",
            relative_path=path,
        )


def test_strict_contract_rejects_extra_fields_and_mutation() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(
            {
                "artifact_id": "extra",
                "artifact_type": "TEST",
                "content_hash": HASH_A,
                "schema_version": "1.0.0",
                "relative_path": None,
                "unexpected": True,
            }
        )

    artifact = ArtifactRef(
        artifact_id="frozen",
        artifact_type="TEST",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path=None,
    )
    with pytest.raises(ValidationError):
        artifact.artifact_id = "changed"  # type: ignore[misc]


def test_research_contract_requires_utc_datetime() -> None:
    artifact = ArtifactRef(
        artifact_id="a",
        artifact_type="TEST",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path=None,
    )
    payload = {
        "schema_version": "1.0.0",
        "protocol_id": "TARCA-E2E-STAGE-PROTOCOL-2.0",
        "preregistration_ref": artifact,
        "novelty_claims_ref": artifact,
        "assumption_ledger_ref": artifact,
        "terminology_ref": artifact,
        "environment_lock_ref": artifact,
        "related_work_ref": artifact,
        "created_at": datetime(2026, 8, 20, 0, 0),
        "status": "FROZEN",
    }

    with pytest.raises(ValidationError):
        ResearchContractManifest.model_validate(payload)

    payload["created_at"] = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    assert ResearchContractManifest.model_validate(payload).status == "FROZEN"


def test_gate_decision_must_bind_gate_spec_hash() -> None:
    spec = build_gate_spec(
        gate_id="GATE_A",
        protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
        gate_version="1.0.0",
        predicates=(),
        required_evidence_types=("NOVELTY_CLAIMS", "RELATED_WORK_BUNDLE"),
        failure_action="DROP_CLAIM",
    )
    spec_ref = ArtifactRef(
        artifact_id="gate-a-spec",
        artifact_type="GATE_SPEC",
        content_hash=spec.spec_hash,
        schema_version=spec.gate_version,
        relative_path="artifacts/gates/gate_a_spec.json",
    )
    novelty_ref = ArtifactRef(
        artifact_id="novelty",
        artifact_type="NOVELTY_CLAIMS",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path="docs/novelty_claims.md",
    )
    related_ref = ArtifactRef(
        artifact_id="related",
        artifact_type="RELATED_WORK_BUNDLE",
        content_hash=HASH_B,
        schema_version="1.0.0",
        relative_path="artifacts/stage0/related_work_bundle.json",
    )
    decision = GateDecision(
        gate_id=spec.gate_id,
        status=GateStatus.PASS,
        rationale="Narrow claims remain falsifiable.",
        evidence=(spec_ref, novelty_ref, related_ref),
    )

    assert validate_gate_decision(spec, decision) is decision

    wrong_ref = spec_ref.model_copy(update={"content_hash": HASH_B})
    wrong_decision = decision.model_copy(update={"evidence": (wrong_ref, novelty_ref, related_ref)})
    with pytest.raises(ValueError, match="GateSpec hash"):
        validate_gate_decision(spec, wrong_decision)
