from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from .artifacts import ArtifactRef
from .base import Sha256Hash, StrictContractModel, canonical_json_hash


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class GatePredicate(StrictContractModel):
    predicate_id: str
    metric_name: str
    scope: Mapping[str, str | int | float | bool]
    operator: Literal[">", ">=", "<", "<=", "=="]
    threshold: float
    aggregation: Literal["ALL", "MEAN", "WORST", "MEDIAN"]


class GateSpec(StrictContractModel):
    gate_id: str
    protocol_id: str
    gate_version: str
    predicates: tuple[GatePredicate, ...]
    required_evidence_types: tuple[str, ...]
    failure_action: Literal["REPAIR", "STOP", "DROP_CLAIM", "CONTINUE_EXPLORATORY"]
    spec_hash: Sha256Hash

    @model_validator(mode="after")
    def _hash_matches_payload(self) -> GateSpec:
        if self.spec_hash != gate_spec_payload_hash(self):
            raise ValueError("spec_hash does not match GateSpec payload")
        return self


class GateDecision(StrictContractModel):
    gate_id: str
    status: GateStatus
    rationale: str
    evidence: tuple[ArtifactRef, ...]


def gate_spec_payload(spec: GateSpec) -> dict[str, object]:
    return spec.model_dump(mode="json", exclude={"spec_hash"})


def gate_spec_payload_hash(spec: GateSpec) -> Sha256Hash:
    return canonical_json_hash(gate_spec_payload(spec))


def build_gate_spec(
    *,
    gate_id: str,
    protocol_id: str,
    gate_version: str,
    predicates: Sequence[GatePredicate],
    required_evidence_types: Sequence[str],
    failure_action: Literal["REPAIR", "STOP", "DROP_CLAIM", "CONTINUE_EXPLORATORY"],
) -> GateSpec:
    predicate_tuple = tuple(predicates)
    evidence_type_tuple = tuple(required_evidence_types)
    payload = {
        "gate_id": gate_id,
        "protocol_id": protocol_id,
        "gate_version": gate_version,
        "predicates": [item.model_dump(mode="json") for item in predicate_tuple],
        "required_evidence_types": list(evidence_type_tuple),
        "failure_action": failure_action,
    }
    return GateSpec(
        gate_id=gate_id,
        protocol_id=protocol_id,
        gate_version=gate_version,
        predicates=predicate_tuple,
        required_evidence_types=evidence_type_tuple,
        failure_action=failure_action,
        spec_hash=canonical_json_hash(payload),
    )


def validate_gate_decision(spec: GateSpec, decision: GateDecision) -> GateDecision:
    if decision.gate_id != spec.gate_id:
        raise ValueError("GateDecision gate_id does not match GateSpec")
    spec_receipts = [
        item
        for item in decision.evidence
        if item.artifact_type == "GATE_SPEC" and item.content_hash == spec.spec_hash
    ]
    if not spec_receipts:
        raise ValueError("GateDecision does not bind the GateSpec hash")
    evidence_types = {item.artifact_type for item in decision.evidence}
    missing = set(spec.required_evidence_types) - evidence_types
    if missing:
        raise ValueError(f"GateDecision is missing evidence types: {sorted(missing)}")
    return decision
