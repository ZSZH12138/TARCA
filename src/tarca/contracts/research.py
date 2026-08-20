from __future__ import annotations

from typing import Literal

from .artifacts import ArtifactRef
from .base import Sha256Hash, StrictContractModel, UtcDatetime, canonical_json_hash


class ResearchContractManifest(StrictContractModel):
    schema_version: str
    protocol_id: str
    preregistration_ref: ArtifactRef
    novelty_claims_ref: ArtifactRef
    assumption_ledger_ref: ArtifactRef
    terminology_ref: ArtifactRef
    environment_lock_ref: ArtifactRef
    related_work_ref: ArtifactRef
    created_at: UtcDatetime
    status: Literal["FROZEN", "SUPERSEDED"]


def validate_research_contract(
    value: ResearchContractManifest,
) -> ResearchContractManifest:
    return ResearchContractManifest.model_validate(value)


def research_contract_hash(value: ResearchContractManifest) -> Sha256Hash:
    return canonical_json_hash(value)
