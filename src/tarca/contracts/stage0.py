from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .artifacts import ArtifactRef
from .base import Sha256Hash, StrictContractModel, UtcDatetime


class EnvironmentPlatform(StrictContractModel):
    system: str
    release: str
    machine: str


class PythonEnvironment(StrictContractModel):
    version: str
    implementation: str
    executable: str


class EnvironmentResources(StrictContractModel):
    logical_cpu_count: int = Field(ge=1)
    memory_total_bytes: int | None = Field(default=None, ge=1)
    disk_total_bytes: int = Field(ge=1)
    disk_free_bytes: int = Field(ge=1)


class AcceleratorCapabilities(StrictContractModel):
    cuda_available: bool
    cuda_device_count: int = Field(ge=0)
    mps_available: bool
    tested_torch_dtypes: tuple[Literal["float32", "float64"], ...]


class EnvironmentProfile(StrictContractModel):
    schema_version: str
    profile_id: str
    profile_role: Literal["DEFAULT_EXECUTION_PROFILE"]
    execution_backend_replaceable: Literal[True]
    compute_boundary_fixed: Literal[False]
    platform: EnvironmentPlatform
    python: PythonEnvironment
    resources: EnvironmentResources
    accelerators: AcceleratorCapabilities


class EnvironmentBundle(StrictContractModel):
    schema_version: str
    pyproject_ref: ArtifactRef
    lock_ref: ArtifactRef
    profile_ref: ArtifactRef
    default_profile_id: str
    profile_selection_policy: Literal["DEFAULT_WITH_RUNTIME_OVERRIDE"]


class RelatedWorkBundle(StrictContractModel):
    schema_version: str
    matrix_ref: ArtifactRef
    third_party_versions_ref: ArtifactRef


class ArtifactIndex(StrictContractModel):
    schema_version: str
    artifacts: tuple[ArtifactRef, ...]


class AuthorizedOverwriteReceipt(StrictContractModel):
    schema_version: str
    action: Literal["AUTHORIZED_FROZEN_OVERWRITE"]
    authorization_reason: str
    archived_previous_artifacts_at: str
    previous_manifest_hash: Sha256Hash
    replacement_manifest_hash: Sha256Hash

    @field_validator("authorization_reason", "archived_previous_artifacts_at")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class Stage0CompletionReceipt(StrictContractModel):
    schema_version: str
    status: Literal["COMPLETED"]
    completed_at: UtcDatetime
    research_contract_ref: ArtifactRef
    gate_decision_ref: ArtifactRef
    artifact_index_ref: ArtifactRef
