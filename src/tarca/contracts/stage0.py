from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

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


class DoctorCheckResults(StrictContractModel):
    pot_sinkhorn: Literal["PASS"] | None = None
    python_version: Literal["PASS"] | None = None
    pyvene_import: Literal["PASS"] | None = None
    torch_basic: Literal["PASS"] | None = None
    torch_hook: Literal["PASS"] | None = None
    workspace_disk: Literal["PASS"] | None = None
    workspace_write: Literal["PASS"] | None = None


class DoctorVersions(StrictContractModel):
    numpy: str | None = None
    pot: str | None = None
    pyvene: str | None = None
    torch: str | None = None


class DoctorResources(StrictContractModel):
    logical_cpu_count: int = Field(ge=1)
    memory_total_bytes: int | None = Field(default=None, ge=1)
    disk_total_bytes: int = Field(ge=1)
    disk_free_bytes: int = Field(ge=1)
    python_version: str
    python_executable: str


class DoctorReport(StrictContractModel):
    status: Literal["PASS", "FAIL"]
    gpu_required: Literal[False]
    checks: DoctorCheckResults
    versions: DoctorVersions
    resources: DoctorResources | None = None
    cuda_available: bool | None = None
    cuda_device_count: int | None = Field(default=None, ge=0)
    default_profile_id: str | None = None
    execution_backend_replaceable: Literal[True] | None = None
    tested_torch_dtypes: tuple[Literal["float32", "float64"], ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _status_fields_are_consistent(self) -> DoctorReport:
        if self.status == "FAIL":
            if self.error is None or not self.error.strip():
                raise ValueError("failed doctor report requires an error")
            return self
        if self.error is not None:
            raise ValueError("passing doctor report cannot contain an error")
        required = (
            self.resources,
            self.cuda_available,
            self.cuda_device_count,
            self.default_profile_id,
            self.execution_backend_replaceable,
        )
        if any(value is None for value in required):
            raise ValueError("passing doctor report is missing required capability fields")
        if any(value != "PASS" for value in self.checks.model_dump().values()):
            raise ValueError("passing doctor report requires every check to pass")
        if not self.tested_torch_dtypes:
            raise ValueError("passing doctor report requires tested torch dtypes")
        return self


class Stage0VerificationReport(StrictContractModel):
    status: Literal["PASS"]
    row_count: int = Field(ge=0)
    unique_work_ids: int = Field(ge=0)
    dependency_release_count: int = Field(ge=0)
    locked_dependency_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    research_contract_status: Literal["FROZEN"]
    gate0_status: Literal["PASS"]
    completion_status: Literal["COMPLETED"]
    doctor: DoctorReport | None = None
