from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

from .base import GitCommit, Sha256Hash, StrictContractModel, UtcDatetime


class ArtifactRef(StrictContractModel):
    artifact_id: str
    artifact_type: str
    content_hash: Sha256Hash
    schema_version: str
    relative_path: str | None

    @field_validator("artifact_id", "artifact_type", "schema_version")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value or "\\" in value or re.match(r"^[A-Za-z]:/", value):
            raise ValueError("relative_path must be a canonical POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay inside the repository")
        return value

    def identity_key(self) -> tuple[str, str, str]:
        return (self.artifact_type, self.content_hash, self.schema_version)


class ArtifactManifest(StrictContractModel):
    artifact: ArtifactRef
    media_type: str
    serializer_id: str
    producer_stage: str
    producer_task_id: str
    scientific_identity_hash: Sha256Hash
    dependencies: tuple[ArtifactRef, ...]
    size_bytes: int = Field(ge=0)
    created_at: UtcDatetime

    @field_validator("media_type", "serializer_id", "producer_stage", "producer_task_id")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact manifest text fields must not be blank")
        return value

    @field_validator("dependencies")
    @classmethod
    def _dependencies_are_unique(cls, value: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        identities = tuple(dependency.identity_key() for dependency in value)
        if len(set(identities)) != len(identities):
            raise ValueError("artifact dependencies must be unique")
        return value


class RunManifest(StrictContractModel):
    experiment_id: str
    run_id: str
    config_hash: Sha256Hash
    data_hash: Sha256Hash
    git_commit: GitCommit
    schema_version: Literal["1.0.0"]
    created_at: UtcDatetime
    status: str

    @field_validator("experiment_id", "run_id", "status")
    @classmethod
    def _logical_text_not_blank(cls, value: str) -> str:
        if (
            not value.strip()
            or value in {".", ".."}
            or "\x00" in value
            or "/" in value
            or "\\" in value
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("run identifiers and status must be nonblank logical values")
        return value
