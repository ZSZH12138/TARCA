from __future__ import annotations

import re
from pathlib import PurePosixPath

from pydantic import field_validator

from .base import Sha256Hash, StrictContractModel


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
