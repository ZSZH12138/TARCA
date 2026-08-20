from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from .artifacts import ArtifactRef
from .base import StrictContractModel, UtcDatetime


class DatasetWindowPartition(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    TEST_SEEN_REGIME = "TEST_SEEN_REGIME"
    TEST_UNSEEN_REGIME = "TEST_UNSEEN_REGIME"


class DatasetSpec(StrictContractModel):
    name: str
    version: str

    @field_validator("name", "version")
    @classmethod
    def _logical_key(cls, value: str) -> str:
        if (
            not value.strip()
            or value in {".", ".."}
            or "\x00" in value
            or "/" in value
            or "\\" in value
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("dataset name and version must be logical keys, not paths")
        return value


class AccessScope(StrictContractModel):
    sealed: bool
    scope_name: str

    @field_validator("scope_name")
    @classmethod
    def _scope_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scope_name must not be blank")
        return value


class SealedAccessGrant(StrictContractModel):
    grant_id: str
    dataset: DatasetSpec
    scope_name: str
    allowed_partitions: tuple[DatasetWindowPartition, ...]
    authorization_ref: ArtifactRef
    issued_at: UtcDatetime
    expires_at: UtcDatetime

    @field_validator("grant_id", "scope_name")
    @classmethod
    def _identifier_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("grant identifiers must not be blank")
        return value

    @field_validator("allowed_partitions")
    @classmethod
    def _partitions_are_nonempty_and_unique(
        cls, value: tuple[DatasetWindowPartition, ...]
    ) -> tuple[DatasetWindowPartition, ...]:
        if not value:
            raise ValueError("allowed_partitions must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("allowed_partitions must be unique")
        return value

    @model_validator(mode="after")
    def _authorization_is_bounded(self) -> Self:
        if self.authorization_ref.artifact_type != "SEALED_ACCESS_AUTHORIZATION":
            raise ValueError("authorization_ref must be a sealed-access authorization")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        return self


def validate_sealed_access(
    dataset: DatasetSpec,
    partition: DatasetWindowPartition,
    access: AccessScope,
    grant: SealedAccessGrant | None,
    accessed_at: datetime,
) -> None:
    """Fail closed before a caller performs a sealed physical read."""
    if not access.sealed:
        return
    if accessed_at.tzinfo is None or accessed_at.utcoffset() != timedelta(0):
        raise PermissionError("sealed access time must be timezone-aware UTC")
    if grant is None:
        raise PermissionError("sealed access requires a grant")
    if grant.dataset != dataset:
        raise PermissionError("sealed access grant dataset mismatch")
    if grant.scope_name != access.scope_name:
        raise PermissionError("sealed access grant scope mismatch")
    if partition not in grant.allowed_partitions:
        raise PermissionError("sealed access grant partition mismatch")
    if accessed_at < grant.issued_at or accessed_at >= grant.expires_at:
        raise PermissionError("sealed access grant is not current")
