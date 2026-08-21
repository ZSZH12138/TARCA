from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from tarca.contracts.base import Sha256Hash, StrictContractModel, UtcDatetime
from tarca.contracts.data_access import DatasetSpec, DatasetWindowPartition

PayloadRole = Literal[
    "x",
    "y",
    "observed_covariates",
    "known_future_covariates",
    "x_observed_mask",
    "y_observed_mask",
    "observed_covariates_mask",
    "known_future_covariates_mask",
    "regime",
    "metadata",
]


class PersistedPayloadFile(StrictContractModel):
    role: PayloadRole
    relative_path: str
    content_hash: Sha256Hash
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or re.match(r"^[A-Za-z]:/", value)
            or path.is_absolute()
            or value == "."
            or ".." in path.parts
        ):
            raise ValueError("payload path must be a canonical POSIX relative path")
        return value

    @model_validator(mode="after")
    def _extension_matches_role(self) -> Self:
        suffix = PurePosixPath(self.relative_path).suffix
        if self.role == "metadata" and suffix != ".json":
            raise ValueError("metadata payload must use a .json file")
        if self.role != "metadata" and suffix != ".npy":
            raise ValueError("array payloads must use individual .npy files")
        return self


class PersistedPartitionPayload(StrictContractModel):
    partition: DatasetWindowPartition
    files: tuple[PersistedPayloadFile, ...]

    @model_validator(mode="after")
    def _roles_are_complete_and_unique(self) -> Self:
        roles = tuple(file.role for file in self.files)
        if len(set(roles)) != len(roles):
            raise ValueError("payload file roles must be unique")
        paths = tuple(file.relative_path for file in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("payload file paths must be unique")
        if not {"x", "metadata"}.issubset(roles):
            raise ValueError("persisted partition requires x and metadata files")
        return self


class PersistedDatasetPayloadManifest(StrictContractModel):
    schema_version: Literal["1.0.0"]
    dataset: DatasetSpec
    partitions: tuple[PersistedPartitionPayload, ...]

    @field_validator("partitions")
    @classmethod
    def _partitions_are_nonempty_and_unique(
        cls, value: tuple[PersistedPartitionPayload, ...]
    ) -> tuple[PersistedPartitionPayload, ...]:
        if not value:
            raise ValueError("payload manifest requires at least one partition")
        partitions = tuple(payload.partition for payload in value)
        if len(set(partitions)) != len(partitions):
            raise ValueError("payload partitions must be unique")
        return value


class PersistedWindowMetadata(StrictContractModel):
    window_id: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    observed_covariate_names: tuple[str, ...]
    known_future_covariate_names: tuple[str, ...]
    feature_start: tuple[UtcDatetime, ...]
    feature_end: tuple[UtcDatetime, ...]
    prediction_start: tuple[UtcDatetime, ...]
    label_end: tuple[UtcDatetime, ...]
    forecast_time: tuple[tuple[UtcDatetime, ...], ...]
    metadata: dict[str, JsonValue]

    @field_validator(
        "window_id",
        "input_feature_names",
        "target_names",
        "observed_covariate_names",
        "known_future_covariate_names",
    )
    @classmethod
    def _logical_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("persisted window names must be nonblank and unique")
        return value
