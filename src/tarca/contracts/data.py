from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .base import Sha256Hash, StrictContractModel, UtcDatetime
from .data_access import DatasetSpec, DatasetWindowPartition


class SplitPartition(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class DatasetSourceKind(StrEnum):
    STAGE1_SYNTHETIC_CONFIG = "STAGE1_SYNTHETIC_CONFIG"
    PERSISTED_STAGE1 = "PERSISTED_STAGE1"


class DataSplitSummary(StrictContractModel):
    partition: SplitPartition
    split_hash: Sha256Hash
    count: int = Field(ge=0)


class WindowContractSummary(StrictContractModel):
    history_length: int = Field(gt=0)
    horizon: int = Field(gt=0)
    input_feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    observed_covariate_names: tuple[str, ...]
    known_future_covariate_names: tuple[str, ...]
    timezone: Literal["UTC"]
    missingness_protocol: str

    @field_validator(
        "input_feature_names",
        "target_names",
        "observed_covariate_names",
        "known_future_covariate_names",
    )
    @classmethod
    def _names_are_unique_and_nonblank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value):
            raise ValueError("feature names must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("feature names must be unique")
        return value

    @field_validator("input_feature_names", "target_names")
    @classmethod
    def _core_names_are_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("input and target names must not be empty")
        return value

    @field_validator("missingness_protocol")
    @classmethod
    def _missingness_protocol_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("missingness_protocol must not be blank")
        return value

    @model_validator(mode="after")
    def _known_future_does_not_overlap_targets(self) -> Self:
        if set(self.known_future_covariate_names) & set(self.target_names):
            raise ValueError("known-future covariates and targets must not overlap")
        return self


class DatasetRegistryEntry(StrictContractModel):
    dataset: DatasetSpec
    source_kind: DatasetSourceKind
    relative_location: str
    expected_dataset_hash: Sha256Hash
    sealed: bool
    available_partitions: tuple[DatasetWindowPartition, ...]

    @field_validator("relative_location")
    @classmethod
    def _safe_relative_location(cls, value: str) -> str:
        if (
            not value
            or "\\" in value
            or re.match(r"^[A-Za-z]:/", value)
            or PurePosixPath(value).is_absolute()
            or value == "."
            or ".." in PurePosixPath(value).parts
        ):
            raise ValueError("relative_location must be a canonical POSIX relative path")
        return value

    @field_validator("available_partitions")
    @classmethod
    def _partitions_are_nonempty_and_unique(
        cls, value: tuple[DatasetWindowPartition, ...]
    ) -> tuple[DatasetWindowPartition, ...]:
        if not value:
            raise ValueError("available_partitions must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("available_partitions must be unique")
        return value


class DatasetRegistryManifest(StrictContractModel):
    registry_id: str
    registry_version: str
    entries: tuple[DatasetRegistryEntry, ...]

    @field_validator("registry_id", "registry_version")
    @classmethod
    def _identifiers_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("registry identifiers must not be blank")
        return value

    @field_validator("entries")
    @classmethod
    def _dataset_identities_are_unique(
        cls, value: tuple[DatasetRegistryEntry, ...]
    ) -> tuple[DatasetRegistryEntry, ...]:
        identities = tuple((entry.dataset.name, entry.dataset.version) for entry in value)
        if len(set(identities)) != len(identities):
            raise ValueError("dataset identities must be unique")
        return value


class DataManifest(StrictContractModel):
    schema_version: Literal["1.0.0"]
    dataset_name: str
    dataset_version: str
    dataset_hash: Sha256Hash
    splits: tuple[DataSplitSummary, ...]
    window_contract: WindowContractSummary
    source_description: str
    created_at: UtcDatetime

    @field_validator("dataset_name", "dataset_version")
    @classmethod
    def _dataset_identity_is_logical(cls, value: str) -> str:
        if (
            not value.strip()
            or value in {".", ".."}
            or "\x00" in value
            or "/" in value
            or "\\" in value
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("dataset identity must be a logical key")
        return value

    @field_validator("splits")
    @classmethod
    def _splits_are_nonempty_and_unique(
        cls, value: tuple[DataSplitSummary, ...]
    ) -> tuple[DataSplitSummary, ...]:
        if not value:
            raise ValueError("splits must not be empty")
        partitions = tuple(split.partition for split in value)
        if len(set(partitions)) != len(partitions):
            raise ValueError("split partitions must be unique")
        return value

    @field_validator("source_description")
    @classmethod
    def _source_description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_description must not be blank")
        return value


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    passed: bool
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if not isinstance(self.findings, tuple):
            raise TypeError("findings must be a tuple")
        if any(not isinstance(finding, str) or not finding.strip() for finding in self.findings):
            raise ValueError("findings must contain nonblank strings")
        if len(set(self.findings)) != len(self.findings):
            raise ValueError("findings must be unique")
        if self.passed and self.findings:
            raise ValueError("leakage audit cannot pass with findings")
        if not self.passed and not self.findings:
            raise ValueError("failed leakage audit must contain findings")
