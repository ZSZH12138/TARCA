from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from math import isfinite
from pathlib import PurePosixPath
from typing import Literal, Self

import torch
from pydantic import Field, field_validator, model_validator
from torch import Tensor

from .base import JSONValue, Sha256Hash, StrictContractModel, UtcDatetime
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
class WindowBatch:
    x: Tensor
    y: Tensor | None
    observed_covariates: Tensor | None
    known_future_covariates: Tensor | None
    x_observed_mask: Tensor | None
    y_observed_mask: Tensor | None
    observed_covariates_mask: Tensor | None
    known_future_covariates_mask: Tensor | None
    regime: Tensor | None
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
    metadata: Mapping[str, JSONValue]


def _require_shape(tensor: Tensor, expected: tuple[int, ...], name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape must be {expected}, got {tuple(tensor.shape)}")


def _require_optional_rank_three(tensor: Tensor | None, name: str) -> None:
    if tensor is not None and (not isinstance(tensor, Tensor) or tensor.ndim != 3):
        raise ValueError(f"{name} must be a rank-3 Tensor")


def _validate_data_tensor(
    tensor: Tensor,
    expected: tuple[int, ...],
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    _require_shape(tensor, expected, name)
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must have floating dtype")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must match x dtype")
    if tensor.device != device:
        raise ValueError(f"{name} device must match x device")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_optional_data(
    tensor: Tensor | None,
    expected: tuple[int, ...],
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor is not None:
        _validate_data_tensor(tensor, expected, name, dtype, device)


def _validate_mask(
    mask: Tensor | None,
    data: Tensor | None,
    name: str,
    device: torch.device,
) -> None:
    if data is None:
        if mask is not None:
            raise ValueError(f"{name} cannot exist without its data tensor")
        return
    if mask is None:
        return
    _require_shape(mask, tuple(data.shape), name)
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must have bool dtype")
    if mask.device != device:
        raise ValueError(f"{name} device must match x device")


def _validate_names(names: tuple[str, ...], size: int, label: str) -> None:
    if len(names) != size:
        raise ValueError(f"{label} count must match its feature axis")
    if any(not name.strip() for name in names):
        raise ValueError(f"{label} must not contain blank names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique")


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == timedelta(0)


def _validate_times(batch: WindowBatch, batch_size: int, horizon: int) -> None:
    axes = (batch.feature_start, batch.feature_end, batch.prediction_start, batch.label_end)
    if any(len(axis) != batch_size for axis in axes) or len(batch.forecast_time) != batch_size:
        raise ValueError("time axes must match batch size")
    for row in range(batch_size):
        boundaries = tuple(axis[row] for axis in axes)
        times = batch.forecast_time[row]
        if not all(_is_utc(value) for value in (*boundaries, *times)):
            raise ValueError("all window times must be timezone-aware UTC")
        if not boundaries[0] <= boundaries[1] < boundaries[2] <= boundaries[3]:
            raise ValueError("window time boundaries are out of order")
        if len(times) != horizon:
            raise ValueError("forecast_time horizon must match tensor horizon")
        if any(left >= right for left, right in pairwise(times)):
            raise ValueError("forecast_time must be strictly increasing")
        if times and not boundaries[2] <= times[0] <= times[-1] <= boundaries[3]:
            raise ValueError("forecast_time must stay inside the prediction interval")


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _validate_window_metadata(metadata: Mapping[str, JSONValue]) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if not all(isinstance(key, str) and _is_json_value(value) for key, value in metadata.items()):
        raise ValueError("metadata must contain JSON-compatible finite values")


def _window_dimensions(batch: WindowBatch) -> tuple[int, int, int, int, int, int, int]:
    if not isinstance(batch.x, Tensor) or batch.x.ndim != 3:
        raise ValueError("x must be a rank-3 Tensor")
    batch_size, history, input_size = batch.x.shape
    if min(batch_size, history, input_size) <= 0:
        raise ValueError("x dimensions must be positive")
    if not batch.x.is_floating_point() or not bool(torch.isfinite(batch.x).all()):
        raise ValueError("x must have floating dtype and contain only finite values")
    _require_optional_rank_three(batch.y, "y")
    _require_optional_rank_three(batch.observed_covariates, "observed_covariates")
    _require_optional_rank_three(batch.known_future_covariates, "known_future_covariates")
    horizon = batch.y.shape[1] if batch.y is not None else len(batch.forecast_time[0])
    if horizon <= 0:
        raise ValueError("forecast horizon must be positive")
    target_size = batch.y.shape[2] if batch.y is not None else len(batch.target_names)
    observed_size = (
        batch.observed_covariates.shape[2] if batch.observed_covariates is not None else 0
    )
    future_size = (
        batch.known_future_covariates.shape[2] if batch.known_future_covariates is not None else 0
    )
    return batch_size, history, input_size, horizon, target_size, observed_size, future_size


def _validate_window_names(
    batch: WindowBatch,
    batch_size: int,
    input_size: int,
    target_size: int,
    observed_size: int,
    future_size: int,
) -> None:
    _validate_names(batch.window_id, batch_size, "window_id")
    _validate_names(batch.input_feature_names, input_size, "input_feature_names")
    _validate_names(batch.target_names, target_size, "target_names")
    _validate_names(batch.observed_covariate_names, observed_size, "observed_covariate_names")
    _validate_names(batch.known_future_covariate_names, future_size, "known_future_covariate_names")
    if set(batch.known_future_covariate_names) & set(batch.target_names):
        raise ValueError("known-future covariates and targets must not overlap")


def _validate_window_tensors(
    batch: WindowBatch,
    batch_size: int,
    history: int,
    horizon: int,
    target_size: int,
    observed_size: int,
    future_size: int,
) -> None:
    _validate_optional_data(
        batch.y, (batch_size, horizon, target_size), "y", batch.x.dtype, batch.x.device
    )
    _validate_optional_data(
        batch.observed_covariates,
        (batch_size, history, observed_size),
        "observed_covariates",
        batch.x.dtype,
        batch.x.device,
    )
    _validate_optional_data(
        batch.known_future_covariates,
        (batch_size, horizon, future_size),
        "known_future_covariates",
        batch.x.dtype,
        batch.x.device,
    )
    for mask, data, name in (
        (batch.x_observed_mask, batch.x, "x_observed_mask"),
        (batch.y_observed_mask, batch.y, "y_observed_mask"),
        (batch.observed_covariates_mask, batch.observed_covariates, "observed_covariates_mask"),
        (
            batch.known_future_covariates_mask,
            batch.known_future_covariates,
            "known_future_covariates_mask",
        ),
    ):
        _validate_mask(mask, data, name, batch.x.device)


def _validate_regime(batch: WindowBatch, batch_size: int) -> None:
    if batch.regime is not None:
        _require_shape(batch.regime, (batch_size,), "regime")
        if batch.regime.dtype is torch.bool or batch.regime.device != batch.x.device:
            raise ValueError("regime must be numeric and on the x device")
        if batch.regime.is_floating_point() and not bool(torch.isfinite(batch.regime).all()):
            raise ValueError("regime must contain only finite values")


def validate_window_batch(batch: WindowBatch) -> WindowBatch:
    dimensions = _window_dimensions(batch)
    batch_size, history, input_size, horizon, target_size, observed_size, future_size = dimensions
    _validate_window_names(batch, batch_size, input_size, target_size, observed_size, future_size)
    _validate_window_tensors(
        batch, batch_size, history, horizon, target_size, observed_size, future_size
    )
    _validate_regime(batch, batch_size)
    _validate_times(batch, batch_size, horizon)
    _validate_window_metadata(batch.metadata)
    return batch


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
