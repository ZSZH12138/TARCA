"""Validated immutable batch data contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

import torch
from torch import Tensor

from .types import JSONMetadata
from .validation import validate_json_metadata


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """A validated batch of historical windows and aligned forecast labels."""

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
    feature_start: tuple[datetime, ...]
    feature_end: tuple[datetime, ...]
    prediction_start: tuple[datetime, ...]
    label_end: tuple[datetime, ...]
    forecast_time: tuple[tuple[datetime, ...], ...]
    metadata: JSONMetadata

    def __post_init__(self) -> None:
        batch_size, history, feature_count = _validate_primary_tensor(self.x, "x")
        horizon_candidates: list[tuple[str, int]] = []

        if self.y is not None:
            target_horizon, target_count = _validate_aligned_tensor(
                self.y, "y", batch_size, "horizon"
            )
            horizon_candidates.append(("y", target_horizon))
            _validate_names(self.target_names, "target_names", target_count)
        else:
            _validate_absent_names(self.target_names, "target_names")

        if self.observed_covariates is not None:
            observed_length, observed_count = _validate_aligned_tensor(
                self.observed_covariates, "observed_covariates", batch_size, "history"
            )
            if observed_length != history:
                raise ValueError("observed_covariates: expected history dimension to match x")
            _validate_names(
                self.observed_covariate_names,
                "observed_covariate_names",
                observed_count,
            )
        else:
            _validate_absent_names(self.observed_covariate_names, "observed_covariate_names")

        if self.known_future_covariates is not None:
            future_horizon, future_count = _validate_aligned_tensor(
                self.known_future_covariates, "known_future_covariates", batch_size, "horizon"
            )
            horizon_candidates.append(("known_future_covariates", future_horizon))
            _validate_names(
                self.known_future_covariate_names,
                "known_future_covariate_names",
                future_count,
            )
        else:
            _validate_absent_names(
                self.known_future_covariate_names, "known_future_covariate_names"
            )

        _validate_names(self.input_feature_names, "input_feature_names", feature_count)
        if set(self.target_names).intersection(self.known_future_covariate_names):
            raise ValueError("target_names and known_future_covariate_names must be disjoint")
        _validate_window_ids(self.window_id, batch_size)

        _validate_mask(self.x_observed_mask, self.x, "x_observed_mask")
        _validate_mask(self.y_observed_mask, self.y, "y_observed_mask")
        _validate_mask(
            self.observed_covariates_mask,
            self.observed_covariates,
            "observed_covariates_mask",
        )
        _validate_mask(
            self.known_future_covariates_mask,
            self.known_future_covariates,
            "known_future_covariates_mask",
        )
        _validate_regime(self.regime, batch_size, self.x.device)

        boundary_times = _normalize_boundary_times(self, batch_size)
        forecast_time = _normalize_forecast_time(self.forecast_time, batch_size)
        forecast_horizons = {len(times) for times in forecast_time}
        if len(forecast_horizons) != 1:
            raise ValueError("forecast_time: every sample must use the same horizon")
        forecast_horizon = forecast_horizons.pop()
        horizon_candidates.append(("forecast_time", forecast_horizon))
        horizon = _resolve_horizon(horizon_candidates)

        _validate_temporal_order(boundary_times, forecast_time, horizon)
        object.__setattr__(self, "window_id", _normalize_string_tuple(self.window_id, "window_id"))
        object.__setattr__(
            self,
            "input_feature_names",
            _normalize_string_tuple(self.input_feature_names, "input_feature_names"),
        )
        object.__setattr__(
            self, "target_names", _normalize_string_tuple(self.target_names, "target_names")
        )
        object.__setattr__(
            self,
            "observed_covariate_names",
            _normalize_string_tuple(self.observed_covariate_names, "observed_covariate_names"),
        )
        object.__setattr__(
            self,
            "known_future_covariate_names",
            _normalize_string_tuple(
                self.known_future_covariate_names, "known_future_covariate_names"
            ),
        )
        object.__setattr__(self, "feature_start", boundary_times["feature_start"])
        object.__setattr__(self, "feature_end", boundary_times["feature_end"])
        object.__setattr__(self, "prediction_start", boundary_times["prediction_start"])
        object.__setattr__(self, "label_end", boundary_times["label_end"])
        object.__setattr__(self, "forecast_time", forecast_time)
        object.__setattr__(self, "metadata", validate_json_metadata(self.metadata))


def _validate_primary_tensor(tensor: Tensor, field_name: str) -> tuple[int, int, int]:
    _validate_float_tensor(tensor, field_name)
    if tensor.ndim != 3:
        raise ValueError(f"{field_name}: expected rank 3 [B, L, D]")
    if any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(f"{field_name}: dimensions must all be positive")
    return tuple(tensor.shape)  # type: ignore[return-value]


def _validate_aligned_tensor(
    tensor: Tensor, field_name: str, batch_size: int, dimension_name: str
) -> tuple[int, int]:
    _validate_float_tensor(tensor, field_name)
    if tensor.ndim != 3:
        raise ValueError(f"{field_name}: expected rank 3")
    if any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(f"{field_name}: dimensions must all be positive")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{field_name}: batch dimension must match x")
    return tensor.shape[1], tensor.shape[2]


def _validate_float_tensor(tensor: Tensor, field_name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{field_name}: expected a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{field_name}: expected a floating tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"{field_name}: expected a materialized non-meta tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{field_name}: values must be finite")


def _normalize_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    normalized = tuple(value)
    if not all(isinstance(item, str) for item in normalized):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    return normalized


def _validate_names(value: object, field_name: str, expected_size: int) -> None:
    names = _normalize_string_tuple(value, field_name)
    if len(names) != expected_size:
        raise ValueError(f"{field_name}: expected {expected_size} names")
    if any(not name.strip() for name in names):
        raise ValueError(f"{field_name}: names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name}: names must be unique")


def _validate_absent_names(value: object, field_name: str) -> None:
    names = _normalize_string_tuple(value, field_name)
    if names:
        raise ValueError(f"{field_name}: names require the corresponding tensor")


def _validate_window_ids(value: object, batch_size: int) -> None:
    window_ids = _normalize_string_tuple(value, "window_id")
    if len(window_ids) != batch_size:
        raise ValueError("window_id: expected one ID per batch element")
    if any(not window.strip() for window in window_ids):
        raise ValueError("window_id: IDs must be non-empty")
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("window_id: IDs must be unique")


def _validate_mask(mask: Tensor | None, tensor: Tensor | None, field_name: str) -> None:
    if mask is None:
        return
    if tensor is None:
        raise ValueError(f"{field_name}: a mask requires its tensor")
    if not isinstance(mask, Tensor):
        raise ValueError(f"{field_name}: expected a torch.Tensor")
    if mask.dtype != torch.bool:
        raise ValueError(f"{field_name}: expected bool dtype")
    if mask.shape != tensor.shape:
        raise ValueError(f"{field_name}: shape must exactly match its tensor")
    if mask.device != tensor.device:
        raise ValueError(f"{field_name}: device must match its tensor")


def _validate_regime(regime: Tensor | None, batch_size: int, device: torch.device) -> None:
    if regime is None:
        return
    if not isinstance(regime, Tensor):
        raise ValueError("regime: expected a torch.Tensor")
    if regime.ndim != 1 or regime.shape[0] != batch_size:
        raise ValueError("regime: expected shape [B]")
    if regime.dtype == torch.bool or not _is_integer_dtype(regime.dtype):
        raise ValueError("regime: expected an integer dtype")
    if regime.device.type == "meta":
        raise ValueError("regime: expected a materialized non-meta tensor")
    if regime.device != device:
        raise ValueError("regime: device must match x")
    if not bool(torch.isfinite(regime).all()):
        raise ValueError("regime: values must be finite")


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _normalize_boundary_times(
    batch: WindowBatch, batch_size: int
) -> dict[str, tuple[datetime, ...]]:
    fields = ("feature_start", "feature_end", "prediction_start", "label_end")
    normalized: dict[str, tuple[datetime, ...]] = {}
    for field_name in fields:
        value = getattr(batch, field_name)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{field_name}: expected a datetime sequence")
        times = tuple(value)
        if len(times) != batch_size:
            raise ValueError(f"{field_name}: expected one datetime per batch element")
        for index, time in enumerate(times):
            _validate_utc_datetime(time, f"{field_name}[{index}]")
        normalized[field_name] = times
    return normalized


def _normalize_forecast_time(value: object, batch_size: int) -> tuple[tuple[datetime, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("forecast_time: expected a sequence of datetime sequences")
    samples = tuple(value)
    if len(samples) != batch_size:
        raise ValueError("forecast_time: expected one sequence per batch element")
    normalized: list[tuple[datetime, ...]] = []
    for sample_index, sample_times in enumerate(samples):
        if isinstance(sample_times, (str, bytes)) or not isinstance(sample_times, Sequence):
            raise ValueError(f"forecast_time[{sample_index}]: expected a datetime sequence")
        times = tuple(sample_times)
        for horizon_index, time in enumerate(times):
            _validate_utc_datetime(time, f"forecast_time[{sample_index}][{horizon_index}]")
        normalized.append(times)
    return tuple(normalized)


def _validate_utc_datetime(value: object, field_path: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_path}: expected a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_path}: datetime must be timezone-aware UTC")


def _resolve_horizon(candidates: list[tuple[str, int]]) -> int:
    horizons = {horizon for _, horizon in candidates}
    if len(horizons) != 1:
        details = ", ".join(f"{name}={horizon}" for name, horizon in candidates)
        raise ValueError(f"horizon: contradictory values ({details})")
    horizon = horizons.pop()
    if horizon <= 0:
        raise ValueError("horizon: must be positive")
    return horizon


def _validate_temporal_order(
    boundary_times: dict[str, tuple[datetime, ...]],
    forecast_time: tuple[tuple[datetime, ...], ...],
    horizon: int,
) -> None:
    for sample_index, times in enumerate(forecast_time):
        if len(times) != horizon:
            raise ValueError(f"forecast_time[{sample_index}]: expected horizon {horizon}")
        feature_start = boundary_times["feature_start"][sample_index]
        feature_end = boundary_times["feature_end"][sample_index]
        prediction_start = boundary_times["prediction_start"][sample_index]
        label_end = boundary_times["label_end"][sample_index]
        if not feature_start <= feature_end < prediction_start <= label_end:
            raise ValueError(f"boundary order: invalid values for sample {sample_index}")
        if any(current >= following for current, following in pairwise(times)):
            raise ValueError(f"forecast_time[{sample_index}]: values must be strictly increasing")
        if any(time < prediction_start or time > label_end for time in times):
            raise ValueError(
                f"forecast_time[{sample_index}]: values must be within prediction "
                "and label boundaries"
            )
