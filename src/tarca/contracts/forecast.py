"""Validated immutable forecast distribution contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from types import MappingProxyType

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    """Aligned point, probabilistic, and sampled forecasts."""

    mean: Tensor
    scale: Tensor | None
    quantiles: Mapping[float, Tensor]
    logits: Tensor | None
    samples: Tensor | None
    window_id: tuple[str, ...] | None
    target_names: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size, _, target_count = _validate_mean(self.mean)
        if self.scale is not None:
            _validate_mean_aligned(self.scale, "scale", self.mean)
            if not bool(torch.all(self.scale > 0)):
                raise ValueError("scale: values must be strictly positive")

        quantiles = _validate_quantiles(self.quantiles, self.mean)
        if self.logits is not None:
            _validate_logits(self.logits, self.mean)
        if self.samples is not None:
            _validate_samples(self.samples, self.mean)

        window_id = (
            None
            if self.window_id is None
            else _validate_string_tuple(self.window_id, "window_id", batch_size)
        )
        target_names = _validate_string_tuple(self.target_names, "target_names", target_count)

        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "target_names", target_names)


def _validate_mean(mean: object) -> tuple[int, int, int]:
    tensor = _validate_floating_tensor(mean, "mean")
    if tensor.ndim != 3:
        raise ValueError("mean: expected rank 3 [B, H, Dy]")
    if any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError("mean: dimensions must all be positive")
    return tuple(tensor.shape)  # type: ignore[return-value]


def _validate_floating_tensor(value: object, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{field_name}: expected a torch.Tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{field_name}: expected a floating tensor")
    if value.device.type == "meta":
        raise ValueError(f"{field_name}: values must be materialized and finite")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field_name}: values must be finite")
    return value


def _validate_mean_aligned(tensor: object, field_name: str, mean: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{field_name}: expected a torch.Tensor")
    if tensor.shape != mean.shape:
        raise ValueError(f"{field_name}: shape must exactly match mean")
    _validate_device_and_dtype(tensor, field_name, mean)
    return _validate_floating_tensor(tensor, field_name)


def _validate_device_and_dtype(tensor: Tensor, field_name: str, mean: Tensor) -> None:
    if tensor.device != mean.device:
        raise ValueError(f"{field_name}: device must match mean")
    if tensor.dtype != mean.dtype:
        raise ValueError(f"{field_name}: dtype must match mean")


def _validate_quantiles(quantiles: object, mean: Tensor) -> Mapping[float, Tensor]:
    if not isinstance(quantiles, Mapping):
        raise ValueError("quantiles: expected a mapping")

    normalized: dict[float, Tensor] = {}
    for level, value in quantiles.items():
        numeric_level = _validate_quantile_level(level)
        if numeric_level in normalized:
            raise ValueError("quantiles: levels must remain unique when normalized to float")
        tensor = _validate_mean_aligned(value, f"quantiles[{level!r}]", mean)
        normalized[numeric_level] = tensor

    ordered = sorted(normalized.items())
    for (lower_level, lower), (upper_level, upper) in pairwise(ordered):
        if not bool(torch.all(lower <= upper)):
            raise ValueError(
                "quantiles: predictions must be elementwise nondecreasing "
                f"between levels {lower_level} and {upper_level}"
            )
    return MappingProxyType(normalized)


def _validate_quantile_level(level: object) -> float:
    if isinstance(level, bool) or not isinstance(level, Real):
        raise ValueError("quantiles: levels must be real numbers, excluding bool")
    try:
        numeric_level = float(level)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("quantiles: levels must be finite") from error
    if not math.isfinite(numeric_level):
        raise ValueError("quantiles: levels must be finite")
    if not 0.0 < numeric_level < 1.0:
        raise ValueError("quantiles: levels must be strictly inside (0, 1)")
    return numeric_level


def _validate_logits(logits: object, mean: Tensor) -> None:
    if not isinstance(logits, Tensor):
        raise ValueError("logits: expected a torch.Tensor")
    if logits.ndim != 4:
        raise ValueError("logits: expected rank 4 [B, H, Dy, C]")
    if logits.shape[:3] != mean.shape:
        raise ValueError("logits: first three dimensions must match mean")
    if logits.shape[3] <= 1:
        raise ValueError("logits: class dimension C must be greater than 1")
    _validate_device_and_dtype(logits, "logits", mean)
    _validate_floating_tensor(logits, "logits")


def _validate_samples(samples: object, mean: Tensor) -> None:
    if not isinstance(samples, Tensor):
        raise ValueError("samples: expected a torch.Tensor")
    if samples.ndim != 4:
        raise ValueError("samples: expected rank 4 [S, B, H, Dy]")
    if samples.shape[0] <= 0:
        raise ValueError("samples: sample dimension S must be positive")
    if samples.shape[1:] != mean.shape:
        raise ValueError("samples: final three dimensions must match mean")
    _validate_device_and_dtype(samples, "samples", mean)
    _validate_floating_tensor(samples, "samples")


def _validate_string_tuple(value: object, field_name: str, expected_size: int) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    normalized = tuple(value)
    if not all(isinstance(item, str) for item in normalized):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    if len(normalized) != expected_size:
        raise ValueError(f"{field_name}: expected {expected_size} entries")
    if any(not item.strip() for item in normalized):
        raise ValueError(f"{field_name}: entries must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name}: entries must be unique")
    return normalized
