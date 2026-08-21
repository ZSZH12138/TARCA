from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ForecastDistribution:
    mean: Tensor
    scale: Tensor | None
    quantiles: Mapping[float, Tensor]
    logits: Tensor | None
    samples: Tensor | None
    window_id: tuple[str, ...] | None
    target_names: tuple[str, ...]


def _validate_aligned_tensor(
    tensor: Tensor,
    expected_shape: tuple[int, ...],
    name: str,
    mean: Tensor,
) -> None:
    if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}")
    if not tensor.is_floating_point() or tensor.dtype != mean.dtype:
        raise ValueError(f"{name} dtype must match mean floating dtype")
    if tensor.device != mean.device:
        raise ValueError(f"{name} device must match mean device")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_quantiles(distribution: ForecastDistribution) -> None:
    unsorted_levels = tuple(distribution.quantiles)
    if any(
        isinstance(level, bool) or not isinstance(level, (int, float)) for level in unsorted_levels
    ):
        raise ValueError("quantile levels must be numeric")
    levels = sorted(unsorted_levels)
    if any(not 0.0 < float(level) < 1.0 for level in levels):
        raise ValueError("quantile levels must be between 0 and 1")
    for level in levels:
        _validate_aligned_tensor(
            distribution.quantiles[level],
            tuple(distribution.mean.shape),
            f"quantile {level}",
            distribution.mean,
        )
    for lower, upper in pairwise(levels):
        if bool((distribution.quantiles[lower] > distribution.quantiles[upper]).any()):
            raise ValueError("quantile predictions must not cross")


def _validate_forecast_identity(distribution: ForecastDistribution, batch_size: int) -> None:
    if distribution.window_id is not None:
        if len(distribution.window_id) != batch_size:
            raise ValueError("window_id count must match forecast batch")
        if any(not value.strip() for value in distribution.window_id):
            raise ValueError("window_id must not contain blank values")
        if len(set(distribution.window_id)) != len(distribution.window_id):
            raise ValueError("window_id must be unique")
    if len(distribution.target_names) != distribution.mean.shape[2]:
        raise ValueError("target_names count must match forecast targets")
    if any(not value.strip() for value in distribution.target_names):
        raise ValueError("target_names must not contain blank values")
    if len(set(distribution.target_names)) != len(distribution.target_names):
        raise ValueError("target_names must be unique")


def validate_forecast_distribution(
    distribution: ForecastDistribution,
) -> ForecastDistribution:
    mean = distribution.mean
    if not isinstance(mean, Tensor) or mean.ndim != 3 or any(size <= 0 for size in mean.shape):
        raise ValueError("mean must be a nonempty rank-3 Tensor")
    _validate_aligned_tensor(mean, tuple(mean.shape), "mean", mean)
    if distribution.scale is not None:
        _validate_aligned_tensor(distribution.scale, tuple(mean.shape), "scale", mean)
        if bool((distribution.scale <= 0).any()):
            raise ValueError("scale must be strictly positive")
    _validate_quantiles(distribution)
    if distribution.logits is not None:
        if distribution.logits.ndim != 4 or distribution.logits.shape[-1] <= 0:
            raise ValueError("logits must be a nonempty rank-4 Tensor")
        shape = (*mean.shape, distribution.logits.shape[-1])
        _validate_aligned_tensor(distribution.logits, shape, "logits", mean)
    if distribution.samples is not None:
        if distribution.samples.ndim != 4 or distribution.samples.shape[0] <= 0:
            raise ValueError("samples must be a nonempty rank-4 Tensor")
        shape = (distribution.samples.shape[0], *mean.shape)
        _validate_aligned_tensor(distribution.samples, shape, "samples", mean)
    _validate_forecast_identity(distribution, mean.shape[0])
    return distribution
