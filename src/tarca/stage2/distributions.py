from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import Tensor

from tarca.contracts import ForecastDistribution, validate_forecast_distribution

DEFAULT_GAUSSIAN_QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.975)


def residual_scale(residuals: Tensor, *, floor: float, ceiling: Tensor) -> Tensor:
    if residuals.ndim != 3 or not residuals.is_floating_point():
        raise ValueError("residuals must be a floating rank-three tensor")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals must be finite")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("residual scale floor must be finite and positive")
    expected = tuple(residuals.shape[1:])
    if ceiling.shape != expected or not ceiling.is_floating_point():
        raise ValueError(f"residual scale ceiling shape must be {expected}")
    resolved_ceiling = ceiling.to(dtype=torch.float64, device=residuals.device)
    if not bool(torch.isfinite(resolved_ceiling).all()) or bool(
        (resolved_ceiling < floor).any()
    ):
        raise ValueError("residual scale ceiling must be finite and at least the floor")
    rms = torch.sqrt(torch.mean(residuals.to(torch.float64).square(), dim=0))
    floor_tensor = torch.full_like(rms, floor)
    return torch.minimum(torch.maximum(rms, floor_tensor), resolved_ceiling)


def gaussian_quantiles(
    mean: Tensor,
    scale: Tensor,
    levels: tuple[float, ...],
) -> Mapping[float, Tensor]:
    if mean.ndim != 3 or scale.shape != mean.shape:
        raise ValueError("Gaussian mean and scale must be aligned rank-three tensors")
    if mean.dtype != scale.dtype or mean.device != scale.device:
        raise ValueError("Gaussian mean and scale must share dtype and device")
    if not mean.is_floating_point() or not all(
        bool(torch.isfinite(tensor).all()) for tensor in (mean, scale)
    ):
        raise ValueError("Gaussian mean and scale must be finite floating tensors")
    if bool((scale <= 0).any()):
        raise ValueError("Gaussian scale must be strictly positive")
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError("Gaussian quantile levels must be unique and increasing")
    if any(isinstance(level, bool) or not 0.0 < level < 1.0 for level in levels):
        raise ValueError("Gaussian quantile levels must be between zero and one")
    return MappingProxyType(
        {
            level: mean
            + scale
            * (
                math.sqrt(2.0)
                * torch.erfinv(
                    2.0 * torch.tensor(level, dtype=mean.dtype, device=mean.device) - 1.0
                )
            )
            for level in levels
        }
    )


def gaussian_forecast(
    mean: Tensor,
    scale: Tensor,
    *,
    window_id: tuple[str, ...] | None,
    target_names: tuple[str, ...],
    levels: tuple[float, ...] = DEFAULT_GAUSSIAN_QUANTILES,
) -> ForecastDistribution:
    return validate_forecast_distribution(
        ForecastDistribution(
            mean=mean,
            scale=scale,
            quantiles=gaussian_quantiles(mean, scale, levels),
            logits=None,
            samples=None,
            window_id=window_id,
            target_names=target_names,
        )
    )
