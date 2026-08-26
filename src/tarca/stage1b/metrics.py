from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    replicates: int
    unit_count: int


@dataclass(frozen=True, slots=True)
class MetricBundle:
    crps: float
    nll: float
    mae: float


def _validate_gaussian_inputs(mean: Tensor, scale: Tensor, target: Tensor) -> None:
    if mean.shape != scale.shape or mean.shape != target.shape:
        raise ValueError("Gaussian mean, scale, and target shapes must match")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (mean, scale, target)):
        raise ValueError("Gaussian metric inputs must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("Gaussian scale must be strictly positive")


def gaussian_crps(mean: Tensor, scale: Tensor, target: Tensor) -> Tensor:
    _validate_gaussian_inputs(mean, scale, target)
    standardized = (target - mean) / scale
    density = torch.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
    distribution = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    return scale * (
        standardized * (2.0 * distribution - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi)
    )


def gaussian_nll(mean: Tensor, scale: Tensor, target: Tensor) -> Tensor:
    _validate_gaussian_inputs(mean, scale, target)
    return torch.log(scale) + 0.5 * ((target - mean) / scale) ** 2 + 0.5 * math.log(2.0 * math.pi)


def summarize_gaussian(mean: Tensor, scale: Tensor, target: Tensor) -> MetricBundle:
    return MetricBundle(
        crps=float(gaussian_crps(mean, scale, target).mean()),
        nll=float(gaussian_nll(mean, scale, target).mean()),
        mae=float(torch.abs(target - mean).mean()),
    )


def paired_bootstrap_interval(
    improvements: Tensor,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> BootstrapInterval:
    if improvements.ndim != 1 or improvements.numel() < 2:
        raise ValueError("paired bootstrap requires at least two whole-trajectory units")
    if not bool(torch.isfinite(improvements).all()):
        raise ValueError("paired bootstrap improvements must be finite")
    if replicates < 1000:
        raise ValueError("paired bootstrap requires at least 1000 replicates")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between 0.5 and 1")
    values = improvements.to(torch.float64).cpu()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=values.numel(),
        size=(replicates, values.numel()),
        generator=generator,
    )
    bootstrap_means = values[indices].mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = torch.quantile(
        bootstrap_means,
        torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64),
    )
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        replicates=replicates,
        unit_count=values.numel(),
    )
