from __future__ import annotations

from dataclasses import dataclass

import torch


class WorldHealthError(RuntimeError):
    """Raised when a trajectory is numerically unusable before model training."""


@dataclass(frozen=True, slots=True)
class WorldHealthReport:
    passed: bool
    minimum_temporal_std: float
    median_temporal_std: float
    mean_absolute_step: float
    mean_absolute_lag2_step: float
    linear_residual_ratio: float


def assess_world_health(values: torch.Tensor) -> WorldHealthReport:
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 2:
        raise WorldHealthError("world health requires at least four multivariate observations")
    if not bool(torch.isfinite(values).all()):
        raise WorldHealthError("trajectory contains non-finite values")
    work = values.to(torch.float64)
    temporal_std = work.std(dim=0, unbiased=False)
    scale = max(1.0, float(torch.sqrt(torch.mean(work.square())).item()))
    collapse_floor = 1e-8 * scale
    if float(temporal_std.max().item()) <= collapse_floor:
        raise WorldHealthError("trajectory collapsed to a fixed point")
    lag1 = float(torch.mean(torch.abs(work[1:] - work[:-1])).item())
    lag2 = float(torch.mean(torch.abs(work[2:] - work[:-2])).item())
    if lag1 > collapse_floor and lag2 <= max(collapse_floor, lag1 * 1e-6):
        raise WorldHealthError("trajectory collapsed to a period-2 orbit")
    predictors = torch.cat((work[:-1], torch.ones((work.shape[0] - 1, 1), dtype=work.dtype)), dim=1)
    targets = work[1:]
    coefficients = torch.linalg.lstsq(predictors, targets).solution
    residual = targets - predictors @ coefficients
    denominator = float(torch.mean((targets - targets.mean(dim=0)).square()).item())
    residual_ratio = (
        0.0 if denominator == 0.0 else float(torch.mean(residual.square()).item()) / denominator
    )
    return WorldHealthReport(
        passed=True,
        minimum_temporal_std=float(temporal_std.min().item()),
        median_temporal_std=float(temporal_std.median().item()),
        mean_absolute_step=lag1,
        mean_absolute_lag2_step=lag2,
        linear_residual_ratio=residual_ratio,
    )
