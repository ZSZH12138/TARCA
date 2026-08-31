from __future__ import annotations

import math
from types import MappingProxyType

import pytest
import torch

from tarca.contracts import ForecastDistribution
from tarca.e02.scoring import (
    TrajectoryLineage,
    score_trajectory,
    summarize_scores,
)


def _forecast(mean: torch.Tensor, scale: torch.Tensor) -> ForecastDistribution:
    return ForecastDistribution(
        mean=mean,
        scale=scale,
        quantiles=MappingProxyType({}),
        logits=None,
        samples=None,
        window_id=tuple(f"window-{index}" for index in range(mean.shape[0])),
        target_names=("x0",),
    )


def test_score_trajectory_matches_analytical_standard_gaussian_at_mean() -> None:
    target = torch.zeros((2, 6, 1), dtype=torch.float64)
    score = score_trajectory(
        _forecast(target.clone(), torch.ones_like(target)),
        target,
        TrajectoryLineage("trajectory-1", 1729, "SEEN", 2),
    )

    expected_crps = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert score.crps == pytest.approx(expected_crps)
    assert score.nll == pytest.approx(0.5 * math.log(2.0 * math.pi))
    assert score.mae == 0.0
    assert score.origin_count == 2
    assert score.horizon_count == 6


def test_coverage_error_uses_four_nominal_levels() -> None:
    target = torch.zeros((2, 6, 1), dtype=torch.float64)
    neural = score_trajectory(
        _forecast(target.clone(), torch.ones_like(target)),
        target,
        TrajectoryLineage("trajectory-1", 1729, "SEEN", 2),
    )
    baseline = score_trajectory(
        _forecast(torch.full_like(target, 0.5), torch.ones_like(target)),
        target,
        TrajectoryLineage("trajectory-1", 1729, "SEEN", 2),
    )

    summary = summarize_scores((neural,), (baseline,))

    assert summary.coverage_levels == (0.50, 0.80, 0.90, 0.95)
    assert summary.observed_coverage == (1.0, 1.0, 1.0, 1.0)
    assert summary.coverage_error == pytest.approx(0.2125)


def test_score_trajectory_rejects_window_level_lineage_count() -> None:
    target = torch.zeros((2, 6, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="origin count"):
        score_trajectory(
            _forecast(target.clone(), torch.ones_like(target)),
            target,
            TrajectoryLineage("trajectory-1", 1729, "SEEN", 3),
        )
