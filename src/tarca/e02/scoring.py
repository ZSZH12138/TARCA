from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from tarca.contracts import ForecastDistribution, validate_forecast_distribution
from tarca.stage1b.metrics import gaussian_crps, gaussian_nll

Regime = Literal["SEEN", "UNSEEN"]
COVERAGE_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class TrajectoryLineage:
    trajectory_id: str
    formal_seed: int
    regime: Regime
    origin_count: int

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or self.formal_seed <= 0 or self.origin_count <= 0:
            raise ValueError("trajectory lineage identity and counts must be positive")
        if self.regime not in ("SEEN", "UNSEEN"):
            raise ValueError("trajectory lineage regime must be SEEN or UNSEEN")


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    trajectory_id: str
    formal_seed: int
    regime: Regime
    origin_count: int
    horizon_count: int
    variable_count: int
    crps: float
    nll: float
    mae: float
    coverage: tuple[tuple[float, float], ...]
    horizon_crps: tuple[float, ...]
    horizon_nll: tuple[float, ...]
    horizon_mae: tuple[float, ...]
    horizon_coverage: tuple[tuple[float, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or self.formal_seed <= 0:
            raise ValueError("trajectory score identity must be valid")
        if self.regime not in ("SEEN", "UNSEEN"):
            raise ValueError("trajectory score regime must be SEEN or UNSEEN")
        if min(self.origin_count, self.horizon_count, self.variable_count) <= 0:
            raise ValueError("trajectory score dimensions must be positive")
        metrics = (
            self.crps,
            self.nll,
            self.mae,
            *(value for _, value in self.coverage),
            *self.horizon_crps,
            *self.horizon_nll,
            *self.horizon_mae,
            *(value for _, values in self.horizon_coverage for value in values),
        )
        if any(not math.isfinite(value) for value in metrics):
            raise ValueError("trajectory score metrics must be finite")
        if not all(
            len(values) == self.horizon_count
            for values in (self.horizon_crps, self.horizon_nll, self.horizon_mae)
        ):
            raise ValueError("trajectory score horizon metrics must align")
        if tuple(level for level, _ in self.coverage) != COVERAGE_LEVELS:
            raise ValueError("trajectory score coverage levels must match E02")
        if tuple(level for level, _ in self.horizon_coverage) != COVERAGE_LEVELS:
            raise ValueError("trajectory score horizon coverage levels must match E02")
        if any(not 0.0 <= value <= 1.0 for _, value in self.coverage):
            raise ValueError("trajectory coverage must be between zero and one")


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    trajectory_count: int
    crps: float
    nll: float
    mae: float
    baseline_crps: float
    baseline_nll: float
    baseline_mae: float
    crps_skill: float
    relative_nll: float
    relative_mae: float
    coverage_levels: tuple[float, ...]
    observed_coverage: tuple[float, ...]
    coverage_error: float
    regime_crps_skill: tuple[tuple[str, float], ...]
    regime_coverage_error: tuple[tuple[str, float], ...]
    secondary_horizon_skill: tuple[tuple[str, float], ...]
    data_seed_primary_skill: tuple[tuple[int, float], ...]


def _coverage_by_horizon(
    mean: Tensor,
    scale: Tensor,
    target: Tensor,
    level: float,
) -> Tensor:
    tail = (1.0 + level) / 2.0
    quantile = math.sqrt(2.0) * torch.erfinv(
        torch.tensor(2.0 * tail - 1.0, dtype=mean.dtype, device=mean.device)
    )
    inside = (target >= mean - quantile * scale) & (target <= mean + quantile * scale)
    return inside.to(torch.float64).mean(dim=(0, 2))


def score_trajectory(
    prediction: ForecastDistribution,
    target: Tensor,
    lineage: TrajectoryLineage,
) -> TrajectoryScore:
    validated = validate_forecast_distribution(prediction)
    if validated.scale is None:
        raise ValueError("E02 trajectory scoring requires a Gaussian scale")
    if target.shape != validated.mean.shape or target.dtype != validated.mean.dtype:
        raise ValueError("E02 trajectory target must align with the forecast")
    if target.device != validated.mean.device or not bool(torch.isfinite(target).all()):
        raise ValueError("E02 trajectory target must be finite and colocated")
    if target.shape[0] != lineage.origin_count:
        raise ValueError("trajectory lineage origin count does not match forecast windows")
    crps_by_horizon = gaussian_crps(validated.mean, validated.scale, target).mean(
        dim=(0, 2)
    )
    nll_by_horizon = gaussian_nll(validated.mean, validated.scale, target).mean(dim=(0, 2))
    mae_by_horizon = torch.abs(validated.mean - target).mean(dim=(0, 2))
    horizon_coverages = tuple(
        (level, _coverage_by_horizon(validated.mean, validated.scale, target, level))
        for level in COVERAGE_LEVELS
    )
    return TrajectoryScore(
        trajectory_id=lineage.trajectory_id,
        formal_seed=lineage.formal_seed,
        regime=lineage.regime,
        origin_count=lineage.origin_count,
        horizon_count=target.shape[1],
        variable_count=target.shape[2],
        crps=float(crps_by_horizon.mean()),
        nll=float(nll_by_horizon.mean()),
        mae=float(mae_by_horizon.mean()),
        coverage=tuple((level, float(values.mean())) for level, values in horizon_coverages),
        horizon_crps=tuple(float(value) for value in crps_by_horizon),
        horizon_nll=tuple(float(value) for value in nll_by_horizon),
        horizon_mae=tuple(float(value) for value in mae_by_horizon),
        horizon_coverage=tuple(
            (level, tuple(float(value) for value in values))
            for level, values in horizon_coverages
        ),
    )


def _paired_scores(
    scores: tuple[TrajectoryScore, ...],
    baseline_scores: tuple[TrajectoryScore, ...],
) -> tuple[tuple[TrajectoryScore, TrajectoryScore], ...]:
    if not scores or len(scores) != len(baseline_scores):
        raise ValueError("score summaries require aligned nonempty trajectory sets")
    score_by_id = {score.trajectory_id: score for score in scores}
    baseline_by_id = {score.trajectory_id: score for score in baseline_scores}
    if len(score_by_id) != len(scores) or len(baseline_by_id) != len(baseline_scores):
        raise ValueError("trajectory score IDs must be unique")
    if set(score_by_id) != set(baseline_by_id):
        raise ValueError("neural and baseline trajectory IDs must match")
    pairs = tuple(
        (score_by_id[identifier], baseline_by_id[identifier])
        for identifier in sorted(score_by_id)
    )
    if any(
        (score.formal_seed, score.regime, score.horizon_count)
        != (baseline.formal_seed, baseline.regime, baseline.horizon_count)
        for score, baseline in pairs
    ):
        raise ValueError("paired trajectory score lineage must match")
    return pairs


def _mean_horizons(score: TrajectoryScore, start: int, end: int) -> float:
    return statistics.fmean(score.horizon_crps[start:end])


def _skill(
    pairs: tuple[tuple[TrajectoryScore, TrajectoryScore], ...],
    start: int,
    end: int,
) -> float:
    neural = statistics.fmean(_mean_horizons(score, start, end) for score, _ in pairs)
    baseline = statistics.fmean(
        _mean_horizons(score, start, end) for _, score in pairs
    )
    if baseline <= 0:
        raise ValueError("baseline CRPS must be positive")
    return 1.0 - neural / baseline


def summarize_scores(
    scores: tuple[TrajectoryScore, ...],
    baseline_scores: tuple[TrajectoryScore, ...],
) -> ScoreSummary:
    pairs = _paired_scores(scores, baseline_scores)
    horizon_count = pairs[0][0].horizon_count
    if any(score.horizon_count != horizon_count for pair in pairs for score in pair):
        raise ValueError("trajectory scores must share a horizon count")
    observed = tuple(
        statistics.fmean(dict(score.coverage)[level] for score, _ in pairs)
        for level in COVERAGE_LEVELS
    )
    coverage_error = statistics.fmean(
        abs(value - level) for value, level in zip(observed, COVERAGE_LEVELS, strict=True)
    )
    primary_end = min(6, horizon_count)
    regime_skill = tuple(
        (
            regime,
            _skill(tuple(pair for pair in pairs if pair[0].regime == regime), 0, primary_end),
        )
        for regime in ("SEEN", "UNSEEN")
        if any(pair[0].regime == regime for pair in pairs)
    )
    regime_coverage = tuple(
        (
            regime,
            statistics.fmean(
                abs(
                    statistics.fmean(
                        dict(score.coverage)[level]
                        for score, _ in pairs
                        if score.regime == regime
                    )
                    - level
                )
                for level in COVERAGE_LEVELS
            ),
        )
        for regime in ("SEEN", "UNSEEN")
        if any(pair[0].regime == regime for pair in pairs)
    )
    secondary = tuple(
        (label, _skill(pairs, start, min(end, horizon_count)))
        for label, start, end in (("h7_12", 6, 12), ("h13_24", 12, 24))
        if horizon_count > start
    )
    seed_skill = tuple(
        (
            seed,
            _skill(
                tuple(pair for pair in pairs if pair[0].formal_seed == seed),
                0,
                primary_end,
            ),
        )
        for seed in sorted({pair[0].formal_seed for pair in pairs})
    )
    crps = statistics.fmean(score.crps for score, _ in pairs)
    nll = statistics.fmean(score.nll for score, _ in pairs)
    mae = statistics.fmean(score.mae for score, _ in pairs)
    baseline_crps = statistics.fmean(score.crps for _, score in pairs)
    baseline_nll = statistics.fmean(score.nll for _, score in pairs)
    baseline_mae = statistics.fmean(score.mae for _, score in pairs)
    return ScoreSummary(
        trajectory_count=len(pairs),
        crps=crps,
        nll=nll,
        mae=mae,
        baseline_crps=baseline_crps,
        baseline_nll=baseline_nll,
        baseline_mae=baseline_mae,
        crps_skill=_skill(pairs, 0, primary_end),
        relative_nll=nll / baseline_nll - 1.0,
        relative_mae=mae / baseline_mae - 1.0,
        coverage_levels=COVERAGE_LEVELS,
        observed_coverage=observed,
        coverage_error=coverage_error,
        regime_crps_skill=regime_skill,
        regime_coverage_error=regime_coverage,
        secondary_horizon_skill=secondary,
        data_seed_primary_skill=seed_skill,
    )
