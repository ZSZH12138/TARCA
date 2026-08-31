from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tarca.e02.config import E02BootstrapConfig
from tarca.e02.scoring import TrajectoryScore

_FORMAL_SEEDS = (1729, 2718, 3141, 5772, 8111)
_REGIMES = ("SEEN", "UNSEEN")


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    units_per_replicate: int
    stratum_count: int


def _validated_pairs(
    neural: tuple[TrajectoryScore, ...],
    baseline: tuple[TrajectoryScore, ...],
) -> dict[tuple[int, str], tuple[tuple[TrajectoryScore, TrajectoryScore], ...]]:
    if len(neural) != 120 or len(baseline) != 120:
        raise ValueError("E02 bootstrap requires exactly 120 trajectory units")
    neural_by_id = {score.trajectory_id: score for score in neural}
    baseline_by_id = {score.trajectory_id: score for score in baseline}
    if len(neural_by_id) != 120 or len(baseline_by_id) != 120:
        raise ValueError("E02 bootstrap trajectory IDs must be unique")
    if set(neural_by_id) != set(baseline_by_id):
        raise ValueError("E02 bootstrap neural and baseline IDs must match")
    strata: dict[tuple[int, str], list[tuple[TrajectoryScore, TrajectoryScore]]] = {
        (seed, regime): [] for seed in _FORMAL_SEEDS for regime in _REGIMES
    }
    for identifier in sorted(neural_by_id):
        pair = (neural_by_id[identifier], baseline_by_id[identifier])
        if (pair[0].formal_seed, pair[0].regime) != (
            pair[1].formal_seed,
            pair[1].regime,
        ):
            raise ValueError("E02 bootstrap paired lineage must match")
        key = (pair[0].formal_seed, pair[0].regime)
        if key not in strata:
            raise ValueError("E02 bootstrap contains an unknown stratum")
        strata[key].append(pair)
    if any(len(pairs) != 12 for pairs in strata.values()):
        raise ValueError("E02 bootstrap requires 12 trajectories in each stratum")
    return {key: tuple(pairs) for key, pairs in strata.items()}


def _primary_crps(score: TrajectoryScore) -> float:
    if score.horizon_count < 6:
        raise ValueError("E02 bootstrap requires at least six forecast horizons")
    return sum(score.horizon_crps[:6]) / 6.0


def paired_stratified_bootstrap(
    neural: tuple[TrajectoryScore, ...],
    baseline: tuple[TrajectoryScore, ...],
    config: E02BootstrapConfig,
) -> BootstrapInterval:
    strata = _validated_pairs(neural, baseline)
    generator = torch.Generator().manual_seed(config.seed)
    neural_totals = torch.zeros(config.replicates, dtype=torch.float64)
    baseline_totals = torch.zeros(config.replicates, dtype=torch.float64)
    for key in sorted(strata):
        pairs = strata[key]
        neural_values = torch.tensor(
            [_primary_crps(score) for score, _ in pairs], dtype=torch.float64
        )
        baseline_values = torch.tensor(
            [_primary_crps(score) for _, score in pairs], dtype=torch.float64
        )
        indices = torch.randint(
            0,
            12,
            (config.replicates, 12),
            generator=generator,
        )
        neural_totals += neural_values[indices].sum(dim=1)
        baseline_totals += baseline_values[indices].sum(dim=1)
    if bool((baseline_totals <= 0).any()):
        raise ValueError("E02 bootstrap baseline CRPS must be positive")
    skills = 1.0 - neural_totals / baseline_totals
    lower_tail = (1.0 - config.confidence) / 2.0
    lower, upper = torch.quantile(
        skills,
        torch.tensor([lower_tail, 1.0 - lower_tail], dtype=torch.float64),
    )
    point_neural = sum(_primary_crps(score) for score in neural) / len(neural)
    point_baseline = sum(_primary_crps(score) for score in baseline) / len(baseline)
    estimate = 1.0 - point_neural / point_baseline
    if not all(math.isfinite(value) for value in (estimate, float(lower), float(upper))):
        raise RuntimeError("E02 bootstrap produced a non-finite interval")
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=config.confidence,
        replicates=config.replicates,
        units_per_replicate=120,
        stratum_count=10,
    )

