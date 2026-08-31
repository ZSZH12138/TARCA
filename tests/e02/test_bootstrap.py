from __future__ import annotations

from pathlib import Path

import pytest

from tarca.e02.bootstrap import paired_stratified_bootstrap
from tarca.e02.config import load_e02_config
from tarca.e02.scoring import TrajectoryScore

E02_CONFIG = load_e02_config(Path("configs/e02/e02_v1.yaml"))


def _scores() -> tuple[tuple[TrajectoryScore, ...], tuple[TrajectoryScore, ...]]:
    neural: list[TrajectoryScore] = []
    baseline: list[TrajectoryScore] = []
    for seed in E02_CONFIG.formal_seeds:
        for regime in ("SEEN", "UNSEEN"):
            for index in range(12):
                identifier = f"{seed}-{regime.lower()}-{index}"
                shared = {
                    "trajectory_id": identifier,
                    "formal_seed": seed,
                    "regime": regime,
                    "origin_count": 2,
                    "horizon_count": 24,
                    "variable_count": 1,
                    "nll": 1.0,
                    "mae": 0.5,
                    "coverage": ((0.5, 0.5), (0.8, 0.8), (0.9, 0.9), (0.95, 0.95)),
                    "horizon_nll": (1.0,) * 24,
                    "horizon_mae": (0.5,) * 24,
                    "horizon_coverage": tuple(
                        (level, (level,) * 24) for level in (0.5, 0.8, 0.9, 0.95)
                    ),
                }
                baseline.append(
                    TrajectoryScore(
                        crps=1.0,
                        horizon_crps=(1.0,) * 24,
                        **shared,
                    )
                )
                neural.append(
                    TrajectoryScore(
                        crps=0.9,
                        horizon_crps=(0.9 + index / 1000.0,) * 24,
                        **shared,
                    )
                )
    return tuple(neural), tuple(baseline)


def test_bootstrap_resamples_twelve_trajectories_inside_each_stratum() -> None:
    neural, baseline = _scores()

    interval = paired_stratified_bootstrap(neural, baseline, E02_CONFIG.bootstrap)

    assert interval.replicates == 5000
    assert interval.units_per_replicate == 120
    assert interval.stratum_count == 10
    assert interval.estimate == pytest.approx(0.0945)
    assert interval.lower < interval.estimate < interval.upper


def test_bootstrap_is_deterministic_and_input_order_invariant() -> None:
    neural, baseline = _scores()

    first = paired_stratified_bootstrap(neural, baseline, E02_CONFIG.bootstrap)
    reordered = paired_stratified_bootstrap(
        tuple(reversed(neural)), tuple(reversed(baseline)), E02_CONFIG.bootstrap
    )

    assert first == reordered


def test_bootstrap_rejects_119_or_duplicate_trajectory_units() -> None:
    neural, baseline = _scores()
    with pytest.raises(ValueError, match="120"):
        paired_stratified_bootstrap(neural[:-1], baseline[:-1], E02_CONFIG.bootstrap)
    with pytest.raises(ValueError, match="unique"):
        paired_stratified_bootstrap(
            (*neural[:-1], neural[0]), (*baseline[:-1], baseline[0]), E02_CONFIG.bootstrap
        )

