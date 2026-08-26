from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.health import WorldHealthError, assess_world_health
from tarca.stage1b.worlds import (
    SimulationRequest,
    TrajectoryValidationError,
    build_world,
    corrected_cml_step,
    lorenz96_tendency,
    predator_prey_tendency,
    two_scale_lorenz96_tendency,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORLD_CONFIG = REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml"


def test_lorenz96_tendency_matches_published_index_equation() -> None:
    state = np.asarray([0.2, -0.3, 0.7, 1.1, -0.4], dtype=np.float64)
    expected = np.asarray(
        [
            (state[(i + 1) % 5] - state[(i - 2) % 5]) * state[(i - 1) % 5] - state[i] + 10.0
            for i in range(5)
        ]
    )
    np.testing.assert_allclose(lorenz96_tendency(state, forcing=10.0), expected)


def test_two_scale_tendency_matches_published_constant_state_case() -> None:
    slow = np.ones(8, dtype=np.float64)
    fast = np.zeros(256, dtype=np.float64)
    slow_dt, fast_dt = two_scale_lorenz96_tendency(slow, fast, h=1.0, forcing=20.0, b=10.0, c=10.0)
    np.testing.assert_allclose(slow_dt, np.full(8, 19.0))
    np.testing.assert_allclose(fast_dt, np.ones(256))


def test_predator_prey_tendency_matches_gvar_equation() -> None:
    prey = np.full(2, 10.0)
    predator = np.full(2, 2.0)
    prey_dt, predator_dt = predator_prey_tendency(
        prey,
        predator,
        parents_per_node=1,
        alpha=1.1,
        beta=0.2,
        gamma=1.1,
        delta=0.2,
    )
    np.testing.assert_allclose(prey_dt, np.full(2, 6.99725))
    np.testing.assert_allclose(predator_dt, np.full(2, 1.8))


def test_corrected_cml_divides_only_neighbor_sum_by_degree() -> None:
    values = np.asarray([0.0, 0.5, -0.5, 0.25], dtype=np.float64)
    mapped = 1.0 - 2.0 * values**2
    expected_first = 0.7 * mapped[0] + 0.3 * (mapped[1] + mapped[3]) / 2.0
    stepped = corrected_cml_step(values, alpha=2.0, epsilon=0.3)
    assert stepped[0] == pytest.approx(expected_first)


@pytest.mark.parametrize(
    "world_id",
    [
        "control_var_v2",
        "lorenz96_f10_v2",
        "lorenz96_f40_v2",
        "lorenz96_twoscale_v2",
        "gvar_predator_prey_v2",
        "corrected_cml_v2",
    ],
)
def test_world_replay_is_bitwise_deterministic_and_healthy(world_id: str) -> None:
    suite = load_world_suite(WORLD_CONFIG)
    world = build_world(suite.world(world_id))
    request = SimulationRequest(
        seed=701,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=world.config.regimes[0].regime_id,
        length=24,
        warmup_steps=0,
    )
    first = world.simulate(request)
    second = world.simulate(request)
    torch.testing.assert_close(first.values, second.values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.future_noise, second.future_noise, rtol=0.0, atol=0.0)
    assert first.future_noise_sha256 == second.future_noise_sha256
    assert first.values.shape == (24, world.config.dimension)
    assert first.health.passed
    if world_id == "gvar_predator_prey_v2":
        assert first.boundary_event_count > 0
    else:
        assert first.boundary_event_count == 0


def test_lorenz_truth_contains_published_parents_and_state_dependent_signs() -> None:
    suite = load_world_suite(WORLD_CONFIG)
    truth = build_world(suite.world("lorenz96_f10_v2")).truth
    sources_for_zero = set(torch.where(truth.adjacency[0] != 0)[0].tolist())
    assert sources_for_zero == {0, 1, 18, 19}
    assert truth.signed_adjacency[0, 0].item() == -1
    assert truth.signed_adjacency[0, 1].item() == 2
    assert truth.shortest_path_lags[0][0] == 0


def test_two_scale_truth_records_latent_dimension() -> None:
    suite = load_world_suite(WORLD_CONFIG)
    truth = build_world(suite.world("lorenz96_twoscale_v2")).truth
    assert truth.latent_dimension == 256
    assert truth.adjacency.shape == (8, 8)


def test_health_rejects_constant_and_period_two_trajectories() -> None:
    constant = torch.ones((20, 4), dtype=torch.float64)
    period_two = torch.tensor([[0.0, 1.0], [1.0, 0.0]] * 10, dtype=torch.float64)
    with pytest.raises(WorldHealthError, match="collapsed"):
        assess_world_health(constant)
    with pytest.raises(WorldHealthError, match="period-2"):
        assess_world_health(period_two)


def test_nonfinite_trajectory_fails_closed() -> None:
    suite = load_world_suite(WORLD_CONFIG)
    world = build_world(suite.world("gvar_predator_prey_v2"))
    values = torch.zeros((4, 20), dtype=torch.float64)
    values[0, 0] = float("inf")
    with pytest.raises(TrajectoryValidationError, match="non-finite"):
        world.validate_values(values)
