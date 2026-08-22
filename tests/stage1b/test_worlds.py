from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.worlds import (
    SimulationRequest,
    TrajectoryValidationError,
    build_world,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world_id", ["network_cml_v1", "ecology_lv_sde_v1"])
def test_world_replay_is_bitwise_deterministic(world_id: str) -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world(world_id))
    request = SimulationRequest(
        seed=701,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=world.config.regimes[0].regime_id,
        length=48,
        warmup_steps=8,
    )

    first = world.simulate(request)
    second = world.simulate(request)

    torch.testing.assert_close(first.values, second.values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.future_noise, second.future_noise, rtol=0.0, atol=0.0)
    assert first.future_noise_sha256 == second.future_noise_sha256
    assert first.values.shape == (48, 8)
    assert first.truth.adjacency.shape == (8, 8)


def test_truth_contains_ring_graph_and_path_lags() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world("network_cml_v1"))

    truth = world.truth

    assert truth.shortest_path_lags[0][0] == 0
    assert truth.shortest_path_lags[0][1] == 1
    assert truth.shortest_path_lags[0][4] == 4
    assert int(truth.adjacency.sum().item()) == 16


def test_nonfinite_trajectory_fails_closed() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world("ecology_lv_sde_v1"))

    with pytest.raises(TrajectoryValidationError, match="non-finite"):
        values = torch.zeros((1, 8))
        values[0, 0] = float("inf")
        world.validate_values(values)


def test_coupled_map_boundary_clipping_fails_closed() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world("network_cml_v1"))
    clipped = torch.zeros((2, 8), dtype=torch.float64)
    clipped[1, 3] = 1.0

    with pytest.raises(TrajectoryValidationError, match="boundary clipping"):
        world.validate_values(clipped)


def test_linear_control_world_uses_external_varma_adapter() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world("control_varma_v1"))
    request = SimulationRequest(
        seed=704,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id="linear_seen",
        length=32,
        warmup_steps=8,
    )

    trajectory = world.simulate(request)

    assert trajectory.values.shape == (32, 8)
    assert bool(torch.isfinite(trajectory.values).all())
