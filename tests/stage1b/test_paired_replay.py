from __future__ import annotations

from pathlib import Path

import torch

from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.worlds import (
    NodeShock,
    PairedSimulationRequest,
    SimulationRequest,
    build_world,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _base_request(world_id: str, seed: int) -> tuple[object, SimulationRequest]:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    world = build_world(suite.world(world_id))
    request = SimulationRequest(
        seed=seed,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=world.config.regimes[0].regime_id,
        length=40,
        warmup_steps=8,
    )
    return world, request


def test_identity_pair_is_exact_for_same_future_noise() -> None:
    world, request = _base_request("ecology_lv_sde_v1", seed=702)

    pair = world.paired_counterfactual(PairedSimulationRequest(base=request))

    torch.testing.assert_close(pair.factual.values, pair.counterfactual.values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        pair.factual.future_noise,
        pair.counterfactual.future_noise,
        rtol=0.0,
        atol=0.0,
    )


def test_node_shock_uses_same_noise_and_respects_first_effect_lags() -> None:
    world, request = _base_request("network_cml_v1", seed=703)
    shock_step = 10

    pair = world.paired_counterfactual(
        PairedSimulationRequest(
            base=request,
            intervention=NodeShock(source_node=0, step=shock_step, magnitude=0.2),
        )
    )

    torch.testing.assert_close(
        pair.factual.future_noise,
        pair.counterfactual.future_noise,
        rtol=0.0,
        atol=0.0,
    )
    changed = ~torch.isclose(pair.factual.values, pair.counterfactual.values, atol=1e-12, rtol=0.0)
    for target, expected_lag in enumerate(pair.truth.shortest_path_lags[0]):
        changed_steps = torch.where(changed[:, target])[0]
        assert changed_steps.numel() > 0
        assert int(changed_steps[0]) == shock_step + expected_lag
