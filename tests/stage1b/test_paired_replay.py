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
WORLD_CONFIG = REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml"


def _base_request(world_id: str, seed: int):  # type: ignore[no-untyped-def]
    suite = load_world_suite(WORLD_CONFIG)
    world = build_world(suite.world(world_id))
    request = SimulationRequest(
        seed=seed,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=world.config.regimes[0].regime_id,
        length=32,
        warmup_steps=0,
    )
    return world, request


def test_identity_pair_is_bitwise_exact_for_same_future_noise() -> None:
    world, request = _base_request("gvar_predator_prey_v2", seed=702)
    pair = world.paired_counterfactual(PairedSimulationRequest(base=request))
    torch.testing.assert_close(pair.factual.values, pair.counterfactual.values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        pair.factual.future_noise, pair.counterfactual.future_noise, rtol=0.0, atol=0.0
    )


def test_node_shock_changes_only_the_counterfactual_and_reuses_noise() -> None:
    world, request = _base_request("lorenz96_f10_v2", seed=703)
    pair = world.paired_counterfactual(
        PairedSimulationRequest(
            base=request,
            intervention=NodeShock(source_node=0, step=10, magnitude=0.2),
        )
    )
    torch.testing.assert_close(
        pair.factual.future_noise, pair.counterfactual.future_noise, rtol=0.0, atol=0.0
    )
    assert pair.counterfactual.values[10, 0] == pair.factual.values[10, 0] + 0.2
    assert bool(torch.any(pair.counterfactual.values[11:] != pair.factual.values[11:]))
