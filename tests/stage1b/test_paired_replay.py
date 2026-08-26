from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.oracle import (
    ConceptSchedule,
    OfficialSimulation,
    OraclePairRequest,
    paired_rollout,
)
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


class RecordingOracleDriver:
    def __init__(self, *, corrupt_initial_state: bool = False) -> None:
        self.sample_calls = 0
        self.simulate_calls = 0
        self.corrupt_initial_state = corrupt_initial_state

    def sample_future_noise(self, request: SimulationRequest) -> torch.Tensor:
        self.sample_calls += 1
        return torch.arange(request.length * 2, dtype=torch.float64).reshape(request.length, 2)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: torch.Tensor,
    ) -> OfficialSimulation:
        self.simulate_calls += 1
        initial = torch.tensor([1.0, 2.0], dtype=torch.float64)
        if self.corrupt_initial_state and self.simulate_calls == 2:
            initial = initial + 1.0
        values = initial + schedule.trend[:, None] + schedule.scale[:, None] * future_noise
        return OfficialSimulation(
            values=values,
            times=torch.arange(request.length, dtype=torch.float64),
            initial_state=initial,
            future_noise=future_noise,
            regime_sequence=torch.zeros(request.length, dtype=torch.int64),
            boundary_event_count=0,
        )


def _oracle_request(changed_concept: str = "identity") -> OraclePairRequest:
    _, base = _base_request("lorenz96_f10_v2", seed=711)
    factual = ConceptSchedule(
        trend=torch.ones(base.length, dtype=torch.float64),
        scale=torch.full((base.length,), 0.1, dtype=torch.float64),
    )
    counterfactual = ConceptSchedule(
        trend=factual.trend.clone(),
        scale=factual.scale.clone(),
    )
    if changed_concept == "trend":
        counterfactual = ConceptSchedule(
            trend=factual.trend + 0.5,
            scale=factual.scale.clone(),
        )
    return OraclePairRequest(
        base=base,
        factual_schedule=factual,
        counterfactual_schedule=counterfactual,
        changed_concept=changed_concept,  # type: ignore[arg-type]
    )


def test_generator_owned_pair_samples_noise_once_and_identity_is_bitwise() -> None:
    driver = RecordingOracleDriver()

    pair = paired_rollout(driver, _oracle_request())

    assert driver.sample_calls == 1
    assert driver.simulate_calls == 2
    assert torch.equal(pair.factual.future_noise, pair.counterfactual.future_noise)
    assert torch.equal(pair.factual.values, pair.counterfactual.values)


def test_generator_owned_pair_rejects_changed_initial_state() -> None:
    with pytest.raises(ValueError, match="initial state"):
        paired_rollout(RecordingOracleDriver(corrupt_initial_state=True), _oracle_request("trend"))


def test_oracle_request_rejects_non_target_schedule_change() -> None:
    request = _oracle_request("trend")
    contaminated = OraclePairRequest(
        base=request.base,
        factual_schedule=request.factual_schedule,
        counterfactual_schedule=ConceptSchedule(
            trend=request.counterfactual_schedule.trend,
            scale=request.counterfactual_schedule.scale + 0.1,
        ),
        changed_concept="trend",
    )

    with pytest.raises(ValueError, match="non-target scale"):
        paired_rollout(RecordingOracleDriver(), contaminated)
