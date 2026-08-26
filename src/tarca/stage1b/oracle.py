from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from tarca.stage1b.worlds import SimulationRequest


def _immutable_tensor(value: Tensor, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{label} must be a rank-{ndim} Tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain finite values")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class ConceptSchedule:
    trend: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        trend = _immutable_tensor(self.trend, "trend schedule", 1)
        scale = _immutable_tensor(self.scale, "scale schedule", 1)
        if trend.shape != scale.shape or trend.numel() == 0:
            raise ValueError("trend and scale schedules must share a nonempty shape")
        if not trend.is_floating_point() or not scale.is_floating_point():
            raise ValueError("concept schedules must have floating dtype")
        if not bool((scale > 0).all()):
            raise ValueError("scale schedule must be strictly positive")
        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, slots=True)
class OraclePairRequest:
    base: SimulationRequest
    factual_schedule: ConceptSchedule
    counterfactual_schedule: ConceptSchedule
    changed_concept: Literal["trend", "scale", "identity"]

    def __post_init__(self) -> None:
        if self.changed_concept not in {"trend", "scale", "identity"}:
            raise ValueError("changed_concept is not registered")
        expected = (self.base.length,)
        if self.factual_schedule.trend.shape != expected:
            raise ValueError("factual concept schedule length must match simulation")
        if self.counterfactual_schedule.trend.shape != expected:
            raise ValueError("counterfactual concept schedule length must match simulation")


@dataclass(frozen=True, slots=True)
class OfficialSimulation:
    values: Tensor
    times: Tensor
    initial_state: Tensor
    future_noise: Tensor
    regime_sequence: Tensor
    boundary_event_count: int

    def __post_init__(self) -> None:
        values = _immutable_tensor(self.values, "simulation values", 2)
        times = _immutable_tensor(self.times, "simulation times", 1)
        initial = _immutable_tensor(self.initial_state, "initial state", 1)
        noise = _immutable_tensor(self.future_noise, "future noise", 2)
        regimes = _immutable_tensor(self.regime_sequence, "regime sequence", 1)
        length, dimension = values.shape
        if tuple(times.shape) != (length,) or tuple(regimes.shape) != (length,):
            raise ValueError("simulation times and regimes must match value length")
        if tuple(initial.shape) != (dimension,) or tuple(noise.shape) != (length, dimension):
            raise ValueError("simulation initial state or noise shape is invalid")
        if self.boundary_event_count < 0:
            raise ValueError("boundary event count must be nonnegative")
        for name, tensor in (
            ("values", values),
            ("times", times),
            ("initial_state", initial),
            ("future_noise", noise),
            ("regime_sequence", regimes),
        ):
            object.__setattr__(self, name, tensor)


class OfficialWorldDriver(Protocol):
    def sample_future_noise(self, request: SimulationRequest) -> Tensor: ...

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation: ...


@dataclass(frozen=True, slots=True)
class PairedTrajectory:
    factual: OfficialSimulation
    counterfactual: OfficialSimulation
    changed_concept: Literal["trend", "scale", "identity"]


def _validate_schedules(request: OraclePairRequest) -> None:
    factual = request.factual_schedule
    counterfactual = request.counterfactual_schedule
    trend_equal = torch.equal(factual.trend, counterfactual.trend)
    scale_equal = torch.equal(factual.scale, counterfactual.scale)
    if request.changed_concept == "identity" and not (trend_equal and scale_equal):
        raise ValueError("identity pair must preserve both concept schedules")
    if request.changed_concept == "trend":
        if not scale_equal:
            raise ValueError("trend pair changed the non-target scale schedule")
        if trend_equal:
            raise ValueError("trend pair must change the target trend schedule")
    if request.changed_concept == "scale":
        if not trend_equal:
            raise ValueError("scale pair changed the non-target trend schedule")
        if scale_equal:
            raise ValueError("scale pair must change the target scale schedule")


def _validate_pair(
    request: OraclePairRequest,
    sampled_noise: Tensor,
    factual: OfficialSimulation,
    counterfactual: OfficialSimulation,
) -> PairedTrajectory:
    if not torch.equal(factual.initial_state, counterfactual.initial_state):
        raise ValueError("paired simulations do not share the initial state")
    if not torch.equal(factual.future_noise, counterfactual.future_noise):
        raise ValueError("paired simulations do not share future noise")
    if not torch.equal(factual.future_noise, sampled_noise):
        raise ValueError("simulation did not preserve generator-sampled future noise")
    if not torch.equal(factual.times, counterfactual.times):
        raise ValueError("paired simulations do not share time coordinates")
    if not torch.equal(factual.regime_sequence, counterfactual.regime_sequence):
        raise ValueError("paired simulations do not share regimes")
    if request.changed_concept == "identity" and not torch.equal(
        factual.values, counterfactual.values
    ):
        raise ValueError("identity paired rollout must be bitwise exact")
    return PairedTrajectory(
        factual=factual,
        counterfactual=counterfactual,
        changed_concept=request.changed_concept,
    )


def paired_rollout(
    driver: OfficialWorldDriver,
    request: OraclePairRequest,
) -> PairedTrajectory:
    _validate_schedules(request)
    sampled_noise = _immutable_tensor(
        driver.sample_future_noise(request.base),
        "sampled future noise",
        2,
    )
    factual = driver.simulate(
        request.base,
        request.factual_schedule,
        sampled_noise.clone(),
    )
    counterfactual = driver.simulate(
        request.base,
        request.counterfactual_schedule,
        sampled_noise.clone(),
    )
    return _validate_pair(request, sampled_noise, factual, counterfactual)
