from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from tarca.stage1b.config import (
    QualificationPartition,
    RegimeConfig,
    RegimeSplitRole,
    WorldAdapter,
    WorldConfig,
)
from tarca.stage1b.health import WorldHealthError, WorldHealthReport, assess_world_health
from tarca.stage1b.truth import WorldTruth, build_world_truth

FloatArray = NDArray[np.float64]


class TrajectoryValidationError(RuntimeError):
    """Raised when a simulation cannot satisfy the frozen world contract."""


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    seed: int
    partition: QualificationPartition
    regime_id: str
    length: int
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("simulation seed must be a nonnegative integer")
        if self.length < 4:
            raise ValueError("simulation length must be at least four")
        if self.warmup_steps < 0:
            raise ValueError("warmup steps must be nonnegative")


@dataclass(frozen=True, slots=True)
class NodeShock:
    source_node: int
    step: int
    magnitude: float

    def __post_init__(self) -> None:
        if self.source_node < 0 or self.step < 0:
            raise ValueError("shock node and step must be nonnegative")
        if not np.isfinite(self.magnitude) or self.magnitude == 0:
            raise ValueError("shock magnitude must be finite and nonzero")


@dataclass(frozen=True, slots=True)
class PairedSimulationRequest:
    base: SimulationRequest
    intervention: NodeShock | None = None


@dataclass(frozen=True, slots=True)
class SimulatedTrajectory:
    values: torch.Tensor
    times: torch.Tensor
    future_noise: torch.Tensor
    future_noise_sha256: str
    truth: WorldTruth
    health: WorldHealthReport
    boundary_event_count: int
    world_id: str
    family_id: str
    regime_id: str
    partition: QualificationPartition
    seed: int


@dataclass(frozen=True, slots=True)
class PairedTrajectory:
    factual: SimulatedTrajectory
    counterfactual: SimulatedTrajectory
    truth: WorldTruth
    intervention: NodeShock | None


def lorenz96_tendency(state: FloatArray, *, forcing: float) -> FloatArray:
    return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + forcing


def two_scale_lorenz96_tendency(
    slow: FloatArray,
    fast: FloatArray,
    *,
    h: float,
    forcing: float,
    b: float,
    c: float,
) -> tuple[FloatArray, FloatArray]:
    if fast.size % slow.size:
        raise ValueError("fast variables must form equal groups for each slow variable")
    per_slow = fast.size // slow.size
    slow_dt = lorenz96_tendency(slow, forcing=forcing)
    slow_dt -= h * c / b * fast.reshape(slow.size, per_slow).sum(axis=1)
    fast_dt = (
        -c * b * np.roll(fast, -1) * (np.roll(fast, -2) - np.roll(fast, 1))
        - c * fast
        + h * c / b * np.repeat(slow, per_slow)
    )
    return slow_dt, fast_dt


def predator_prey_tendency(
    prey: FloatArray,
    predator: FloatArray,
    *,
    parents_per_node: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> tuple[FloatArray, FloatArray]:
    if prey.shape != predator.shape or prey.size % parents_per_node:
        raise ValueError("predator-prey state does not match its block graph")
    prey_dt = np.empty_like(prey)
    predator_dt = np.empty_like(predator)
    for index in range(prey.size):
        start = (index // parents_per_node) * parents_per_node
        peer_slice = slice(start, start + parents_per_node)
        prey_dt[index] = (
            alpha * prey[index]
            - beta * prey[index] * predator[peer_slice].sum()
            - alpha * (prey[index] / 200.0) ** 2
        )
        predator_dt[index] = (
            delta * prey[peer_slice].sum() * predator[index] - gamma * predator[index]
        )
    return prey_dt, predator_dt


def corrected_cml_step(values: FloatArray, *, alpha: float, epsilon: float) -> FloatArray:
    mapped = 1.0 - alpha * values**2
    neighbor_mean = (np.roll(mapped, 1) + np.roll(mapped, -1)) / 2.0
    return cast(FloatArray, (1.0 - epsilon) * mapped + epsilon * neighbor_mean)


def _rk4_step(
    state: FloatArray,
    step: float,
    tendency: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    first = tendency(state)
    second = tendency(state + first * step / 2.0)
    third = tendency(state + second * step / 2.0)
    fourth = tendency(state + third * step)
    return cast(
        FloatArray,
        state + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0,
    )


def _measurement_noise(
    rng: np.random.Generator, count: int, dimension: int, scale: float
) -> FloatArray:
    return rng.normal(0.0, scale, size=(count, dimension)).astype(np.float64)


class PublishedWorldAdapter:
    formal_origin = "LOCAL_DIAGNOSTIC_ONLY"

    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self.truth = build_world_truth(config)

    def simulate(self, request: SimulationRequest) -> SimulatedTrajectory:
        return self._simulate(request, intervention=None)

    def paired_counterfactual(self, request: PairedSimulationRequest) -> PairedTrajectory:
        factual = self._simulate(request.base, intervention=None)
        counterfactual = self._simulate(request.base, intervention=request.intervention)
        if not torch.equal(factual.future_noise, counterfactual.future_noise):
            raise TrajectoryValidationError("paired trajectories do not share future noise")
        return PairedTrajectory(factual, counterfactual, self.truth, request.intervention)

    def validate_values(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.config.dimension:
            raise TrajectoryValidationError("trajectory topology does not match world dimension")
        if not bool(torch.isfinite(values).all()):
            raise TrajectoryValidationError("trajectory contains non-finite values")

    def _regime(self, request: SimulationRequest) -> RegimeConfig:
        matches = tuple(item for item in self.config.regimes if item.regime_id == request.regime_id)
        if len(matches) != 1:
            raise ValueError(f"unknown regime for {self.config.world_id}: {request.regime_id}")
        regime = matches[0]
        expected = (
            RegimeSplitRole.UNSEEN
            if request.partition is QualificationPartition.QUAL_UNSEEN
            else RegimeSplitRole.SEEN
        )
        if regime.split_role is not expected:
            raise ValueError("qualification partition and regime split role do not match")
        return regime

    def _validate_intervention(
        self, request: SimulationRequest, intervention: NodeShock | None
    ) -> None:
        if intervention is None:
            return
        if intervention.source_node >= self.config.dimension:
            raise ValueError("shock source node is outside the world")
        if intervention.step >= request.length:
            raise ValueError("shock step is outside the returned trajectory")

    def _simulate(
        self, request: SimulationRequest, intervention: NodeShock | None
    ) -> SimulatedTrajectory:
        self._validate_intervention(request, intervention)
        regime = self._regime(request)
        values_array, times_array, noise_array, boundary_event_count = self._simulate_published(
            request, regime, intervention
        )
        values = torch.from_numpy(np.ascontiguousarray(values_array)).to(torch.float64)
        times = torch.from_numpy(np.ascontiguousarray(times_array)).to(torch.float64)
        noise = torch.from_numpy(np.ascontiguousarray(noise_array)).to(torch.float64)
        self.validate_values(values)
        try:
            health = assess_world_health(values)
        except WorldHealthError as exc:
            raise TrajectoryValidationError(f"world health failed: {exc}") from exc
        return SimulatedTrajectory(
            values=values,
            times=times,
            future_noise=noise,
            future_noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
            truth=self.truth,
            health=health,
            boundary_event_count=boundary_event_count,
            world_id=self.config.world_id,
            family_id=self.config.family_id,
            regime_id=regime.regime_id,
            partition=request.partition,
            seed=request.seed,
        )

    def _simulate_published(
        self,
        request: SimulationRequest,
        regime: RegimeConfig,
        intervention: NodeShock | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, int]:
        raise NotImplementedError


def _apply_shock(state: FloatArray, intervention: NodeShock | None, returned_step: int) -> None:
    if intervention is not None and returned_step == intervention.step:
        state[intervention.source_node] += intervention.magnitude


class VarWorldAdapter(PublishedWorldAdapter):
    def _simulate_published(self, request, regime, intervention):  # type: ignore[no-untyped-def]
        generator = self.config.generator_map()
        count = request.warmup_steps + request.length
        rng = np.random.default_rng(request.seed)
        innovations = _measurement_noise(
            rng, count, self.config.dimension, generator["innovation_scale"]
        )
        ring = np.roll(np.eye(self.config.dimension), 1, axis=0)
        coefficient = 0.75 * np.eye(self.config.dimension) + 0.25 * ring
        coefficient *= generator["spectral_radius"] * regime.parameter_map()["coefficient_scale"]
        state = rng.normal(0.0, generator["innovation_scale"], self.config.dimension)
        outputs = np.empty((request.length, self.config.dimension), dtype=np.float64)
        for index in range(count):
            state = coefficient @ state + innovations[index]
            returned = index - request.warmup_steps
            if returned >= 0:
                _apply_shock(state, intervention, returned)
                outputs[returned] = state
        return outputs, np.arange(request.length, dtype=np.float64), innovations, 0


class Lorenz96WorldAdapter(PublishedWorldAdapter):
    def _simulate_published(self, request, regime, intervention):  # type: ignore[no-untyped-def]
        generator = self.config.generator_map()
        burn = int(generator["burn_in_observations"]) + request.warmup_steps
        count = burn + request.length
        dt = generator["integration_step"]
        observation_interval = generator["observation_interval"]
        internal_steps = round(observation_interval / dt)
        rng = np.random.default_rng(request.seed)
        state = rng.uniform(0.0, 1.0, self.config.dimension).astype(np.float64)
        measurement = _measurement_noise(
            rng,
            count,
            self.config.dimension,
            regime.parameter_map()["measurement_noise"],
        )
        outputs = np.empty((request.length, self.config.dimension), dtype=np.float64)

        def tendency(value: FloatArray) -> FloatArray:
            return lorenz96_tendency(value, forcing=generator["forcing"])

        for observation in range(count):
            returned = observation - burn
            if returned >= 0:
                _apply_shock(state, intervention, returned)
                outputs[returned] = state + measurement[observation]
            for _ in range(internal_steps):
                state = _rk4_step(state, dt, tendency)
        times = np.arange(request.length, dtype=np.float64) * observation_interval
        return outputs, times, measurement, 0


class TwoScaleLorenz96WorldAdapter(PublishedWorldAdapter):
    def _simulate_published(self, request, regime, intervention):  # type: ignore[no-untyped-def]
        generator = self.config.generator_map()
        burn = int(generator["burn_in_observations"]) + request.warmup_steps
        count = burn + request.length
        dt = generator["integration_step"]
        observation_interval = generator["observation_interval"]
        internal_steps = round(observation_interval / dt)
        rng = np.random.default_rng(request.seed)
        slow = rng.uniform(0.0, 1.0, self.config.dimension).astype(np.float64)
        fast = rng.uniform(-0.5, 0.5, self.config.latent_dimension).astype(np.float64)
        measurement = _measurement_noise(
            rng,
            count,
            self.config.dimension,
            regime.parameter_map()["measurement_noise"],
        )
        outputs = np.empty((request.length, self.config.dimension), dtype=np.float64)
        h, forcing = generator["coupling_h"], generator["forcing"]
        b, c = generator["fast_amplitude_b"], generator["time_scale_c"]
        for observation in range(count):
            returned = observation - burn
            if returned >= 0:
                _apply_shock(slow, intervention, returned)
                outputs[returned] = slow + measurement[observation]
            for _ in range(internal_steps):
                k1x, k1y = two_scale_lorenz96_tendency(slow, fast, h=h, forcing=forcing, b=b, c=c)
                k2x, k2y = two_scale_lorenz96_tendency(
                    slow + k1x * dt / 2, fast + k1y * dt / 2, h=h, forcing=forcing, b=b, c=c
                )
                k3x, k3y = two_scale_lorenz96_tendency(
                    slow + k2x * dt / 2, fast + k2y * dt / 2, h=h, forcing=forcing, b=b, c=c
                )
                k4x, k4y = two_scale_lorenz96_tendency(
                    slow + k3x * dt, fast + k3y * dt, h=h, forcing=forcing, b=b, c=c
                )
                slow = slow + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
                fast = fast + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        times = np.arange(request.length, dtype=np.float64) * observation_interval
        return outputs, times, measurement, 0


class PredatorPreyWorldAdapter(PublishedWorldAdapter):
    def _simulate_published(self, request, regime, intervention):  # type: ignore[no-untyped-def]
        generator = self.config.generator_map()
        count = request.warmup_steps + request.length
        internal_steps = round(generator["observation_interval"] / generator["integration_step"])
        total_internal = count * internal_steps
        rng = np.random.default_rng(request.seed)
        species = self.config.dimension // 2
        state = rng.uniform(10.0, 100.0, self.config.dimension).astype(np.float64)
        innovations = _measurement_noise(
            rng,
            total_internal,
            self.config.dimension,
            regime.parameter_map()["dynamic_noise_scale"],
        )
        measurement = np.zeros((count, self.config.dimension), dtype=np.float64)
        outputs = np.empty((request.length, self.config.dimension), dtype=np.float64)
        dt = generator["integration_step"]
        parents = int(generator["parents_per_node"])

        def tendency(value: FloatArray) -> FloatArray:
            prey_dt, predator_dt = predator_prey_tendency(
                value[:species],
                value[species:],
                parents_per_node=parents,
                alpha=generator["alpha"],
                beta=generator["beta"],
                gamma=generator["gamma"],
                delta=generator["delta"],
            )
            return np.concatenate((prey_dt, predator_dt))

        clipped = 0
        noise_index = 0
        for observation in range(count):
            returned = observation - request.warmup_steps
            if returned >= 0:
                _apply_shock(state, intervention, returned)
                outputs[returned] = state + measurement[observation]
            for _ in range(internal_steps):
                proposed = _rk4_step(state, dt, tendency) + innovations[noise_index]
                clipped += int(np.count_nonzero(proposed < 0.0))
                state = np.maximum(proposed, 0.0)
                noise_index += 1
        if clipped and self.config.boundary_policy != "DECLARED_ZERO_CLIP":
            raise TrajectoryValidationError(
                f"published predator-prey integration required {clipped} undeclared boundary clips"
            )
        noise = np.concatenate((innovations, measurement), axis=0)
        times = np.arange(request.length, dtype=np.float64) * generator["observation_interval"]
        return outputs, times, noise, clipped


class CorrectedCmlWorldAdapter(PublishedWorldAdapter):
    def _simulate_published(self, request, regime, intervention):  # type: ignore[no-untyped-def]
        generator = self.config.generator_map()
        count = request.warmup_steps + request.length
        rng = np.random.default_rng(request.seed)
        noise = _measurement_noise(rng, count, self.config.dimension, generator["sigma"])
        state = rng.uniform(-0.8, 0.8, self.config.dimension).astype(np.float64)
        outputs = np.empty((request.length, self.config.dimension), dtype=np.float64)
        epsilon = regime.parameter_map()["epsilon"]
        for observation in range(count):
            returned = observation - request.warmup_steps
            if returned >= 0:
                _apply_shock(state, intervention, returned)
                outputs[returned] = state
            state = corrected_cml_step(state, alpha=generator["alpha"], epsilon=epsilon)
            state = state + noise[observation]
            if np.any(np.abs(state) > 1.0 + 1e-12):
                raise TrajectoryValidationError(
                    "corrected CML left its declared [-1, 1] state space"
                )
        return outputs, np.arange(request.length, dtype=np.float64), noise, 0


def build_world(config: WorldConfig) -> PublishedWorldAdapter:
    adapters: dict[WorldAdapter, type[PublishedWorldAdapter]] = {
        WorldAdapter.TARCA_VAR: VarWorldAdapter,
        WorldAdapter.LORENZ96: Lorenz96WorldAdapter,
        WorldAdapter.LORENZ96_TWO_SCALE: TwoScaleLorenz96WorldAdapter,
        WorldAdapter.GVAR_PREDATOR_PREY: PredatorPreyWorldAdapter,
        WorldAdapter.CORRECTED_CML: CorrectedCmlWorldAdapter,
    }
    return adapters[config.adapter](config)
