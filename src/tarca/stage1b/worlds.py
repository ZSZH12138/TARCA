from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from interfere.dynamics import (  # type: ignore[import-untyped]
    LotkaVolterraSDE,
    StochasticCoupledMapLattice,
    VARMADynamics,
)
from interfere.dynamics.coupled_map_lattice import quadradic_map  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tarca.stage1b.config import (
    QualificationPartition,
    RegimeConfig,
    RegimeSplitRole,
    WorldAdapter,
    WorldConfig,
)
from tarca.stage1b.truth import WorldTruth, build_world_truth


class TrajectoryValidationError(RuntimeError):
    """Raised when an upstream simulation cannot satisfy the world contract."""


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
        if self.length < 2:
            raise ValueError("simulation length must be at least two")
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


FloatArray = NDArray[np.float64]
Intervention = Callable[[FloatArray, float], FloatArray]


class ExternalWorldAdapter:
    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self.truth = build_world_truth(
            world_id=config.world_id,
            dimension=config.dimension,
            directed=config.graph.directed,
            concepts=config.concepts,
        )

    def simulate(self, request: SimulationRequest) -> SimulatedTrajectory:
        return self._simulate(request, intervention=None)

    def paired_counterfactual(self, request: PairedSimulationRequest) -> PairedTrajectory:
        factual = self._simulate(request.base, intervention=None)
        counterfactual = self._simulate(request.base, intervention=request.intervention)
        if not torch.equal(factual.future_noise, counterfactual.future_noise):
            raise TrajectoryValidationError("paired trajectories do not share future noise")
        return PairedTrajectory(
            factual=factual,
            counterfactual=counterfactual,
            truth=self.truth,
            intervention=request.intervention,
        )

    def validate_values(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.config.dimension:
            raise TrajectoryValidationError("trajectory topology does not match world dimension")
        if not bool(torch.isfinite(values).all()):
            raise TrajectoryValidationError("trajectory contains non-finite values")

    def _regime(self, request: SimulationRequest) -> RegimeConfig:
        matches = tuple(
            regime
            for regime in self.config.regimes
            if regime.regime_id == request.regime_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown regime for {self.config.world_id}: {request.regime_id}")
        regime = matches[0]
        expected_role = (
            RegimeSplitRole.UNSEEN
            if request.partition is QualificationPartition.QUAL_UNSEEN
            else RegimeSplitRole.SEEN
        )
        if regime.split_role is not expected_role:
            raise ValueError("qualification partition and regime split role do not match")
        return regime

    def _simulate(
        self,
        request: SimulationRequest,
        intervention: NodeShock | None,
    ) -> SimulatedTrajectory:
        regime = self._regime(request)
        full_values, full_times, future_noise = self._simulate_upstream(
            request, regime, intervention
        )
        values = torch.from_numpy(
            np.ascontiguousarray(full_values[-request.length :])
        ).to(torch.float64)
        times = torch.from_numpy(
            np.ascontiguousarray(full_times[-request.length :])
        ).to(torch.float64)
        noise = torch.from_numpy(np.ascontiguousarray(future_noise)).to(torch.float64)
        self.validate_values(values)
        noise_sha256 = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
        return SimulatedTrajectory(
            values=values,
            times=times,
            future_noise=noise,
            future_noise_sha256=noise_sha256,
            truth=self.truth,
            world_id=self.config.world_id,
            family_id=self.config.family_id,
            regime_id=regime.regime_id,
            partition=request.partition,
            seed=request.seed,
        )

    def _simulate_upstream(
        self,
        request: SimulationRequest,
        regime: RegimeConfig,
        intervention: NodeShock | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        raise NotImplementedError

    def _intervention(
        self,
        request: SimulationRequest,
        intervention: NodeShock | None,
        times: FloatArray,
    ) -> Intervention | None:
        if intervention is None:
            return None
        if intervention.source_node >= self.config.dimension:
            raise ValueError("shock source node is outside the world")
        if intervention.step >= request.length:
            raise ValueError("shock step is outside the returned trajectory")
        full_step = request.warmup_steps + intervention.step
        shock_time = times[full_step]

        def apply(values: FloatArray, time: float) -> FloatArray:
            result = values.copy()
            if np.isclose(time, shock_time, rtol=0.0, atol=1e-12):
                result[intervention.source_node] += intervention.magnitude
            return result

        return apply


class CoupledMapWorldAdapter(ExternalWorldAdapter):
    def validate_values(self, values: torch.Tensor) -> None:
        super().validate_values(values)
        if bool((values.abs() >= 1.0).any()):
            raise TrajectoryValidationError("coupled-map boundary clipping was detected")

    def _simulate_upstream(
        self,
        request: SimulationRequest,
        regime: RegimeConfig,
        intervention: NodeShock | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        full_length = request.length + request.warmup_steps
        times = np.arange(full_length, dtype=np.float64)
        generator = self.config.generator_map()
        parameters = regime.parameter_map()
        adjacency = self.truth.adjacency.numpy().astype(np.float64)
        model = StochasticCoupledMapLattice(
            adjacency_matrix=adjacency,
            eps=parameters["eps"],
            f=quadradic_map,
            f_params=(generator["alpha"],),
            sigma=generator["sigma"],
            x_min=-1.0,
            x_max=1.0,
            boundary_condition="truncate",
            tsteps_btw_obs=int(generator["tsteps_btw_obs"]),
        )
        initial_rng = np.random.default_rng(request.seed)
        initial = initial_rng.uniform(-0.8, 0.8, size=(1, self.config.dimension))
        values = model.simulate(
            t=times,
            prior_states=initial,
            rng=np.random.RandomState(request.seed),
            intervention=self._intervention(request, intervention, times),
        )
        future_noise = np.zeros((full_length - 1, self.config.dimension), dtype=np.float64)
        return values, times, future_noise


class LotkaVolterraWorldAdapter(ExternalWorldAdapter):
    def _simulate_upstream(
        self,
        request: SimulationRequest,
        regime: RegimeConfig,
        intervention: NodeShock | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        full_length = request.length + request.warmup_steps
        generator = self.config.generator_map()
        parameters = regime.parameter_map()
        time_step = generator["time_step"]
        times = np.arange(full_length, dtype=np.float64) * time_step
        base_growth = np.linspace(
            generator["growth_min"],
            generator["growth_max"],
            self.config.dimension,
        )
        growth_rates = base_growth * parameters["growth_scale"]
        capacities = np.full(self.config.dimension, generator["capacity"], dtype=np.float64)
        interaction = self.truth.adjacency.numpy().astype(np.float64)
        interaction *= parameters["interaction_scale"]
        model = LotkaVolterraSDE(
            growth_rates=growth_rates,
            capacities=capacities,
            interaction_mat=interaction,
            sigma=parameters["sigma"],
        )
        random = np.random.default_rng(request.seed)
        initial = random.uniform(
            generator["initial_min"],
            generator["initial_max"],
            size=(1, self.config.dimension),
        )
        upstream_dt = (times[-1] - times[0]) / full_length
        future_noise = random.normal(
            loc=0.0,
            scale=np.sqrt(upstream_dt),
            size=(full_length - 1, self.config.dimension),
        )
        values = model.simulate(
            t=times,
            prior_states=initial,
            rng=np.random.RandomState(request.seed),
            dW=future_noise,
            numerical_method="EulerMaruyama",
            intervention=self._intervention(request, intervention, times),
        )
        return values, times, future_noise


class VarmaWorldAdapter(ExternalWorldAdapter):
    def _simulate_upstream(
        self,
        request: SimulationRequest,
        regime: RegimeConfig,
        intervention: NodeShock | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        full_length = request.length + request.warmup_steps
        times = np.arange(full_length, dtype=np.float64)
        generator = self.config.generator_map()
        parameters = regime.parameter_map()
        adjacency = self.truth.adjacency.numpy().astype(np.float64)
        normalized = adjacency / adjacency.sum(axis=1, keepdims=True)
        coefficient = 0.75 * np.eye(self.config.dimension) + 0.25 * normalized
        coefficient *= generator["spectral_radius"] * parameters["coefficient_scale"]
        innovation_scale = generator["innovation_scale"]
        covariance = np.eye(self.config.dimension) * innovation_scale**2
        model = VARMADynamics([coefficient], [], covariance)
        initial_rng = np.random.default_rng(request.seed)
        initial = initial_rng.normal(0.0, innovation_scale, size=(1, self.config.dimension))
        replay_rng = np.random.RandomState(request.seed)
        values = model.simulate(
            t=times,
            prior_states=initial,
            rng=replay_rng,
            intervention=self._intervention(request, intervention, times),
        )
        noise_rng = np.random.RandomState(request.seed)
        future_noise = noise_rng.multivariate_normal(
            np.zeros(self.config.dimension), covariance, full_length - 1
        )
        return values, times, future_noise


def build_world(config: WorldConfig) -> ExternalWorldAdapter:
    adapters: dict[WorldAdapter, type[ExternalWorldAdapter]] = {
        WorldAdapter.INTERFERE_CML: CoupledMapWorldAdapter,
        WorldAdapter.INTERFERE_LOTKA_VOLTERRA_SDE: LotkaVolterraWorldAdapter,
        WorldAdapter.INTERFERE_VARMA: VarmaWorldAdapter,
    }
    try:
        adapter_type = adapters[config.adapter]
    except KeyError as exc:
        raise ValueError(f"unsupported world adapter: {config.adapter}") from exc
    return adapter_type(config)
