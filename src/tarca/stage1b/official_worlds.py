from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from tarca.stage1b.config import (
    QualificationPartition,
    RegimeSplitRole,
    WorldAdapter,
    WorldConfig,
)
from tarca.stage1b.oracle import ConceptSchedule, OfficialSimulation, OfficialWorldDriver
from tarca.stage1b.reproduction import (
    _IMPORT_LOCK,
    _load_cml_module,
    _load_file_module,
    _load_module_as,
)
from tarca.stage1b.sources import (
    MaterializedSources,
    SourceVerificationError,
    verify_materialized_source,
)
from tarca.stage1b.worlds import SimulationRequest

FloatArray = NDArray[np.float64]


_OFFICIAL_DRIVER_SOURCES: dict[WorldAdapter, tuple[str, str]] = {
    WorldAdapter.TARCA_VAR: ("neural_gc", "synthetic.py"),
    WorldAdapter.LORENZ96: ("neural_gc", "synthetic.py"),
    WorldAdapter.LORENZ96_TWO_SCALE: ("scoring_rules_l96", "src/models.py"),
    WorldAdapter.GVAR_PREDATOR_PREY: (
        "gvar",
        "datasets/lotkaVolterra/multiple_lotka_volterra.py",
    ),
    WorldAdapter.CORRECTED_CML: (
        "interfere_cml",
        "interfere/dynamics/coupled_map_lattice.py",
    ),
}


def _verified_official_root(
    config: WorldConfig,
    sources: MaterializedSources,
) -> tuple[str, Path]:
    source_id, required_asset = _OFFICIAL_DRIVER_SOURCES[config.adapter]
    if source_id not in {config.source_id, *config.supporting_source_ids}:
        raise SourceVerificationError("official driver source is not authorized by the world")
    receipts = tuple(receipt for receipt in sources.receipts if receipt.source_id == source_id)
    if len(receipts) != 1:
        raise SourceVerificationError("official world source must be materialized exactly once")
    receipt = receipts[0]
    if required_asset not in dict(receipt.asset_sha256):
        raise SourceVerificationError("official world asset is absent from the verified receipt")
    cache_root = receipt.checkout_root.parent.parent
    return source_id, verify_materialized_source(receipt, cache_root)


def _regime_index(config: WorldConfig, request: SimulationRequest) -> int:
    matches = tuple(
        (index, regime)
        for index, regime in enumerate(config.regimes)
        if regime.regime_id == request.regime_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown regime for {config.world_id}: {request.regime_id}")
    index, regime = matches[0]
    expected = (
        RegimeSplitRole.UNSEEN
        if request.partition is QualificationPartition.QUAL_UNSEEN
        else RegimeSplitRole.SEEN
    )
    if regime.split_role is not expected:
        raise ValueError("qualification partition and regime split role do not match")
    return index


def _regime_parameter(config: WorldConfig, request: SimulationRequest, name: str) -> float:
    index = _regime_index(config, request)
    parameters = config.regimes[index].parameter_map()
    try:
        return parameters[name]
    except KeyError as error:
        raise ValueError(f"regime does not register parameter {name}") from error


def _as_noise(value: Tensor, expected_shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or tuple(value.shape) != expected_shape:
        raise ValueError(f"future noise must have shape {expected_shape}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("future noise must be a finite floating tensor")
    return value.detach().cpu().to(torch.float64).contiguous().clone()


def _same_noise(value: Tensor, expected: Tensor) -> Tensor:
    normalized = _as_noise(value, tuple(expected.shape))
    if not torch.equal(normalized, expected):
        raise ValueError("future noise does not match the generator-owned seeded draw")
    return normalized


def _schedule_arrays(schedule: ConceptSchedule, length: int) -> tuple[FloatArray, FloatArray]:
    if tuple(schedule.trend.shape) != (length,) or tuple(schedule.scale.shape) != (length,):
        raise ValueError("concept schedule length must match official simulation request")
    trend = schedule.trend.detach().cpu().to(torch.float64).numpy().copy()
    scale = schedule.scale.detach().cpu().to(torch.float64).numpy().copy()
    return trend, scale


def _official_simulation(
    *,
    config: WorldConfig,
    request: SimulationRequest,
    values: FloatArray,
    times: FloatArray,
    initial_state: FloatArray,
    future_noise: Tensor,
    boundary_event_count: int = 0,
) -> OfficialSimulation:
    regime_index = _regime_index(config, request)
    return OfficialSimulation(
        values=torch.from_numpy(np.ascontiguousarray(values)).to(torch.float64),
        times=torch.from_numpy(np.ascontiguousarray(times)).to(torch.float64),
        initial_state=torch.from_numpy(np.ascontiguousarray(initial_state)).to(torch.float64),
        future_noise=future_noise,
        regime_sequence=torch.full((request.length,), regime_index, dtype=torch.int64),
        boundary_event_count=boundary_event_count,
    )


class _BaseOfficialDriver:
    source_id: str
    formal_origin = "PINNED_OFFICIAL_SOURCE"

    def __init__(self, config: WorldConfig, source_id: str, source_root: Path) -> None:
        self.config = config
        self.source_id = source_id
        self.source_root = source_root


class NeuralGcVarDriver(_BaseOfficialDriver):
    def _components(self, request: SimulationRequest) -> tuple[FloatArray, FloatArray]:
        generator = self.config.generator_map()
        dimension = self.config.dimension
        rng = np.random.RandomState(request.seed)
        beta = np.eye(dimension, dtype=np.float64)
        nonzero = int(dimension * generator["sparsity"]) - 1
        for index in range(dimension):
            choices = rng.choice(dimension - 1, size=nonzero, replace=False)
            choices[choices >= index] += 1
            beta[index, choices] = 1.0
        module = _load_file_module(self.source_root / "synthetic.py", "official_neural_gc_var")
        stationary = cast(Callable[..., object], module.make_var_stationary)
        beta = np.asarray(
            stationary(beta, radius=generator["spectral_radius"]),
            dtype=np.float64,
        )
        total = 100 + request.warmup_steps + request.length
        errors = rng.normal(size=(dimension, total)).astype(np.float64).T
        return beta, errors

    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        _, errors = self._components(request)
        return torch.from_numpy(errors.copy()).to(torch.float64)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        trend, scale = _schedule_arrays(schedule, request.length)
        beta, expected_errors = self._components(request)
        noise = _same_noise(future_noise, torch.from_numpy(expected_errors).to(torch.float64))
        generator = self.config.generator_map()
        burn = 100 + request.warmup_steps
        state = generator["innovation_scale"] * expected_errors[0]
        for index in range(1, burn):
            state = beta @ state + generator["innovation_scale"] * expected_errors[index - 1]
        initial_state = state.copy()
        values = np.empty((request.length, self.config.dimension), dtype=np.float64)
        for returned in range(request.length):
            state = (
                trend[returned] * (beta @ state)
                + scale[returned] * expected_errors[burn + returned - 1]
            )
            values[returned] = state
        return _official_simulation(
            config=self.config,
            request=request,
            values=values,
            times=np.arange(request.length, dtype=np.float64),
            initial_state=initial_state,
            future_noise=noise,
        )


class NeuralGcLorenz96Driver(_BaseOfficialDriver):
    def _seeded_noise(self, request: SimulationRequest) -> tuple[FloatArray, FloatArray]:
        generator = self.config.generator_map()
        rng = np.random.RandomState(request.seed)
        initial = rng.normal(
            scale=generator["initial_perturbation"],
            size=self.config.dimension,
        ).astype(np.float64)
        burn = int(generator["burn_in_observations"]) + request.warmup_steps
        all_noise = rng.normal(size=(burn + request.length, self.config.dimension)).astype(
            np.float64
        )
        return initial, all_noise[-request.length :]

    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        _, noise = self._seeded_noise(request)
        return torch.from_numpy(noise.copy()).to(torch.float64)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        trend, scale = _schedule_arrays(schedule, request.length)
        initial, expected_noise = self._seeded_noise(request)
        noise = _same_noise(future_noise, torch.from_numpy(expected_noise).to(torch.float64))
        generator = self.config.generator_map()
        interval = generator["observation_interval"]
        burn = int(generator["burn_in_observations"]) + request.warmup_steps
        module = _load_file_module(self.source_root / "synthetic.py", "official_neural_gc_l96")
        tendency = cast(Callable[..., object], module.lorenz)
        integrate = cast(Callable[..., object], module.odeint)
        state = initial.copy()
        if burn:
            state = np.asarray(
                integrate(
                    tendency,
                    state,
                    np.linspace(0.0, burn * interval, burn + 1, dtype=np.float64),
                    args=(generator["forcing"],),
                ),
                dtype=np.float64,
            )[-1]
        shared_initial = state.copy()
        values = np.empty((request.length, self.config.dimension), dtype=np.float64)
        for index in range(request.length):
            values[index] = state + scale[index] * expected_noise[index]
            state = np.asarray(
                integrate(
                    tendency,
                    state,
                    np.asarray([0.0, interval], dtype=np.float64),
                    args=(trend[index],),
                ),
                dtype=np.float64,
            )[-1]
        return _official_simulation(
            config=self.config,
            request=request,
            values=values,
            times=np.arange(request.length, dtype=np.float64) * interval,
            initial_state=shared_initial,
            future_noise=noise,
        )


@contextmanager
def _jmlr_module(source_root: Path) -> Iterator[ModuleType]:
    import numba  # type: ignore[import-untyped]

    with _IMPORT_LOCK, TemporaryDirectory(prefix="tarca-stage1b-world-numba-") as cache:
        previous_cache = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = cache
        identity = hashlib.sha256(cache.encode("utf-8")).hexdigest()[:16]
        module_name = f"_tarca_stage1b_world_jmlr_{identity}"
        try:
            yield _load_module_as(source_root / "src/models.py", module_name)
        finally:
            numba.config.CACHE_DIR = previous_cache
            sys.modules.pop(module_name, None)


def _two_scale_step(
    derivative: Callable[..., tuple[object, object]],
    slow: FloatArray,
    fast: FloatArray,
    step: float,
    *,
    forcing: float,
    coupling: float,
    b: float,
    c: float,
) -> tuple[FloatArray, FloatArray]:
    first_slow, first_fast = derivative(slow, fast, coupling, forcing, b, c)
    first_slow = np.asarray(first_slow, dtype=np.float64)
    first_fast = np.asarray(first_fast, dtype=np.float64)
    second_slow, second_fast = derivative(
        slow + first_slow * step / 2,
        fast + first_fast * step / 2,
        coupling,
        forcing,
        b,
        c,
    )
    second_slow = np.asarray(second_slow, dtype=np.float64)
    second_fast = np.asarray(second_fast, dtype=np.float64)
    third_slow, third_fast = derivative(
        slow + second_slow * step / 2,
        fast + second_fast * step / 2,
        coupling,
        forcing,
        b,
        c,
    )
    third_slow = np.asarray(third_slow, dtype=np.float64)
    third_fast = np.asarray(third_fast, dtype=np.float64)
    fourth_slow, fourth_fast = derivative(
        slow + third_slow * step,
        fast + third_fast * step,
        coupling,
        forcing,
        b,
        c,
    )
    fourth_slow = np.asarray(fourth_slow, dtype=np.float64)
    fourth_fast = np.asarray(fourth_fast, dtype=np.float64)
    next_slow = slow + step * (first_slow + 2 * second_slow + 2 * third_slow + fourth_slow) / 6
    next_fast = fast + step * (first_fast + 2 * second_fast + 2 * third_fast + fourth_fast) / 6
    return next_slow, next_fast


class JmlrTwoScaleLorenz96Driver(_BaseOfficialDriver):
    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        return torch.zeros((request.length, self.config.dimension), dtype=torch.float64)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        trend, scale = _schedule_arrays(schedule, request.length)
        expected_noise = self.sample_future_noise(request)
        noise = _same_noise(future_noise, expected_noise)
        generator = self.config.generator_map()
        slow = np.zeros(self.config.dimension, dtype=np.float64)
        fast = np.zeros(self.config.latent_dimension, dtype=np.float64)
        slow[0] = 1.0
        fast[0] = 1.0
        step = generator["integration_step"]
        internal_steps = round(generator["observation_interval"] / step)
        burn = (int(generator["burn_in_observations"]) + request.warmup_steps) * internal_steps
        values = np.empty((request.length, self.config.dimension), dtype=np.float64)
        with _jmlr_module(self.source_root) as module:
            derivative = cast(Callable[..., tuple[object, object]], module.l96_truth_step)
            for _ in range(burn):
                slow, fast = _two_scale_step(
                    derivative,
                    slow,
                    fast,
                    step,
                    forcing=generator["forcing"],
                    coupling=generator["coupling_h"],
                    b=generator["fast_amplitude_b"],
                    c=generator["time_scale_c"],
                )
            shared_initial = slow.copy()
            for observation in range(request.length):
                values[observation] = slow
                for _ in range(internal_steps):
                    slow, fast = _two_scale_step(
                        derivative,
                        slow,
                        fast,
                        step,
                        forcing=trend[observation],
                        coupling=scale[observation],
                        b=generator["fast_amplitude_b"],
                        c=generator["time_scale_c"],
                    )
        return _official_simulation(
            config=self.config,
            request=request,
            values=values,
            times=np.arange(request.length, dtype=np.float64) * generator["observation_interval"],
            initial_state=shared_initial,
            future_noise=noise,
        )


class GvarPredatorPreyDriver(_BaseOfficialDriver):
    def _initial_and_noise(self, request: SimulationRequest) -> tuple[FloatArray, FloatArray]:
        generator = self.config.generator_map()
        rng = np.random.RandomState(request.seed)
        species = self.config.dimension // 2
        prey = rng.uniform(10.0, 100.0, size=species)
        predator = rng.uniform(10.0, 100.0, size=species)
        internal_steps = round(generator["observation_interval"] / generator["integration_step"])
        count = (request.warmup_steps + request.length) * internal_steps
        rows = [
            np.concatenate((rng.normal(size=species), rng.normal(size=species)))
            for _ in range(count)
        ]
        return np.concatenate((prey, predator)).astype(np.float64), np.asarray(
            rows, dtype=np.float64
        )

    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        _, noise = self._initial_and_noise(request)
        return torch.from_numpy(noise.copy()).to(torch.float64)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        trend, scale = _schedule_arrays(schedule, request.length)
        state, expected_noise = self._initial_and_noise(request)
        noise = _same_noise(future_noise, torch.from_numpy(expected_noise).to(torch.float64))
        generator = self.config.generator_map()
        module = _load_file_module(
            self.source_root / "datasets/lotkaVolterra/multiple_lotka_volterra.py",
            "official_gvar_predator_prey",
        )
        model_type = cast(Callable[..., Any], module.MultiLotkaVolterra)
        step = generator["integration_step"]
        internal_steps = round(generator["observation_interval"] / step)
        noise_index = 0
        clipped = 0

        def derivative_for(
            alpha: float,
        ) -> Callable[[FloatArray, FloatArray], tuple[object, object]]:
            model = model_type(
                p=self.config.dimension // 2,
                d=int(generator["parents_per_node"]),
                alpha=alpha,
                beta=generator["beta"],
                gamma=generator["gamma"],
                delta=generator["delta"],
                sigma=0.0,
            )
            return cast(
                Callable[[FloatArray, FloatArray], tuple[object, object]],
                model.f,
            )

        def advance(
            derivative: Callable[[FloatArray, FloatArray], tuple[object, object]],
            sigma: float,
        ) -> None:
            nonlocal state, noise_index, clipped
            prey = state[: self.config.dimension // 2]
            predator = state[self.config.dimension // 2 :]
            first_prey, first_predator = derivative(prey, predator)
            first = np.concatenate(
                (
                    np.asarray(first_prey, dtype=np.float64),
                    np.asarray(first_predator, dtype=np.float64),
                )
            )
            second_prey, second_predator = derivative(
                prey + first[: prey.size] * step / 2,
                predator + first[prey.size :] * step / 2,
            )
            second = np.concatenate(
                (
                    np.asarray(second_prey, dtype=np.float64),
                    np.asarray(second_predator, dtype=np.float64),
                )
            )
            third_prey, third_predator = derivative(
                prey + second[: prey.size] * step / 2,
                predator + second[prey.size :] * step / 2,
            )
            third = np.concatenate(
                (
                    np.asarray(third_prey, dtype=np.float64),
                    np.asarray(third_predator, dtype=np.float64),
                )
            )
            fourth_prey, fourth_predator = derivative(
                prey + third[: prey.size] * step,
                predator + third[prey.size :] * step,
            )
            fourth = np.concatenate(
                (
                    np.asarray(fourth_prey, dtype=np.float64),
                    np.asarray(fourth_predator, dtype=np.float64),
                )
            )
            proposed = state + step * (first + 2 * second + 2 * third + fourth) / 6
            proposed += sigma * expected_noise[noise_index]
            clipped += int(np.count_nonzero(proposed < 0.0))
            state = np.maximum(proposed, 0.0)
            noise_index += 1

        warmup_derivative = derivative_for(generator["alpha"])
        for _ in range(request.warmup_steps * internal_steps):
            advance(warmup_derivative, generator["sigma"])
        shared_initial = state.copy()
        values = np.empty((request.length, self.config.dimension), dtype=np.float64)
        for observation in range(request.length):
            values[observation] = state
            derivative = derivative_for(trend[observation])
            for _ in range(internal_steps):
                advance(derivative, scale[observation])
        return _official_simulation(
            config=self.config,
            request=request,
            values=values,
            times=np.arange(request.length, dtype=np.float64) * generator["observation_interval"],
            initial_state=shared_initial,
            future_noise=noise,
            boundary_event_count=clipped,
        )


class InterfereCmlDriver(_BaseOfficialDriver):
    def _initial_and_noise(self, request: SimulationRequest) -> tuple[FloatArray, FloatArray]:
        rng = np.random.RandomState(request.seed)
        initial = rng.uniform(-0.8, 0.8, size=self.config.dimension).astype(np.float64)
        noise = rng.normal(
            size=(request.warmup_steps + request.length, self.config.dimension)
        ).astype(np.float64)
        return initial, noise

    def sample_future_noise(self, request: SimulationRequest) -> Tensor:
        _, noise = self._initial_and_noise(request)
        return torch.from_numpy(noise.copy()).to(torch.float64)

    def simulate(
        self,
        request: SimulationRequest,
        schedule: ConceptSchedule,
        future_noise: Tensor,
    ) -> OfficialSimulation:
        trend, scale = _schedule_arrays(schedule, request.length)
        state, expected_noise = self._initial_and_noise(request)
        noise = _same_noise(future_noise, torch.from_numpy(expected_noise).to(torch.float64))
        generator = self.config.generator_map()
        module = _load_cml_module(self.source_root)
        model_type = cast(Callable[..., Any], module.CoupledMapLattice)
        map_function = cast(Callable[..., object], module.quadradic_map)
        adjacency = np.roll(np.eye(self.config.dimension), 1, axis=1) + np.roll(
            np.eye(self.config.dimension), -1, axis=1
        )

        def advance(alpha: float, epsilon: float, noise_row: FloatArray) -> None:
            nonlocal state
            model = model_type(
                adjacency,
                eps=epsilon,
                f=map_function,
                f_params=(alpha,),
            )
            stepped = model.step(state.copy(), 0.0, np.random.RandomState(0))
            state = np.asarray(stepped, dtype=np.float64) + generator["sigma"] * noise_row

        base_epsilon = _regime_parameter(self.config, request, "epsilon")
        for index in range(request.warmup_steps):
            advance(generator["alpha"], base_epsilon, expected_noise[index])
        shared_initial = state.copy()
        values = np.empty((request.length, self.config.dimension), dtype=np.float64)
        for observation in range(request.length):
            values[observation] = state
            advance(
                trend[observation],
                scale[observation],
                expected_noise[request.warmup_steps + observation],
            )
        return _official_simulation(
            config=self.config,
            request=request,
            values=values,
            times=np.arange(request.length, dtype=np.float64),
            initial_state=shared_initial,
            future_noise=noise,
        )


def baseline_concept_schedule(
    config: WorldConfig,
    request: SimulationRequest,
) -> ConceptSchedule:
    generator = config.generator_map()
    if config.adapter is WorldAdapter.TARCA_VAR:
        trend, scale = 1.0, generator["innovation_scale"]
    elif config.adapter is WorldAdapter.LORENZ96:
        trend = generator["forcing"]
        scale = _regime_parameter(config, request, "measurement_noise")
    elif config.adapter is WorldAdapter.LORENZ96_TWO_SCALE:
        trend, scale = generator["forcing"], generator["coupling_h"]
    elif config.adapter is WorldAdapter.GVAR_PREDATOR_PREY:
        trend = generator["alpha"]
        scale = _regime_parameter(config, request, "dynamic_noise_scale")
    elif config.adapter is WorldAdapter.CORRECTED_CML:
        trend = generator["alpha"]
        scale = _regime_parameter(config, request, "epsilon")
    else:
        raise ValueError("world adapter has no registered concept schedule")
    return ConceptSchedule(
        trend=torch.full((request.length,), trend, dtype=torch.float64),
        scale=torch.full((request.length,), scale, dtype=torch.float64),
    )


def concept_pair_schedules(
    config: WorldConfig,
    request: SimulationRequest,
    pair_id: str,
) -> tuple[ConceptSchedule, ConceptSchedule, Literal["trend", "scale"]]:
    matches = tuple(pair for pair in config.concept_pairs if pair.pair_id == pair_id)
    if len(matches) != 1:
        raise ValueError(f"world does not register concept pair {pair_id}")
    pair = matches[0]
    baseline = baseline_concept_schedule(config, request)
    factual_trend = baseline.trend.clone()
    counterfactual_trend = baseline.trend.clone()
    factual_scale = baseline.scale.clone()
    counterfactual_scale = baseline.scale.clone()
    if pair.concept == "trend":
        factual_trend = torch.full_like(baseline.trend, pair.factual_value)
        counterfactual_trend = torch.full_like(baseline.trend, pair.counterfactual_value)
    else:
        factual_scale = torch.full_like(baseline.scale, pair.factual_value)
        counterfactual_scale = torch.full_like(baseline.scale, pair.counterfactual_value)
    return (
        ConceptSchedule(trend=factual_trend, scale=factual_scale),
        ConceptSchedule(trend=counterfactual_trend, scale=counterfactual_scale),
        pair.concept,
    )


def build_official_world(
    config: WorldConfig,
    sources: MaterializedSources,
) -> OfficialWorldDriver:
    source_id, source_root = _verified_official_root(config, sources)
    drivers: dict[WorldAdapter, type[_BaseOfficialDriver]] = {
        WorldAdapter.TARCA_VAR: NeuralGcVarDriver,
        WorldAdapter.LORENZ96: NeuralGcLorenz96Driver,
        WorldAdapter.LORENZ96_TWO_SCALE: JmlrTwoScaleLorenz96Driver,
        WorldAdapter.GVAR_PREDATOR_PREY: GvarPredatorPreyDriver,
        WorldAdapter.CORRECTED_CML: InterfereCmlDriver,
    }
    return cast(OfficialWorldDriver, drivers[config.adapter](config, source_id, source_root))
