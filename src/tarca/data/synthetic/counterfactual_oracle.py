"""Paired counterfactual replay over explicit future stochastic inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tarca.contracts import InterventionPair

from .latent_concepts import (
    LatentConceptPath,
    generate_latent_concepts,
    replace_concept_at_origin,
)
from .nonlinear_var import RegimeDynamics, SyntheticTrajectory, rollout_nonlinear_var


@dataclass(frozen=True, slots=True)
class FutureNoiseBank:
    """Read-only ``[H]``/``[H,U]``/``[H,D]`` replay sample with fixed regimes."""

    regime_uniforms: NDArray[np.float64] | None
    regime_path: NDArray[np.int64]
    trend_innovations: NDArray[np.float64]
    scale_innovations: NDArray[np.float64]
    exogenous_inputs: NDArray[np.float64]
    observation_innovations: NDArray[np.float64]
    shocks: NDArray[np.float64]

    def __post_init__(self) -> None:
        trend = _copy_float64_vector(
            self.trend_innovations,
            field_name="trend_innovations",
        )
        horizon = trend.size
        scale = _copy_float64_vector(
            self.scale_innovations,
            field_name="scale_innovations",
            shape=(horizon,),
        )
        exogenous = _copy_float64_matrix(
            self.exogenous_inputs,
            field_name="exogenous_inputs",
            rows=horizon,
            allow_zero_columns=True,
        )
        observation = _copy_float64_matrix(
            self.observation_innovations,
            field_name="observation_innovations",
            rows=horizon,
        )
        shocks = _copy_float64_matrix(
            self.shocks,
            field_name="shocks",
            shape=observation.shape,
        )
        uniforms = _optional_regime_uniforms(self.regime_uniforms, horizon=horizon)
        path = _optional_regime_path(self.regime_path, horizon=horizon)
        if path is None:
            raise ValueError("regime_path: required for deterministic replay")
        values = (uniforms, path, trend, scale, exogenous, observation, shocks)
        for field, value in zip(fields(self), values, strict=True):
            object.__setattr__(self, field.name, value)

    @property
    def horizon(self) -> int:
        return int(self.trend_innovations.shape[0])


@dataclass(frozen=True, slots=True)
class CounterfactualIntervention:
    """Replace one current latent concept with one explicit source value."""

    concept: Literal["trend", "scale"]
    source_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.concept, str) or self.concept not in ("trend", "scale"):
            raise ValueError(
                f"intervention.concept: expected 'trend' or 'scale', got {self.concept!r}"
            )
        source = _finite_real(self.source_value, field_name="intervention.source_value")
        object.__setattr__(self, "source_value", source)


@dataclass(frozen=True, slots=True)
class PairedCounterfactualResult:
    """One factual/counterfactual replay pair under a shared noise bank."""

    factual_path: SyntheticTrajectory
    counterfactual_path: SyntheticTrajectory
    factual_concepts: LatentConceptPath
    counterfactual_concepts: LatentConceptPath
    noise_bank: FutureNoiseBank
    intervention: CounterfactualIntervention | None
    effect: NDArray[np.float64]
    horizon_index: NDArray[np.int64]
    causal_delay: int
    allocation_metadata: InterventionPair | None

    def __post_init__(self) -> None:
        factual_path = _trajectory(self.factual_path, field_name="factual_path")
        counterfactual_path = _trajectory(
            self.counterfactual_path, field_name="counterfactual_path"
        )
        factual_concepts = _copy_latent_path(self.factual_concepts, field_name="factual_concepts")
        counterfactual_concepts = _copy_latent_path(
            self.counterfactual_concepts, field_name="counterfactual_concepts"
        )
        bank = _copy_noise_bank(self.noise_bank)
        intervention = _copy_intervention(self.intervention)
        _validate_paired_replay_inputs(
            factual_path=factual_path,
            counterfactual_path=counterfactual_path,
            factual_concepts=factual_concepts,
            counterfactual_concepts=counterfactual_concepts,
            bank=bank,
            intervention=intervention,
        )
        expected_effect = np.asarray(
            counterfactual_path.full_values - factual_path.full_values,
            dtype=np.float64,
        )
        effect = _copy_float64_matrix(
            self.effect,
            field_name="effect",
            shape=expected_effect.shape,
        )
        if effect.tobytes() != expected_effect.tobytes():
            raise ValueError("effect: expected counterfactual_path - factual_path")
        horizon_index = _copy_horizon_index(self.horizon_index, horizon=bank.horizon)
        causal_delay = _causal_delay(self.causal_delay, horizon=bank.horizon)
        schedule_delay = factual_path.dynamics_schedule[0].trend_delay
        if causal_delay != schedule_delay:
            raise ValueError("causal_delay: expected the first affected schedule horizon")
        allocation = _copy_allocation(
            self.allocation_metadata,
            intervention=intervention,
            factual_concepts=factual_concepts,
        )
        values = (
            factual_path,
            counterfactual_path,
            factual_concepts,
            counterfactual_concepts,
            bank,
            intervention,
            effect,
            horizon_index,
            causal_delay,
            allocation,
        )
        for field, value in zip(fields(self), values, strict=True):
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True, slots=True)
class MonteCarloOracleResult:
    """Per-sample paths/effects and aggregate paired distribution effects."""

    paired_results: tuple[PairedCounterfactualResult, ...]
    factual_paths: NDArray[np.float64]
    counterfactual_paths: NDArray[np.float64]
    sample_effects: NDArray[np.float64]
    mean_effect: NDArray[np.float64]
    std_effect: NDArray[np.float64]
    quantiles: NDArray[np.float64]
    factual_quantiles: NDArray[np.float64]
    counterfactual_quantiles: NDArray[np.float64]
    quantile_effects: NDArray[np.float64]
    horizon_index: NDArray[np.int64]
    causal_delay: int
    intervention: CounterfactualIntervention | None
    allocation_metadata: InterventionPair | None

    def __post_init__(self) -> None:
        pairs = _pair_collection(self.paired_results)
        levels = _quantile_levels(self.quantiles)
        expected = _aggregate_pairs(pairs, levels)
        arrays = {
            name: _copy_expected_array(getattr(self, name), name=name, expected=value)
            for name, value in expected.items()
        }
        first = pairs[0]
        horizon = _copy_horizon_index(self.horizon_index, horizon=first.effect.shape[0])
        delay = _causal_delay(self.causal_delay, horizon=first.effect.shape[0])
        if delay != first.causal_delay:
            raise ValueError("causal_delay: expected every paired result to match")
        intervention = _copy_intervention(self.intervention)
        if intervention != first.intervention:
            raise ValueError("intervention: expected every paired result to match")
        allocation = _copy_allocation(
            self.allocation_metadata,
            intervention=intervention,
            factual_concepts=first.factual_concepts,
        )
        if allocation != first.allocation_metadata:
            raise ValueError("allocation_metadata: expected every paired result to match")
        object.__setattr__(self, "paired_results", pairs)
        object.__setattr__(self, "quantiles", levels)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "horizon_index", horizon)
        object.__setattr__(self, "causal_delay", delay)
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "allocation_metadata", allocation)


def replay_paired_counterfactual(
    *,
    initial_history: NDArray[np.float64],
    trend_history: NDArray[np.float64],
    current_trend: float,
    current_scale: float,
    trend_ar_coefficients: NDArray[np.float64],
    scale_ar_coefficients: NDArray[np.float64],
    dynamics_schedule: tuple[RegimeDynamics, ...],
    noise_bank: FutureNoiseBank,
    intervention: CounterfactualIntervention | None,
    trend_loading: NDArray[np.float64],
    observation_scale_floor: float,
    causal_delay: int,
    allocation_metadata: InterventionPair | None = None,
) -> PairedCounterfactualResult:
    """Replay one pair; Task 4 consumes H current states from H+1 latent states."""
    bank = _copy_noise_bank(noise_bank)
    schedule, regime_labels = _validate_schedule(dynamics_schedule, bank=bank)
    validated_intervention = _copy_intervention(intervention)
    factual_concepts = generate_latent_concepts(
        regime_sequence=regime_labels,
        trend_ar_coefficients=trend_ar_coefficients,
        scale_ar_coefficients=scale_ar_coefficients,
        trend_innovations=bank.trend_innovations,
        scale_innovations=bank.scale_innovations,
        initial_trend=current_trend,
        initial_scale=current_scale,
    )
    counterfactual_concepts = factual_concepts
    if validated_intervention is not None:
        counterfactual_concepts = replace_concept_at_origin(
            factual_concepts,
            concept=validated_intervention.concept,
            origin_index=0,
            source_value=validated_intervention.source_value,
        )
    common_rollout = {
        "initial_history": initial_history,
        "trend_history": trend_history,
        "dynamics_schedule": schedule,
        "regime_labels": regime_labels,
        "exogenous_inputs": bank.exogenous_inputs,
        "observation_innovations": bank.observation_innovations,
        "shocks": bank.shocks,
        "trend_loading": trend_loading,
        "observation_scale_floor": observation_scale_floor,
    }
    factual_path = rollout_nonlinear_var(
        trend_path=factual_concepts.trend[:-1],
        scale_path=factual_concepts.scale[:-1],
        **common_rollout,
    )
    counterfactual_path = rollout_nonlinear_var(
        trend_path=counterfactual_concepts.trend[:-1],
        scale_path=counterfactual_concepts.scale[:-1],
        **common_rollout,
    )
    effect = np.asarray(counterfactual_path.full_values - factual_path.full_values)
    return PairedCounterfactualResult(
        factual_path=factual_path,
        counterfactual_path=counterfactual_path,
        factual_concepts=factual_concepts,
        counterfactual_concepts=counterfactual_concepts,
        noise_bank=bank,
        intervention=validated_intervention,
        effect=effect,
        horizon_index=np.arange(1, bank.horizon + 1, dtype=np.int64),
        causal_delay=causal_delay,
        allocation_metadata=allocation_metadata,
    )


def monte_carlo_oracle(
    *,
    initial_history: NDArray[np.float64],
    trend_history: NDArray[np.float64],
    current_trend: float,
    current_scale: float,
    trend_ar_coefficients: NDArray[np.float64],
    scale_ar_coefficients: NDArray[np.float64],
    dynamics_schedule: tuple[RegimeDynamics, ...],
    noise_banks: tuple[FutureNoiseBank, ...],
    intervention: CounterfactualIntervention | None,
    trend_loading: NDArray[np.float64],
    observation_scale_floor: float,
    causal_delay: int,
    quantiles: NDArray[np.float64],
    allocation_metadata: InterventionPair | None = None,
) -> MonteCarloOracleResult:
    """Aggregate pre-generated paired banks over the sample axis without RNG."""
    if not isinstance(noise_banks, tuple) or not noise_banks:
        raise ValueError("noise_banks: expected a non-empty tuple")
    levels = _quantile_levels(quantiles)
    pairs = tuple(
        replay_paired_counterfactual(
            initial_history=initial_history,
            trend_history=trend_history,
            current_trend=current_trend,
            current_scale=current_scale,
            trend_ar_coefficients=trend_ar_coefficients,
            scale_ar_coefficients=scale_ar_coefficients,
            dynamics_schedule=dynamics_schedule,
            noise_bank=bank,
            intervention=intervention,
            trend_loading=trend_loading,
            observation_scale_floor=observation_scale_floor,
            causal_delay=causal_delay,
            allocation_metadata=allocation_metadata,
        )
        for bank in noise_banks
    )
    aggregate = _aggregate_pairs(pairs, levels)
    return MonteCarloOracleResult(
        paired_results=pairs,
        quantiles=levels,
        horizon_index=pairs[0].horizon_index,
        causal_delay=causal_delay,
        intervention=intervention,
        allocation_metadata=allocation_metadata,
        **aggregate,
    )


def estimate_effect_delay(effect: NDArray[np.float64]) -> int:
    """Return zero-based peak horizon (``h_peak - 1``) by Euclidean magnitude."""
    supplied = _require_float64_array(effect, field_name="effect")
    if supplied.ndim < 1 or supplied.shape[0] == 0:
        raise ValueError(f"effect: expected non-empty shape [H, ...], got {supplied.shape}")
    flattened = supplied.reshape(supplied.shape[0], -1)
    magnitude = np.linalg.norm(flattened, axis=1)
    return int(np.argmax(magnitude))


def _pair_collection(value: object) -> tuple[PairedCounterfactualResult, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("paired_results: expected a non-empty tuple")
    pairs: list[PairedCounterfactualResult] = []
    for item in value:
        if not isinstance(item, PairedCounterfactualResult):
            raise TypeError("paired_results: every item must be PairedCounterfactualResult")
        try:
            pairs.append(
                PairedCounterfactualResult(
                    **{field.name: getattr(item, field.name) for field in fields(item)}
                )
            )
        except AttributeError as error:
            raise TypeError(f"paired_results: malformed item; missing {error.name}") from None
    first = pairs[0]
    for pair in pairs[1:]:
        if (
            pair.effect.shape != first.effect.shape
            or pair.horizon_index.tobytes() != first.horizon_index.tobytes()
            or pair.causal_delay != first.causal_delay
            or pair.intervention != first.intervention
            or pair.allocation_metadata != first.allocation_metadata
        ):
            raise ValueError("paired_results: every sample must share shape and metadata")
    return tuple(pairs)


def _quantile_levels(value: object) -> NDArray[np.float64]:
    levels = _copy_float64_vector(value, field_name="quantiles")
    if np.any(levels <= 0.0) or np.any(levels >= 1.0):
        raise ValueError("quantiles: every level must lie strictly in (0, 1)")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantiles: levels must be unique and strictly increasing")
    return levels


def _aggregate_pairs(
    pairs: tuple[PairedCounterfactualResult, ...],
    levels: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    factual = np.stack([pair.factual_path.full_values for pair in pairs])
    counterfactual = np.stack([pair.counterfactual_path.full_values for pair in pairs])
    effects = np.stack([pair.effect for pair in pairs])
    factual_quantiles = np.quantile(factual, levels, axis=0)
    counterfactual_quantiles = np.quantile(counterfactual, levels, axis=0)
    return {
        "factual_paths": factual,
        "counterfactual_paths": counterfactual,
        "sample_effects": effects,
        "mean_effect": np.mean(effects, axis=0),
        "std_effect": np.std(counterfactual, axis=0) - np.std(factual, axis=0),
        "factual_quantiles": factual_quantiles,
        "counterfactual_quantiles": counterfactual_quantiles,
        "quantile_effects": counterfactual_quantiles - factual_quantiles,
    }


def _copy_expected_array(
    value: object,
    *,
    name: str,
    expected: NDArray[np.float64],
) -> NDArray[np.float64]:
    array = _require_float64_array(value, field_name=name)
    if array.shape != expected.shape or array.tobytes() != expected.tobytes():
        raise ValueError(f"{name}: values do not match paired_results")
    return _read_only_copy(array)


def _copy_noise_bank(value: object) -> FutureNoiseBank:
    if not isinstance(value, FutureNoiseBank):
        raise TypeError("noise_bank: expected a FutureNoiseBank")
    try:
        return FutureNoiseBank(
            regime_uniforms=value.regime_uniforms,
            regime_path=value.regime_path,
            trend_innovations=value.trend_innovations,
            scale_innovations=value.scale_innovations,
            exogenous_inputs=value.exogenous_inputs,
            observation_innovations=value.observation_innovations,
            shocks=value.shocks,
        )
    except AttributeError as error:
        raise TypeError(f"noise_bank: malformed FutureNoiseBank; missing {error.name}") from None


def _copy_intervention(value: object) -> CounterfactualIntervention | None:
    if value is None:
        return None
    if not isinstance(value, CounterfactualIntervention):
        raise TypeError("intervention: expected CounterfactualIntervention or None")
    try:
        return CounterfactualIntervention(value.concept, value.source_value)
    except AttributeError as error:
        raise TypeError(
            f"intervention: malformed CounterfactualIntervention; missing {error.name}"
        ) from None


def _copy_latent_path(value: object, *, field_name: str) -> LatentConceptPath:
    if not isinstance(value, LatentConceptPath):
        raise TypeError(f"{field_name}: expected a LatentConceptPath")
    try:
        return LatentConceptPath(
            trend=value.trend,
            scale=value.scale,
            trend_innovations=value.trend_innovations,
            scale_innovations=value.scale_innovations,
            regime_sequence=value.regime_sequence,
            trend_ar_coefficients=value.trend_ar_coefficients,
            scale_ar_coefficients=value.scale_ar_coefficients,
            initial_trend=value.initial_trend,
            initial_scale=value.initial_scale,
            intervention=value.intervention,
        )
    except AttributeError as error:
        raise TypeError(f"{field_name}: malformed path; missing {error.name}") from None


def _trajectory(value: object, *, field_name: str) -> SyntheticTrajectory:
    if not isinstance(value, SyntheticTrajectory):
        raise TypeError(f"{field_name}: expected a SyntheticTrajectory")
    try:
        return SyntheticTrajectory(
            **{field.name: getattr(value, field.name) for field in fields(value)}
        )
    except AttributeError as error:
        raise TypeError(f"{field_name}: malformed trajectory; missing {error.name}") from None


def _validate_schedule(
    value: object,
    *,
    bank: FutureNoiseBank,
) -> tuple[tuple[RegimeDynamics, ...], NDArray[np.int64]]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("dynamics_schedule: expected a non-empty tuple")
    if len(value) != bank.horizon:
        raise ValueError(
            "dynamics_schedule: expected one item per bank horizon "
            f"({bank.horizon}), got {len(value)}"
        )
    if any(not isinstance(item, RegimeDynamics) for item in value):
        raise TypeError("dynamics_schedule: every item must be RegimeDynamics")
    schedule = value
    dimension = bank.observation_innovations.shape[1]
    exogenous_dimension = bank.exogenous_inputs.shape[1]
    for step, dynamics in enumerate(schedule):
        if dynamics.linear_matrix.shape != (dimension, dimension):
            raise ValueError(
                f"dynamics_schedule[{step}]: expected observation dimension {dimension}"
            )
        if dynamics.exogenous_matrix.shape != (dimension, exogenous_dimension):
            raise ValueError(
                f"dynamics_schedule[{step}]: expected exogenous dimension {exogenous_dimension}"
            )
    labels = np.array([item.regime_label for item in schedule], dtype=np.int64)
    if bank.regime_path is not None and not np.array_equal(bank.regime_path, labels):
        raise ValueError("regime_path: labels must match dynamics_schedule")
    return schedule, _read_only_copy(labels)


def _validate_paired_replay_inputs(
    *,
    factual_path: SyntheticTrajectory,
    counterfactual_path: SyntheticTrajectory,
    factual_concepts: LatentConceptPath,
    counterfactual_concepts: LatentConceptPath,
    bank: FutureNoiseBank,
    intervention: CounterfactualIntervention | None,
) -> None:
    horizon = bank.horizon
    dimension = bank.observation_innovations.shape[1]
    expected_path_shape = (horizon, dimension)
    trajectories = (factual_path, counterfactual_path)
    concepts_pair = (factual_concepts, counterfactual_concepts)
    for name, trajectory in zip(("factual", "counterfactual"), trajectories, strict=True):
        if trajectory.full_values.shape != expected_path_shape or trajectory.burn_in != 0:
            raise ValueError(f"{name}_path: expected shape {expected_path_shape}, burn_in 0")
    if len(factual_path.dynamics_schedule) != horizon:
        raise ValueError("factual_path: dynamics_schedule must match bank horizon")
    if any(
        left is not right
        for left, right in zip(
            factual_path.dynamics_schedule,
            counterfactual_path.dynamics_schedule,
            strict=True,
        )
    ):
        raise ValueError("paired paths: expected the exact same dynamics_schedule")
    for field_name in ("initial_history", "trend_history", "regime_labels", "trend_loading"):
        _same_array(
            getattr(factual_path, field_name),
            getattr(counterfactual_path, field_name),
            field_name=f"paired {field_name}",
        )
    if factual_path.observation_scale_floor != counterfactual_path.observation_scale_floor:
        raise ValueError("paired paths: observation_scale_floor must match")
    for field_name, supplied in (
        ("exogenous_inputs", bank.exogenous_inputs),
        ("observation_innovations", bank.observation_innovations),
        ("shocks", bank.shocks),
    ):
        for trajectory in trajectories:
            _same_array(
                getattr(trajectory, field_name),
                supplied,
                field_name=f"noise_bank.{field_name}",
            )
    for field_name, supplied in (
        ("trend_innovations", bank.trend_innovations),
        ("scale_innovations", bank.scale_innovations),
    ):
        for concepts in concepts_pair:
            _same_array(
                getattr(concepts, field_name),
                supplied,
                field_name=f"noise_bank.{field_name}",
            )
    labels = factual_path.regime_labels
    for allocation in (
        bank.regime_path,
        factual_concepts.regime_sequence,
        counterfactual_concepts.regime_sequence,
    ):
        if allocation is not None:
            _same_array(allocation, labels, field_name="regime_path trajectory/latent labels")
    for name, trajectory, concepts in zip(
        ("factual", "counterfactual"),
        trajectories,
        concepts_pair,
        strict=True,
    ):
        for concept in ("trend", "scale"):
            _same_array(
                getattr(trajectory, f"{concept}_path"),
                getattr(concepts, concept)[:-1],
                field_name=f"{name} {concept} path",
            )
    _validate_concept_pair(
        factual_concepts,
        counterfactual_concepts,
        intervention=intervention,
    )


def _validate_concept_pair(
    factual: LatentConceptPath,
    counterfactual: LatentConceptPath,
    *,
    intervention: CounterfactualIntervention | None,
) -> None:
    if factual.intervention is not None:
        raise ValueError("factual_concepts: expected no intervention")
    for field_name in (
        "trend_innovations",
        "scale_innovations",
        "regime_sequence",
        "trend_ar_coefficients",
        "scale_ar_coefficients",
    ):
        _same_array(
            getattr(factual, field_name),
            getattr(counterfactual, field_name),
            field_name=f"paired concepts {field_name}",
        )
    if intervention is None:
        if counterfactual.intervention is not None:
            raise ValueError("counterfactual_concepts: unexpected intervention")
        isolated_fields = ("trend", "scale")
    else:
        provenance = counterfactual.intervention
        if (
            provenance is None
            or provenance.concept != intervention.concept
            or provenance.origin_index != 0
            or np.float64(provenance.source_value).tobytes()
            != np.float64(intervention.source_value).tobytes()
        ):
            raise ValueError("counterfactual_concepts: intervention provenance mismatch")
        isolated_fields = ("scale",) if intervention.concept == "trend" else ("trend",)
    for field_name in isolated_fields:
        _same_array(
            getattr(factual, field_name),
            getattr(counterfactual, field_name),
            field_name=f"{field_name} concept isolation",
        )


def _copy_horizon_index(value: object, *, horizon: int) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError("horizon_index: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"horizon_index: expected dtype int64, got {value.dtype}")
    expected = np.arange(1, horizon + 1, dtype=np.int64)
    if value.shape != expected.shape or not np.array_equal(value, expected):
        raise ValueError("horizon_index: expected one-based values 1..H")
    return _read_only_copy(value)


def _causal_delay(value: object, *, horizon: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError("causal_delay: expected a non-negative integer")
    result = int(value)
    if result < 0 or result >= horizon:
        raise ValueError(f"causal_delay: expected a value in [0, {horizon - 1}], got {result}")
    return result


def _copy_allocation(
    value: object,
    *,
    intervention: CounterfactualIntervention | None,
    factual_concepts: LatentConceptPath,
) -> InterventionPair | None:
    if value is None:
        return None
    if not isinstance(value, InterventionPair):
        raise TypeError("allocation_metadata: expected an InterventionPair or None")
    payload = value.model_dump(mode="python")
    allocation = InterventionPair.model_validate(payload, strict=True)
    if intervention is None:
        raise ValueError("allocation_metadata: requires an intervention")
    if allocation.concept_name != intervention.concept:
        raise ValueError("allocation_metadata.concept_name: expected intervention concept")
    base = float(getattr(factual_concepts, intervention.concept)[0])
    expected_delta = intervention.source_value - base
    if allocation.concept_delta != expected_delta:
        raise ValueError(
            "allocation_metadata.concept_delta: expected source minus base "
            f"({expected_delta}), got {allocation.concept_delta}"
        )
    return allocation


def _same_array(
    left: NDArray[np.generic],
    right: NDArray[np.generic],
    *,
    field_name: str,
) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
        raise ValueError(f"{field_name}: expected bitwise-identical arrays")


def _optional_regime_uniforms(
    value: object,
    *,
    horizon: int,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    uniforms = _copy_float64_vector(
        value,
        field_name="regime_uniforms",
        shape=(horizon,),
    )
    if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("regime_uniforms: every value must lie in [0, 1)")
    return uniforms


def _optional_regime_path(value: object, *, horizon: int) -> NDArray[np.int64] | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise TypeError("regime_path: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"regime_path: expected dtype int64, got {value.dtype}")
    if value.shape != (horizon,):
        raise ValueError(f"regime_path: expected shape ({horizon},), got {value.shape}")
    if np.any(value < 0):
        raise ValueError("regime_path: labels must be non-negative")
    return _read_only_copy(value)


def _copy_float64_vector(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    array = _require_float64_array(value, field_name=field_name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field_name}: expected non-empty shape [H], got {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{field_name}: expected shape {shape}, got {array.shape}")
    return _read_only_copy(array)


def _copy_float64_matrix(
    value: object,
    *,
    field_name: str,
    rows: int | None = None,
    shape: tuple[int, int] | None = None,
    allow_zero_columns: bool = False,
) -> NDArray[np.float64]:
    array = _require_float64_array(value, field_name=field_name)
    if array.ndim != 2 or (array.shape[1] == 0 and not allow_zero_columns):
        raise ValueError(f"{field_name}: expected non-empty shape [H, D], got {array.shape}")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{field_name}: expected {rows} rows, got shape {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{field_name}: expected shape {shape}, got {array.shape}")
    return _read_only_copy(array)


def _require_float64_array(value: object, *, field_name: str) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype float64")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field_name}: expected dtype float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{field_name}: values must be finite")
    return value


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise TypeError(f"{field_name}: expected a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name}: expected a finite real scalar, got {result}")
    return result


def _read_only_copy(array: NDArray[np.generic]) -> NDArray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result
