"""Deterministic nonlinear VAR; callers supply every stochastic array."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import numpy as np
from numpy.typing import NDArray

_MAX_STABILITY_TARGET: Final = 0.85
_NUMERICAL_TOLERANCE: Final = 1e-10


@dataclass(frozen=True, slots=True)
class RegimeDynamics:
    """Frozen read-only float64 ``[D,D]``/``[D,U]`` regime parameters.
    Delays are causal; recorded radii audit generation scaling. Construction only
    validates ``final <= target <= 0.85`` and never scales or samples.
    """

    regime_label: int
    linear_matrix: NDArray[np.float64]
    nonlinear_matrix: NDArray[np.float64]
    exogenous_matrix: NDArray[np.float64]
    nonlinear_strength: float
    base_log_scale: float
    scale_loading: float
    nonlinear_delay: int
    trend_delay: int
    raw_spectral_radius: float
    spectral_scale_factor: float
    final_spectral_radius: float
    stability_target: float = 0.85
    true_graph: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        regime_label = _non_negative_integer(self.regime_label, field_name="regime_label")
        nonlinear_delay = _non_negative_integer(
            self.nonlinear_delay,
            field_name="nonlinear_delay",
        )
        trend_delay = _non_negative_integer(self.trend_delay, field_name="trend_delay")
        target = _validate_stability_target(
            self.stability_target,
            field_name="stability_target",
        )
        linear = _copy_float64_matrix(self.linear_matrix, field_name="linear_matrix")
        if linear.shape[0] != linear.shape[1]:
            raise ValueError(
                f"linear_matrix: expected square shape [D, D], got shape {linear.shape}"
            )
        dimension = linear.shape[0]
        nonlinear = _copy_float64_matrix(
            self.nonlinear_matrix,
            field_name="nonlinear_matrix",
            expected_shape=(dimension, dimension),
        )
        exogenous = _copy_float64_matrix(
            self.exogenous_matrix,
            field_name="exogenous_matrix",
            expected_rows=dimension,
            allow_empty=True,
        )
        strength = _finite_real(self.nonlinear_strength, field_name="nonlinear_strength")
        base_scale = _finite_real(self.base_log_scale, field_name="base_log_scale")
        scale_loading = _finite_real(self.scale_loading, field_name="scale_loading")
        raw_radius = _non_negative_real(
            self.raw_spectral_radius,
            field_name="raw_spectral_radius",
        )
        factor = _non_negative_real(
            self.spectral_scale_factor,
            field_name="spectral_scale_factor",
            strictly_positive=True,
        )
        final_radius = _non_negative_real(
            self.final_spectral_radius,
            field_name="final_spectral_radius",
        )

        actual_radius = spectral_radius(linear)
        if not _numerically_equal(final_radius, actual_radius):
            raise ValueError(
                f"final_spectral_radius: expected matrix radius {actual_radius}, got {final_radius}"
            )
        if final_radius > target + _NUMERICAL_TOLERANCE:
            raise ValueError(
                "stability_target: direct construction does not rescale matrices; "
                f"final radius {final_radius} exceeds target {target}"
            )
        expected_factor = target / raw_radius if raw_radius > target else 1.0
        expected_final = raw_radius * expected_factor
        if not _numerically_equal(factor, expected_factor) or not _numerically_equal(
            final_radius,
            expected_final,
        ):
            raise ValueError(
                "spectral evidence: expected the one-time generation rule "
                f"(raw={raw_radius}, factor={expected_factor}, final={expected_final}), "
                f"got factor={factor}, final={final_radius}"
            )

        expected_graph = (linear != 0.0) | ((strength != 0.0) & (nonlinear != 0.0))
        if self.true_graph is None:
            graph = _read_only_copy(expected_graph)
        else:
            graph = _bool_array(
                self.true_graph,
                field_name="true_graph",
                shape=(dimension, dimension),
                copy=True,
            )
            if not np.array_equal(graph, expected_graph):
                raise ValueError(
                    "true_graph: expected active state edges implied by linear_matrix "
                    "and nonlinear_strength * nonlinear_matrix"
                )

        validated = (
            regime_label,
            linear,
            nonlinear,
            exogenous,
            strength,
            base_scale,
            scale_loading,
        )
        validated += (nonlinear_delay, trend_delay, raw_radius, factor, final_radius, target, graph)
        for field, value in zip(fields(self), validated, strict=True):
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True, slots=True)
class _RolloutInputs:
    initial_history: NDArray[np.float64]
    trend_history: NDArray[np.float64]
    trend_path: NDArray[np.float64]
    scale_path: NDArray[np.float64]
    dynamics_schedule: tuple[RegimeDynamics, ...]
    regime_labels: NDArray[np.int64]
    exogenous_inputs: NDArray[np.float64]
    observation_innovations: NDArray[np.float64]
    shocks: NDArray[np.float64]
    trend_loading: NDArray[np.float64]
    observation_scale_floor: float
    burn_in: int


@dataclass(frozen=True, slots=True)
class SyntheticTrajectory:
    """Frozen read-only replay truth with no RNG.
    ``full_values[t]`` is float64 ``X[t+1]`` in ``[T,D]``; ``values`` is its
    burn-in suffix. Direct construction replays; rollout uses prepared inputs.
    """

    values: NDArray[np.float64]
    full_values: NDArray[np.float64]
    initial_history: NDArray[np.float64]
    trend_history: NDArray[np.float64]
    trend_path: NDArray[np.float64]
    scale_path: NDArray[np.float64]
    dynamics_schedule: tuple[RegimeDynamics, ...]
    regime_labels: NDArray[np.int64]
    exogenous_inputs: NDArray[np.float64]
    observation_innovations: NDArray[np.float64]
    shocks: NDArray[np.float64]
    trend_loading: NDArray[np.float64]
    observation_scale_floor: float
    burn_in: int

    def __post_init__(self) -> None:
        prepared = _prepare_rollout_inputs(
            initial_history=self.initial_history,
            trend_history=self.trend_history,
            trend_path=self.trend_path,
            scale_path=self.scale_path,
            dynamics_schedule=self.dynamics_schedule,
            regime_labels=self.regime_labels,
            exogenous_inputs=self.exogenous_inputs,
            observation_innovations=self.observation_innovations,
            shocks=self.shocks,
            trend_loading=self.trend_loading,
            observation_scale_floor=self.observation_scale_floor,
            burn_in=self.burn_in,
        )
        expected_full = _compute_rollout(prepared)
        full_values = _copy_float64_matrix(
            self.full_values,
            field_name="full_values",
            expected_shape=expected_full.shape,
        )
        values = _copy_float64_matrix(
            self.values,
            field_name="values",
            expected_shape=expected_full[prepared.burn_in :].shape,
        )
        if full_values.tobytes() != expected_full.tobytes():
            raise ValueError("full_values: values do not satisfy the deterministic recurrence")
        if values.tobytes() != full_values[prepared.burn_in :].tobytes():
            raise ValueError("values: expected bitwise equality with full_values[burn_in:]")
        for field in fields(prepared):
            object.__setattr__(self, field.name, getattr(prepared, field.name))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "full_values", full_values)


def spectral_radius(matrix: NDArray[np.float64]) -> float:
    """Return the radius of finite float64 ``matrix[D,D]`` without mutation or RNG."""

    candidate = _require_float64_array(matrix, field_name="matrix")
    if candidate.ndim != 2 or candidate.shape[0] == 0:
        raise ValueError(f"matrix: expected non-empty shape [D, D], got shape {candidate.shape}")
    if candidate.shape[0] != candidate.shape[1]:
        raise ValueError(f"matrix: expected square shape [D, D], got shape {candidate.shape}")
    try:
        eigenvalues = np.linalg.eigvals(candidate)
    except np.linalg.LinAlgError as error:
        raise ValueError("matrix: spectral radius computation did not converge") from error
    radius = float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(radius):
        raise ValueError("matrix: spectral radius must be finite")
    return radius


def scale_to_spectral_radius(
    matrix: NDArray[np.float64],
    target: float = 0.85,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Scale float64 ``matrix[D,D]`` once to ``target <= 0.85`` without RNG.

    Returns a read-only copy and audited raw radius, factor, and final radius.
    """

    validated_target = _validate_stability_target(target, field_name="target")
    raw_radius = spectral_radius(matrix)
    factor = validated_target / raw_radius if raw_radius > validated_target else 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.asarray(matrix * factor, dtype=np.float64)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("matrix: scaled values must be finite")
    result = _read_only_copy(scaled)
    final_radius = spectral_radius(result)
    if final_radius > validated_target:
        factor *= (validated_target / final_radius) * (1.0 - _NUMERICAL_TOLERANCE)
        result = _read_only_copy(np.asarray(matrix * factor, dtype=np.float64))
        final_radius = spectral_radius(result)
    if final_radius > validated_target:
        raise ValueError(
            f"matrix: final spectral radius {final_radius} exceeds target {validated_target}"
        )
    return result, raw_radius, float(factor), final_radius


def generate_regime_dynamics(
    *,
    linear_candidates: NDArray[np.float64],
    nonlinear_matrices: NDArray[np.float64],
    exogenous_matrices: NDArray[np.float64],
    nonlinear_strengths: NDArray[np.float64],
    base_log_scales: NDArray[np.float64],
    scale_loadings: NDArray[np.float64],
    nonlinear_delays: NDArray[np.int64],
    trend_delays: NDArray[np.int64],
    target: float = 0.85,
    true_graphs: NDArray[np.bool_] | None = None,
) -> tuple[RegimeDynamics, ...]:
    """Build frozen dynamics from explicit arrays without RNG.

    Shapes are float64 ``[R,D,D]``/``[R,D,U]``/``[R]``, int64 delays ``[R]``,
    and optional bool graphs ``[R,D,D]``. Only linear candidates scale once.
    """

    validated_target = _validate_stability_target(target, field_name="target")
    linear = _require_float64_array(linear_candidates, field_name="linear_candidates")
    if linear.ndim != 3 or linear.shape[0] == 0 or linear.shape[1] == 0:
        raise ValueError(
            f"linear_candidates: expected non-empty shape [R, D, D], got shape {linear.shape}"
        )
    regimes, dimension, final_dimension = linear.shape
    if dimension != final_dimension:
        raise ValueError(
            f"linear_candidates: expected square trailing shape [D, D], got shape {linear.shape}"
        )
    nonlinear = _require_float64_array(
        nonlinear_matrices,
        field_name="nonlinear_matrices",
    )
    _require_shape(
        nonlinear,
        field_name="nonlinear_matrices",
        expected_shape=(regimes, dimension, dimension),
    )
    exogenous = _require_float64_array(
        exogenous_matrices,
        field_name="exogenous_matrices",
    )
    if exogenous.ndim != 3 or exogenous.shape[:2] != (regimes, dimension):
        raise ValueError(
            "exogenous_matrices: expected shape "
            f"[{regimes}, {dimension}, U], got shape {exogenous.shape}"
        )

    def float_path(value: object, name: str) -> NDArray[np.float64]:
        return _float64_vector(value, field_name=name, length=regimes)

    strengths = float_path(nonlinear_strengths, "nonlinear_strengths")
    base_scales = float_path(base_log_scales, "base_log_scales")
    loadings = float_path(scale_loadings, "scale_loadings")
    state_delays = _int64_vector(nonlinear_delays, field_name="nonlinear_delays", length=regimes)
    concept_delays = _int64_vector(trend_delays, field_name="trend_delays", length=regimes)
    if np.any(state_delays < 0):
        raise ValueError("nonlinear_delays: every delay must be non-negative")
    if np.any(concept_delays < 0):
        raise ValueError("trend_delays: every delay must be non-negative")
    graphs: NDArray[np.bool_] | None
    if true_graphs is None:
        graphs = None
    else:
        graphs = _bool_array(true_graphs, field_name="true_graphs")
        _require_shape(
            graphs,
            field_name="true_graphs",
            expected_shape=(regimes, dimension, dimension),
        )

    generated: list[RegimeDynamics] = []
    for regime_label in range(regimes):
        scaled, raw_radius, factor, final_radius = scale_to_spectral_radius(
            linear[regime_label],
            target=validated_target,
        )
        generated.append(
            RegimeDynamics(
                regime_label=regime_label,
                linear_matrix=scaled,
                nonlinear_matrix=nonlinear[regime_label],
                exogenous_matrix=exogenous[regime_label],
                nonlinear_strength=float(strengths[regime_label]),
                base_log_scale=float(base_scales[regime_label]),
                scale_loading=float(loadings[regime_label]),
                nonlinear_delay=int(state_delays[regime_label]),
                trend_delay=int(concept_delays[regime_label]),
                raw_spectral_radius=raw_radius,
                spectral_scale_factor=factor,
                final_spectral_radius=final_radius,
                stability_target=validated_target,
                true_graph=None if graphs is None else graphs[regime_label],
            )
        )
    return tuple(generated)


def deterministic_transition(
    *,
    state_history: NDArray[np.float64],
    trend_history: NDArray[np.float64],
    scale_state: float,
    exogenous_input: NDArray[np.float64],
    observation_innovation: NDArray[np.float64],
    shock: NDArray[np.float64],
    dynamics: RegimeDynamics,
    trend_loading: NDArray[np.float64],
    observation_scale_floor: float,
) -> NDArray[np.float64]:
    """Compute read-only float64 ``X[t+1]`` without mutation/RNG.

    Oldest-to-current ``state[H,D]``/``trend[C]`` use exact ``[-1-delay]`` indices;
    exogenous is ``[U]`` and innovation, shock, and trend loading are ``[D]``.
    """

    if not isinstance(dynamics, RegimeDynamics):
        raise TypeError("dynamics: expected a RegimeDynamics")
    dimension = dynamics.linear_matrix.shape[0]
    exogenous_dimension = dynamics.exogenous_matrix.shape[1]
    states = _require_float64_array(state_history, field_name="state_history")
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != dimension:
        raise ValueError(
            f"state_history: expected non-empty shape [H, {dimension}], got shape {states.shape}"
        )
    required_state_count = dynamics.nonlinear_delay + 1
    if states.shape[0] < required_state_count:
        raise ValueError(
            "state_history: expected at least nonlinear_delay + 1 "
            f"({required_state_count}) rows, got {states.shape[0]}"
        )
    concepts = _require_float64_array(trend_history, field_name="trend_history")
    required_concept_count = dynamics.trend_delay + 1
    if concepts.ndim != 1 or concepts.size < required_concept_count:
        raise ValueError(
            "trend_history: expected at least trend_delay + 1 "
            f"({required_concept_count}) values, got shape {concepts.shape}"
        )
    exogenous = _float64_vector(
        exogenous_input,
        field_name="exogenous_input",
        length=exogenous_dimension,
        allow_empty=exogenous_dimension == 0,
    )
    innovation = _float64_vector(
        observation_innovation,
        field_name="observation_innovation",
        length=dimension,
    )
    supplied_shock = _float64_vector(
        shock,
        field_name="shock",
        length=dimension,
    )
    loading = _float64_vector(
        trend_loading,
        field_name="trend_loading",
        length=dimension,
    )
    current_scale = _finite_real(scale_state, field_name="scale_state")
    floor = _finite_real(
        observation_scale_floor,
        field_name="observation_scale_floor",
    )
    if floor <= 0.0:
        raise ValueError(
            f"observation_scale_floor: expected a strictly positive finite value, got {floor}"
        )

    return _trusted_transition(
        current_state=states[-1],
        lagged_state=states[-1 - dynamics.nonlinear_delay],
        trend_state=float(concepts[-1 - dynamics.trend_delay]),
        scale_state=current_scale,
        exogenous_input=exogenous,
        observation_innovation=innovation,
        shock=supplied_shock,
        dynamics=dynamics,
        trend_loading=loading,
        observation_scale_floor=floor,
    )


def _trusted_transition(
    *,
    current_state: NDArray[np.float64],
    lagged_state: NDArray[np.float64],
    trend_state: float,
    scale_state: float,
    exogenous_input: NDArray[np.float64],
    observation_innovation: NDArray[np.float64],
    shock: NDArray[np.float64],
    dynamics: RegimeDynamics,
    trend_loading: NDArray[np.float64],
    observation_scale_floor: float,
) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore"):
        scale_argument = dynamics.base_log_scale + dynamics.scale_loading * scale_state
        observation_scale = np.logaddexp(0.0, scale_argument) + observation_scale_floor
        next_state = (
            dynamics.linear_matrix @ current_state
            + dynamics.nonlinear_strength * np.tanh(dynamics.nonlinear_matrix @ lagged_state)
            + trend_loading * trend_state
            + dynamics.exogenous_matrix @ exogenous_input
            + observation_scale * observation_innovation
            + shock
        )
    if not np.isfinite(scale_argument) or not np.isfinite(observation_scale):
        raise ValueError("observation scale: response must be finite")
    if not np.all(np.isfinite(next_state)):
        raise ValueError("transition output: values must be finite")
    return _read_only_copy(np.asarray(next_state, dtype=np.float64))


def rollout_nonlinear_var(
    *,
    initial_history: NDArray[np.float64],
    trend_history: NDArray[np.float64],
    trend_path: NDArray[np.float64],
    scale_path: NDArray[np.float64],
    dynamics_schedule: tuple[RegimeDynamics, ...],
    regime_labels: NDArray[np.int64],
    exogenous_inputs: NDArray[np.float64],
    observation_innovations: NDArray[np.float64],
    shocks: NDArray[np.float64],
    trend_loading: NDArray[np.float64],
    observation_scale_floor: float,
    burn_in: int = 0,
) -> SyntheticTrajectory:
    """Roll out ``T`` explicit steps without padding or RNG.

    Float64 shapes are initial ``[H,D]``, trend/scale ``[T]``, exogenous ``[T,U]``,
    noise/shocks ``[T,D]``; labels are int64 ``[T]``. Step ``t`` uses its schedule
    object and available history to emit ``X[t+1]``; burn-in cuts aligned outputs.
    """

    prepared = _prepare_rollout_inputs(
        initial_history=initial_history,
        trend_history=trend_history,
        trend_path=trend_path,
        scale_path=scale_path,
        dynamics_schedule=dynamics_schedule,
        regime_labels=regime_labels,
        exogenous_inputs=exogenous_inputs,
        observation_innovations=observation_innovations,
        shocks=shocks,
        trend_loading=trend_loading,
        observation_scale_floor=observation_scale_floor,
        burn_in=burn_in,
    )
    full_values = _compute_rollout(prepared)
    trajectory = object.__new__(SyntheticTrajectory)
    object.__setattr__(trajectory, "values", _read_only_copy(full_values[prepared.burn_in :]))
    object.__setattr__(trajectory, "full_values", full_values)
    for field in fields(prepared):
        object.__setattr__(trajectory, field.name, getattr(prepared, field.name))
    return trajectory


def _prepare_rollout_inputs(
    *,
    initial_history: object,
    trend_history: object,
    trend_path: object,
    scale_path: object,
    dynamics_schedule: object,
    regime_labels: object,
    exogenous_inputs: object,
    observation_innovations: object,
    shocks: object,
    trend_loading: object,
    observation_scale_floor: object,
    burn_in: object,
) -> _RolloutInputs:
    if not isinstance(dynamics_schedule, tuple) or not dynamics_schedule:
        raise ValueError("dynamics_schedule: expected a non-empty tuple")
    if any(not isinstance(item, RegimeDynamics) for item in dynamics_schedule):
        raise TypeError("dynamics_schedule: every item must be RegimeDynamics")
    schedule = dynamics_schedule
    steps = len(schedule)
    dimension = schedule[0].linear_matrix.shape[0]
    exogenous_dimension = schedule[0].exogenous_matrix.shape[1]
    if any(
        item.linear_matrix.shape != (dimension, dimension)
        or item.exogenous_matrix.shape != (dimension, exogenous_dimension)
        for item in schedule
    ):
        raise ValueError("dynamics_schedule: every item must share dimensions D and U")

    labels = _int64_vector(
        regime_labels,
        field_name="regime_labels",
        length=steps,
        copy=True,
    )
    if np.any(labels < 0):
        raise ValueError("regime_labels: labels must be non-negative")
    expected_labels = np.array([item.regime_label for item in schedule], dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("regime_labels: labels must match the per-step dynamics_schedule")

    state_history = _copy_float64_matrix(
        initial_history,
        field_name="initial_history",
        expected_rows=None,
    )
    if state_history.shape[1] != dimension:
        raise ValueError(
            f"initial_history: expected shape [H, {dimension}], got {state_history.shape}"
        )
    required_states = max(item.nonlinear_delay for item in schedule) + 1
    if state_history.shape[0] < required_states:
        raise ValueError(
            "initial_history: expected at least max nonlinear_delay + 1 "
            f"({required_states}) rows, got {state_history.shape[0]}"
        )
    past_trend = _float64_vector(
        trend_history,
        field_name="trend_history",
        allow_empty=True,
        copy=True,
    )
    required_past_trend = max(item.trend_delay for item in schedule)
    if past_trend.size < required_past_trend:
        raise ValueError(
            "trend_history: expected at least max trend_delay "
            f"({required_past_trend}) values, got {past_trend.size}"
        )
    current_trend = _float64_vector(
        trend_path,
        field_name="trend_path",
        length=steps,
        copy=True,
    )
    current_scale = _float64_vector(
        scale_path,
        field_name="scale_path",
        length=steps,
        copy=True,
    )
    exogenous = _copy_float64_matrix(
        exogenous_inputs,
        field_name="exogenous_inputs",
        expected_shape=(steps, exogenous_dimension),
        allow_empty=True,
    )
    innovations = _copy_float64_matrix(
        observation_innovations,
        field_name="observation_innovations",
        expected_shape=(steps, dimension),
    )
    supplied_shocks = _copy_float64_matrix(
        shocks,
        field_name="shocks",
        expected_shape=(steps, dimension),
    )
    loading = _float64_vector(
        trend_loading,
        field_name="trend_loading",
        length=dimension,
        copy=True,
    )
    floor = _finite_real(observation_scale_floor, field_name="observation_scale_floor")
    if floor <= 0.0:
        raise ValueError("observation_scale_floor: expected a strictly positive value")
    removed_steps = _non_negative_integer(burn_in, field_name="burn_in")
    if removed_steps >= steps:
        raise ValueError(f"burn_in: expected a value in [0, {steps - 1}], got {removed_steps}")
    prepared = (state_history, past_trend, current_trend, current_scale, schedule, labels)
    prepared += (exogenous, innovations, supplied_shocks, loading, floor, removed_steps)
    return _RolloutInputs(*prepared)


def _compute_rollout(inputs: _RolloutInputs) -> NDArray[np.float64]:
    history_length, dimension = inputs.initial_history.shape
    steps = len(inputs.dynamics_schedule)
    state_buffer = np.empty((history_length + steps, dimension), dtype=np.float64)
    state_buffer[:history_length] = inputs.initial_history
    complete_trend = np.concatenate((inputs.trend_history, inputs.trend_path))
    trend_offset = inputs.trend_history.size
    for step, dynamics in enumerate(inputs.dynamics_schedule):
        current_index = history_length + step - 1
        state_buffer[history_length + step] = _trusted_transition(
            current_state=state_buffer[current_index],
            lagged_state=state_buffer[current_index - dynamics.nonlinear_delay],
            trend_state=float(complete_trend[trend_offset + step - dynamics.trend_delay]),
            scale_state=float(inputs.scale_path[step]),
            exogenous_input=inputs.exogenous_inputs[step],
            observation_innovation=inputs.observation_innovations[step],
            shock=inputs.shocks[step],
            dynamics=dynamics,
            trend_loading=inputs.trend_loading,
            observation_scale_floor=inputs.observation_scale_floor,
        )
    return _read_only_copy(state_buffer[history_length:])


def _require_float64_array(value: object, *, field_name: str) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype float64")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field_name}: expected dtype float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{field_name}: values must be finite")
    return value


def _require_shape(
    value: NDArray[np.generic],
    *,
    field_name: str,
    expected_shape: tuple[int, ...],
) -> None:
    if value.shape != expected_shape:
        raise ValueError(f"{field_name}: expected shape {expected_shape}, got shape {value.shape}")


def _float64_vector(
    value: object,
    *,
    field_name: str,
    length: int | None = None,
    allow_empty: bool = False,
    copy: bool = False,
) -> NDArray[np.float64]:
    vector = _require_float64_array(value, field_name=field_name)
    if vector.ndim != 1 or (vector.size == 0 and not allow_empty):
        raise ValueError(f"{field_name}: expected shape [N], got shape {vector.shape}")
    if length is not None and vector.shape != (length,):
        raise ValueError(f"{field_name}: expected shape ({length},), got shape {vector.shape}")
    return _read_only_copy(vector) if copy else vector


def _int64_vector(
    value: object,
    *,
    field_name: str,
    length: int,
    copy: bool = False,
) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"{field_name}: expected dtype int64, got {value.dtype}")
    if value.shape != (length,):
        raise ValueError(f"{field_name}: expected shape ({length},), got shape {value.shape}")
    return _read_only_copy(value) if copy else value


def _bool_array(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...] | None = None,
    copy: bool = False,
) -> NDArray[np.bool_]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype bool")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field_name}: expected dtype bool, got {value.dtype}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{field_name}: expected shape {shape}, got shape {value.shape}")
    return _read_only_copy(value) if copy else value


def _copy_float64_matrix(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, int] | None = None,
    expected_rows: int | None = None,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    matrix = _require_float64_array(value, field_name=field_name)
    if matrix.ndim != 2 or (matrix.size == 0 and not allow_empty):
        raise ValueError(f"{field_name}: expected a non-empty matrix, got shape {matrix.shape}")
    if expected_shape is not None and matrix.shape != expected_shape:
        raise ValueError(f"{field_name}: expected shape {expected_shape}, got shape {matrix.shape}")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{field_name}: expected shape [{expected_rows}, U], got shape {matrix.shape}"
        )
    return _read_only_copy(matrix)


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


def _non_negative_real(
    value: object,
    *,
    field_name: str,
    strictly_positive: bool = False,
) -> float:
    result = _finite_real(value, field_name=field_name)
    if result < 0.0 or (strictly_positive and result == 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{field_name}: expected a {qualifier} finite value, got {result}")
    return result


def _non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{field_name}: expected a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name}: expected a non-negative integer, got {result}")
    return result


def _validate_stability_target(value: object, *, field_name: str) -> float:
    target = _finite_real(value, field_name=field_name)
    if target <= 0.0 or target > _MAX_STABILITY_TARGET:
        raise ValueError(f"{field_name}: expected value in (0, 0.85], got {target}")
    return target


def _numerically_equal(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=_NUMERICAL_TOLERANCE, atol=_NUMERICAL_TOLERANCE))


def _read_only_copy(array: NDArray[np.generic]) -> NDArray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result
