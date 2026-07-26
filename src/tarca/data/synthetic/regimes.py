"""Deterministic random streams and Markov-regime utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

RANDOM_STREAM_NAMES: Final[tuple[str, ...]] = (
    "regime_transitions",
    "trend_innovations",
    "scale_innovations",
    "exogenous_variables",
    "observation_innovations",
    "sparse_shocks",
    "missingness",
    "parameter_generation",
    "counterfactual_mc_bank",
    "random_concept_negative_control",
)

_PROBABILITY_TOLERANCE: Final = 1e-12

ParameterScalar: TypeAlias = int | float
ParameterValue: TypeAlias = ParameterScalar | NDArray[np.float64]
RegimeParameterSet: TypeAlias = Mapping[str, ParameterValue]
RegimeParameters: TypeAlias = Mapping[int, RegimeParameterSet]
PersistenceStatistic: TypeAlias = int | float


@dataclass(frozen=True, slots=True)
class RandomStream:
    """Named child random stream derived from one root ``SeedSequence``.

    ``spawn_key`` records the exact child position. The record cannot be reassigned,
    while callers intentionally advance its local ``generator`` when pre-generating
    stochastic arrays.
    """

    name: str
    spawn_key: tuple[int, ...]
    generator: Generator


def spawn_random_streams(root_seed: int) -> Mapping[str, RandomStream]:
    """Create the ten isolated random streams in their specified order.

    Args:
        root_seed: Non-negative Python integer passed to ``numpy.random.SeedSequence``.

    Returns:
        An immutable name-to-stream mapping. Repeating ``root_seed`` reconstructs every
        child stream, and consuming one child cannot advance another.

    Raises:
        TypeError: If ``root_seed`` is not a Python integer.
        ValueError: If ``root_seed`` is negative.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed: expected a non-negative Python int")
    if root_seed < 0:
        raise ValueError(f"root_seed: expected a non-negative Python int, got {root_seed}")

    children = np.random.SeedSequence(root_seed).spawn(len(RANDOM_STREAM_NAMES))
    streams = {
        name: RandomStream(
            name=name,
            spawn_key=tuple(int(part) for part in child.spawn_key),
            generator=np.random.default_rng(child),
        )
        for name, child in zip(RANDOM_STREAM_NAMES, children, strict=True)
    }
    return MappingProxyType(streams)


def validate_transition_matrix(matrix: NDArray[np.float64]) -> None:
    """Validate a finite float64 row-stochastic matrix of shape ``[R, R]``.

    Validation never normalizes, clips, or mutates ``matrix``. Row sums must equal one
    within an absolute tolerance of ``1e-12`` and zero relative tolerance.

    Args:
        matrix: Candidate transition probabilities with float64 dtype.

    Raises:
        TypeError: If ``matrix`` is not a NumPy float64 array.
        ValueError: If its shape or probability values are invalid.
    """

    transition = _require_float64_array(matrix, field_name="transition")
    if transition.ndim != 2 or transition.shape[0] == 0:
        raise ValueError(
            f"transition: expected non-empty shape [R, R], got shape {transition.shape}"
        )
    if transition.shape[0] != transition.shape[1]:
        raise ValueError(f"transition: expected square shape [R, R], got shape {transition.shape}")
    if np.any(transition < 0.0) or np.any(transition > 1.0):
        raise ValueError("transition: probabilities must lie in [0, 1]")
    row_sums = transition.sum(axis=1, dtype=np.float64)
    if not np.allclose(
        row_sums,
        np.ones_like(row_sums),
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(
            "transition: every row must sum to 1 within absolute tolerance "
            f"{_PROBABILITY_TOLERANCE}; got {row_sums.tolist()}"
        )


def sample_regime_sequence(
    transition: NDArray[np.float64],
    initial_probabilities: NDArray[np.float64],
    uniforms: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Roll out a Markov-regime path using only pre-generated uniforms.

    ``transition`` has shape ``[R, R]`` and ``initial_probabilities`` has shape
    ``[R]``. ``uniforms`` is a non-empty float64 array of shape ``[T]`` with values
    in ``[0, 1)``. Its first element selects the initial regime; each remaining
    element selects the next regime from the preceding regime's transition row.
    The returned shape is ``[T]``, dtype is int64, and the array is read-only.

    No random generator or global random state is read by this function.
    """

    validate_transition_matrix(transition)
    number_of_regimes = transition.shape[0]
    initial = _validate_probability_vector(
        initial_probabilities,
        field_name="initial_probabilities",
        expected_length=number_of_regimes,
    )
    supplied_uniforms = _require_float64_array(uniforms, field_name="uniforms")
    if supplied_uniforms.ndim != 1 or supplied_uniforms.size == 0:
        raise ValueError(
            f"uniforms: expected non-empty shape [T], got shape {supplied_uniforms.shape}"
        )
    if np.any(supplied_uniforms < 0.0) or np.any(supplied_uniforms >= 1.0):
        raise ValueError("uniforms: every value must lie in [0, 1)")

    sequence = np.empty(supplied_uniforms.size, dtype=np.int64)
    sequence[0] = _categorical_index(initial, float(supplied_uniforms[0]))
    for time_index in range(1, supplied_uniforms.size):
        previous_regime = int(sequence[time_index - 1])
        sequence[time_index] = _categorical_index(
            transition[previous_regime],
            float(supplied_uniforms[time_index]),
        )
    return _as_read_only(sequence)


def compute_stationary_distribution(
    transition: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the unique stationary distribution of ``transition``.

    Args:
        transition: Finite float64 row-stochastic matrix with shape ``[R, R]``.

    Returns:
        A read-only float64 probability vector of shape ``[R]`` satisfying
        ``distribution @ transition == distribution`` to numerical precision.

    Raises:
        ValueError: If the transition matrix does not have a unique stationary
            distribution or a valid numerical solution cannot be obtained.
    """

    validate_transition_matrix(transition)
    number_of_regimes = transition.shape[0]
    constraints = np.concatenate(
        (
            transition.T - np.eye(number_of_regimes, dtype=np.float64),
            np.ones((1, number_of_regimes), dtype=np.float64),
        ),
        axis=0,
    )
    if np.linalg.matrix_rank(constraints) < number_of_regimes:
        raise ValueError("transition: expected a unique stationary distribution")

    target = np.zeros(number_of_regimes + 1, dtype=np.float64)
    target[-1] = 1.0
    distribution, _, _, _ = np.linalg.lstsq(constraints, target, rcond=None)
    if not np.all(np.isfinite(distribution)):
        raise ValueError("transition: stationary distribution must be finite")
    if np.any(distribution < -_PROBABILITY_TOLERANCE):
        raise ValueError("transition: stationary distribution contains negative probability")

    distribution = np.where(distribution < 0.0, 0.0, distribution)
    probability_sum = float(distribution.sum(dtype=np.float64))
    if probability_sum <= 0.0:
        raise ValueError("transition: stationary distribution has zero probability mass")
    distribution = np.asarray(distribution / probability_sum, dtype=np.float64)
    if not np.allclose(
        distribution @ transition,
        distribution,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("transition: stationary distribution residual exceeds tolerance")
    return _as_read_only(distribution)


def regime_persistence_statistics(
    regime_sequence: NDArray[np.int64],
) -> Mapping[str, PersistenceStatistic]:
    """Summarize contiguous dwell runs in an int64 regime path of shape ``[T]``.

    Returns an immutable mapping containing run and transition counts plus aggregate
    dwell lengths. ``regime_sequence`` is validated and never modified.
    """

    sequence = _validate_regime_sequence(regime_sequence)
    change_points = np.flatnonzero(sequence[1:] != sequence[:-1]) + 1
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            change_points.astype(np.int64, copy=False),
            np.array([sequence.size], dtype=np.int64),
        )
    )
    dwell_lengths = np.diff(boundaries)
    statistics: dict[str, PersistenceStatistic] = {
        "number_of_runs": int(dwell_lengths.size),
        "number_of_transitions": int(change_points.size),
        "mean_dwell_time": float(np.mean(dwell_lengths)),
        "median_dwell_time": float(np.median(dwell_lengths)),
        "maximum_dwell_time": int(np.max(dwell_lengths)),
    }
    return MappingProxyType(statistics)


def resolve_regime_parameters(
    parameters_by_regime: Mapping[int, Mapping[str, ParameterValue]],
) -> RegimeParameters:
    """Validate and deeply freeze numeric parameters indexed by regime label.

    Labels must be exactly ``0..R-1``. Every regime must define the same non-empty
    field set, and corresponding parameters must have matching scalar kinds or array
    shapes. Float64 arrays are copied and marked read-only; numeric scalars are copied
    into built-in ``int`` or ``float`` values.
    """

    if not isinstance(parameters_by_regime, Mapping) or not parameters_by_regime:
        raise ValueError("parameters_by_regime: expected a non-empty mapping")

    labels = tuple(parameters_by_regime)
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise TypeError("parameters_by_regime: regime labels must be Python ints")
    sorted_labels = tuple(sorted(labels))
    expected_labels = tuple(range(len(labels)))
    if sorted_labels != expected_labels:
        raise ValueError(
            "parameters_by_regime: regime labels must be contiguous 0..R-1; "
            f"got {list(sorted_labels)}"
        )

    resolved: dict[int, RegimeParameterSet] = {}
    expected_fields: tuple[str, ...] | None = None
    expected_signatures: dict[str, tuple[str, tuple[int, ...] | None]] | None = None
    for label in sorted_labels:
        parameter_set = parameters_by_regime[label]
        if not isinstance(parameter_set, Mapping) or not parameter_set:
            raise ValueError(
                f"parameters_by_regime[{label}]: expected a non-empty parameter mapping"
            )
        fields = tuple(parameter_set)
        if any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ValueError(
                f"parameters_by_regime[{label}]: parameter names must be non-empty strings"
            )
        sorted_fields = tuple(sorted(fields))
        if expected_fields is None:
            expected_fields = sorted_fields
        elif sorted_fields != expected_fields:
            raise ValueError(
                f"parameters_by_regime[{label}]: expected fields {list(expected_fields)}, "
                f"got {list(sorted_fields)}"
            )

        frozen_values = {
            field: _copy_parameter_value(
                parameter_set[field],
                field_name=f"parameters_by_regime[{label}].{field}",
            )
            for field in sorted_fields
        }
        signatures = {field: _parameter_signature(value) for field, value in frozen_values.items()}
        if expected_signatures is None:
            expected_signatures = signatures
        else:
            for field, signature in signatures.items():
                if signature != expected_signatures[field]:
                    raise ValueError(
                        f"parameters_by_regime[{label}].{field}: expected parameter "
                        f"signature {expected_signatures[field]}, got {signature}"
                    )
        resolved[label] = MappingProxyType(frozen_values)
    return MappingProxyType(resolved)


def build_regime_parameter_schedule(
    regime_sequence: NDArray[np.int64],
    parameters_by_regime: Mapping[int, Mapping[str, ParameterValue]],
) -> tuple[RegimeParameterSet, ...]:
    """Resolve one immutable parameter set per element of ``regime_sequence``.

    ``regime_sequence`` must be a non-empty int64 array of shape ``[T]``. The returned
    tuple also has length ``T`` and never aliases mutable parameter arrays supplied by
    the caller.
    """

    sequence = _validate_regime_sequence(regime_sequence)
    parameters = resolve_regime_parameters(parameters_by_regime)
    unknown_labels = sorted(set(int(label) for label in sequence) - set(parameters))
    if unknown_labels:
        raise ValueError(
            "regime_sequence: labels must exist in parameters_by_regime; "
            f"unknown labels {unknown_labels}"
        )
    return tuple(parameters[int(label)] for label in sequence)


def make_unseen_parameter_shift(
    parameters_by_regime: Mapping[int, Mapping[str, ParameterValue]],
    shifts: Mapping[str, ParameterValue],
) -> RegimeParameters:
    """Build immutable unseen-environment parameters using explicit additive shifts.

    ``shifts`` maps existing parameter names to finite scalar or float64-array
    increments. A scalar increment may be broadcast over an array; an array increment
    must exactly match the base array shape. Integer parameters require integer shifts.
    Regime labels and unshifted fields are preserved, and all returned arrays are
    independent read-only copies.
    """

    parameters = resolve_regime_parameters(parameters_by_regime)
    if not isinstance(shifts, Mapping) or not shifts:
        raise ValueError("shifts: expected a non-empty parameter mapping")
    available_fields = set(next(iter(parameters.values())))
    unknown_fields = sorted(set(shifts) - available_fields)
    if unknown_fields:
        raise ValueError(f"shifts: unknown parameter fields {unknown_fields}")

    validated_shifts = {
        field: _copy_parameter_value(value, field_name=f"shifts.{field}")
        for field, value in shifts.items()
    }
    shifted_parameters: dict[int, dict[str, ParameterValue]] = {}
    for label, parameter_set in parameters.items():
        shifted_parameters[label] = {
            field: _shift_parameter_value(
                value,
                validated_shifts[field],
                field_name=f"parameters_by_regime[{label}].{field}",
            )
            if field in validated_shifts
            else _copy_parameter_value(
                value,
                field_name=f"parameters_by_regime[{label}].{field}",
            )
            for field, value in parameter_set.items()
        }
    return resolve_regime_parameters(shifted_parameters)


def _require_float64_array(
    value: object,
    *,
    field_name: str,
) -> NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype float64")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field_name}: expected dtype float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{field_name}: values must be finite")
    return value


def _validate_probability_vector(
    probabilities: NDArray[np.float64],
    *,
    field_name: str,
    expected_length: int,
) -> NDArray[np.float64]:
    vector = _require_float64_array(probabilities, field_name=field_name)
    if vector.ndim != 1 or vector.shape != (expected_length,):
        raise ValueError(
            f"{field_name}: expected shape [{expected_length}], got shape {vector.shape}"
        )
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{field_name}: probabilities must lie in [0, 1]")
    probability_sum = float(vector.sum(dtype=np.float64))
    if not np.isclose(
        probability_sum,
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(
            f"{field_name}: probabilities must sum to 1 within absolute tolerance "
            f"{_PROBABILITY_TOLERANCE}; got {probability_sum}"
        )
    return vector


def _categorical_index(probabilities: NDArray[np.float64], uniform: float) -> int:
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, uniform, side="right"))


def _validate_regime_sequence(value: object) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError("regime_sequence: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"regime_sequence: expected dtype int64, got {value.dtype}")
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"regime_sequence: expected non-empty shape [T], got shape {value.shape}")
    if np.any(value < 0):
        raise ValueError("regime_sequence: labels must be non-negative")
    return value


def _copy_parameter_value(value: object, *, field_name: str) -> ParameterValue:
    if isinstance(value, np.ndarray):
        array = _require_float64_array(value, field_name=field_name)
        if array.size == 0:
            raise ValueError(f"{field_name}: expected a non-empty parameter array")
        return _as_read_only(array.copy())
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{field_name}: expected a finite numeric scalar or float64 array")
    if not np.isfinite(value):
        raise ValueError(f"{field_name}: values must be finite")
    if isinstance(value, int | np.integer):
        return int(value)
    return float(value)


def _parameter_signature(value: ParameterValue) -> tuple[str, tuple[int, ...] | None]:
    if isinstance(value, np.ndarray):
        return ("array", value.shape)
    if isinstance(value, int):
        return ("integer", None)
    return ("floating", None)


def _shift_parameter_value(
    base: ParameterValue,
    shift: ParameterValue,
    *,
    field_name: str,
) -> ParameterValue:
    if isinstance(base, np.ndarray):
        if isinstance(shift, np.ndarray) and shift.shape != base.shape:
            raise ValueError(
                f"{field_name}: shift shape must be scalar or {base.shape}, got shape {shift.shape}"
            )
        shifted_array = np.asarray(base + shift, dtype=np.float64)
        if not np.all(np.isfinite(shifted_array)):
            raise ValueError(f"{field_name}: shifted values must be finite")
        return _as_read_only(shifted_array.copy())
    if isinstance(shift, np.ndarray):
        raise ValueError(
            f"{field_name}: shift shape must be scalar for a scalar parameter, "
            f"got shape {shift.shape}"
        )
    if isinstance(base, int) and not isinstance(shift, int):
        raise TypeError(f"{field_name}: expected an integer shift for an integer parameter")
    shifted_scalar = base + shift
    if not np.isfinite(shifted_scalar):
        raise ValueError(f"{field_name}: shifted value must be finite")
    if isinstance(base, int):
        return int(shifted_scalar)
    return float(shifted_scalar)


def _as_read_only(array: NDArray[np.generic]) -> NDArray:
    array.setflags(write=False)
    return array
