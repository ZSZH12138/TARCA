"""Deterministic, isolated trend and scale concepts for the synthetic SCM.

This module deliberately has no random-number generator. Callers must pre-generate
trend and scale innovations on their respective independent random streams.
``concept_overlap`` is not a state-equation parameter: overlap belongs exclusively
to downstream observation loadings, so neither latent state can read the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

ConceptName: TypeAlias = Literal["trend", "scale"]
ScaleParameter: TypeAlias = float | NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LatentConceptPath:
    """Immutable truth record for two independent regime-specific AR paths.

    For ``T`` transitions, ``trend`` and ``scale`` have float64 shape ``[T+1]``.
    Index 0 is the explicit initial state. ``regime_sequence``,
    ``trend_innovations``, and ``scale_innovations`` have shape ``[T]``; transition
    ``t`` therefore obeys ``C[t+1] = coef[regime_sequence[t]] * C[t] +
    innovation[t]``. Coefficient arrays have matching float64 shape ``[R]``.

    Every array is validated, copied, and made read-only. The two state arrays may
    contain any finite real values, including non-positive scale states. This record
    contains no random generator and constructing it consumes no random state.
    """

    trend: NDArray[np.float64]
    scale: NDArray[np.float64]
    trend_innovations: NDArray[np.float64]
    scale_innovations: NDArray[np.float64]
    regime_sequence: NDArray[np.int64]
    trend_ar_coefficients: NDArray[np.float64]
    scale_ar_coefficients: NDArray[np.float64]
    initial_trend: float
    initial_scale: float

    def __post_init__(self) -> None:
        trend = _copy_float64_vector(self.trend, field_name="trend")
        scale = _copy_float64_vector(self.scale, field_name="scale")
        if trend.shape != scale.shape:
            raise ValueError(
                "trend and scale: expected matching state shape [T+1], "
                f"got {trend.shape} and {scale.shape}"
            )
        if trend.size < 2:
            raise ValueError(
                f"trend and scale: expected state shape [T+1] with T >= 1, got {trend.shape}"
            )

        number_of_transitions = trend.size - 1
        trend_innovations = _copy_float64_vector(
            self.trend_innovations,
            field_name="trend_innovations",
            expected_shape=(number_of_transitions,),
        )
        scale_innovations = _copy_float64_vector(
            self.scale_innovations,
            field_name="scale_innovations",
            expected_shape=(number_of_transitions,),
        )
        regime_sequence = _copy_regime_sequence(
            self.regime_sequence,
            expected_shape=(number_of_transitions,),
        )
        trend_coefficients, scale_coefficients = _copy_and_validate_coefficients(
            self.trend_ar_coefficients,
            self.scale_ar_coefficients,
        )
        _validate_regime_labels(
            regime_sequence,
            number_of_regimes=trend_coefficients.size,
        )

        initial_trend = _finite_real(self.initial_trend, field_name="initial_trend")
        initial_scale = _finite_real(self.initial_scale, field_name="initial_scale")
        if initial_trend != float(trend[0]):
            raise ValueError(
                "initial_trend: expected to equal trend[0], "
                f"got {initial_trend} and {float(trend[0])}"
            )
        if initial_scale != float(scale[0]):
            raise ValueError(
                "initial_scale: expected to equal scale[0], "
                f"got {initial_scale} and {float(scale[0])}"
            )

        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "trend_innovations", trend_innovations)
        object.__setattr__(self, "scale_innovations", scale_innovations)
        object.__setattr__(self, "regime_sequence", regime_sequence)
        object.__setattr__(self, "trend_ar_coefficients", trend_coefficients)
        object.__setattr__(self, "scale_ar_coefficients", scale_coefficients)
        object.__setattr__(self, "initial_trend", initial_trend)
        object.__setattr__(self, "initial_scale", initial_scale)


def generate_latent_concepts(
    regime_sequence: NDArray[np.int64],
    trend_ar_coefficients: NDArray[np.float64],
    scale_ar_coefficients: NDArray[np.float64],
    trend_innovations: NDArray[np.float64],
    scale_innovations: NDArray[np.float64],
    initial_trend: float,
    initial_scale: float,
) -> LatentConceptPath:
    """Generate isolated trend and scale paths from supplied deterministic inputs.

    Args:
        regime_sequence: Int64 regime labels with shape ``[T]``. Label at index
            ``t`` selects the coefficient used to produce state index ``t+1``.
        trend_ar_coefficients: Float64 coefficients with shape ``[R]`` and strict
            absolute value below one.
        scale_ar_coefficients: Float64 coefficients with the same shape ``[R]`` and
            strict absolute value below one.
        trend_innovations: Finite float64 innovations with shape ``[T]``. These are
            read only by the trend recurrence.
        scale_innovations: Finite float64 innovations with shape ``[T]``. These are
            read only by the scale recurrence.
        initial_trend: Finite real value stored at ``trend[0]``.
        initial_scale: Finite real value stored at ``scale[0]``; it need not be
            positive.

    Returns:
        A frozen path with float64 state shape ``[T+1]`` and read-only copied arrays.

    Raises:
        TypeError: If an array/dtype or scalar type is invalid.
        ValueError: If shapes, labels, coefficients, or finite-value rules fail.

    Randomness:
        None. The function consumes only the supplied arrays and never samples or
        reads global RNG state. Concept overlap must be expressed downstream through
        observation loadings, never by cross-reading the other latent state.
    """

    regimes = _copy_regime_sequence(regime_sequence)
    trend_coefficients, scale_coefficients = _copy_and_validate_coefficients(
        trend_ar_coefficients,
        scale_ar_coefficients,
    )
    _validate_regime_labels(regimes, number_of_regimes=trend_coefficients.size)
    number_of_transitions = regimes.size
    trend_noise = _copy_float64_vector(
        trend_innovations,
        field_name="trend_innovations",
        expected_shape=(number_of_transitions,),
    )
    scale_noise = _copy_float64_vector(
        scale_innovations,
        field_name="scale_innovations",
        expected_shape=(number_of_transitions,),
    )
    trend_initial = _finite_real(initial_trend, field_name="initial_trend")
    scale_initial = _finite_real(initial_scale, field_name="initial_scale")

    trend = _rollout_ar_path(
        regimes,
        trend_coefficients,
        trend_noise,
        initial_state=trend_initial,
        field_name="trend",
    )
    scale = _rollout_ar_path(
        regimes,
        scale_coefficients,
        scale_noise,
        initial_state=scale_initial,
        field_name="scale",
    )
    return LatentConceptPath(
        trend=trend,
        scale=scale,
        trend_innovations=trend_noise,
        scale_innovations=scale_noise,
        regime_sequence=regimes,
        trend_ar_coefficients=trend_coefficients,
        scale_ar_coefficients=scale_coefficients,
        initial_trend=trend_initial,
        initial_scale=scale_initial,
    )


def replace_concept_at_origin(
    path: LatentConceptPath,
    *,
    concept: ConceptName,
    origin_index: int,
    source_value: float,
) -> LatentConceptPath:
    """Replace one current concept value and naturally replay only its future.

    ``origin_index`` indexes the state arrays (shape ``[T+1]``) and must lie in
    ``[0, T-1]`` so a future transition exists. Only ``path.<concept>[origin_index]``
    is replaced. For every ``t >= origin_index``, the selected path is then replayed
    with the same ``regime_sequence[t]`` and the same selected innovation ``[t]``.
    The other concept and both innovation arrays remain bitwise unchanged; no source
    future path is accepted or copied.

    Args:
        path: Frozen base path to replay.
        concept: Exactly ``"trend"`` or ``"scale"``.
        origin_index: Python/NumPy integer index into the current state path.
        source_value: Finite real value to install at the prediction origin.

    Returns:
        A new frozen, read-only ``LatentConceptPath``. Replacing with the existing
        base value produces bitwise-identical arrays.

    Raises:
        TypeError: If ``path``, ``origin_index``, or ``source_value`` has a wrong type.
        ValueError: If the concept, index, value, or replay result is invalid.

    Randomness:
        None. Future regimes and innovations are reused exactly as stored in ``path``.
    """

    if not isinstance(path, LatentConceptPath):
        raise TypeError("path: expected a LatentConceptPath")
    if concept not in ("trend", "scale"):
        raise ValueError(f"concept: expected 'trend' or 'scale', got {concept!r}")
    if isinstance(origin_index, bool) or not isinstance(origin_index, int | np.integer):
        raise TypeError("origin_index: expected an integer state index")
    state_index = int(origin_index)
    number_of_transitions = path.regime_sequence.size
    if state_index < 0 or state_index >= number_of_transitions:
        raise ValueError(
            "origin_index: expected a state index in "
            f"[0, {number_of_transitions - 1}], got {state_index}"
        )
    replacement = _finite_real(source_value, field_name="source_value")

    trend = path.trend.copy()
    scale = path.scale.copy()
    target = trend if concept == "trend" else scale
    coefficients = path.trend_ar_coefficients if concept == "trend" else path.scale_ar_coefficients
    innovations = path.trend_innovations if concept == "trend" else path.scale_innovations
    target[state_index] = replacement
    _replay_future_in_place(
        target,
        path.regime_sequence,
        coefficients,
        innovations,
        origin_index=state_index,
        field_name=concept,
    )

    return LatentConceptPath(
        trend=trend,
        scale=scale,
        trend_innovations=path.trend_innovations,
        scale_innovations=path.scale_innovations,
        regime_sequence=path.regime_sequence,
        trend_ar_coefficients=path.trend_ar_coefficients,
        scale_ar_coefficients=path.scale_ar_coefficients,
        initial_trend=(
            replacement if concept == "trend" and state_index == 0 else path.initial_trend
        ),
        initial_scale=(
            replacement if concept == "scale" and state_index == 0 else path.initial_scale
        ),
    )


def scale_function(
    scale_state: NDArray[np.float64],
    floor: float,
    loading: ScaleParameter,
    *,
    base_log_scale: ScaleParameter = 0.0,
) -> NDArray[np.float64]:
    """Map finite real scale states to strictly positive observation-noise scales.

    The response is exactly ``softplus(base_log_scale + loading * scale_state) +
    floor``, evaluated stably with ``logaddexp``. ``scale_state`` is a non-empty
    float64 array of any shape. ``loading`` and ``base_log_scale`` may each be a
    finite real scalar or a float64 array with exactly the same shape; implicit
    non-scalar broadcasting is rejected. ``floor`` must be a finite real scalar
    strictly greater than zero.

    The returned float64 array preserves ``scale_state.shape``, is finite, strictly
    positive, independently owned, and read-only. Inputs are never modified and no
    random state is accessed.
    """

    state = _require_float64_array(scale_state, field_name="scale_state")
    if state.size == 0:
        raise ValueError(f"scale_state: expected a non-empty array, got shape {state.shape}")
    floor_value = _finite_real(floor, field_name="floor")
    if floor_value <= 0.0:
        raise ValueError(f"floor: expected a strictly positive finite value, got {floor_value}")
    loading_value = _scale_parameter(
        loading,
        field_name="loading",
        expected_shape=state.shape,
    )
    base_value = _scale_parameter(
        base_log_scale,
        field_name="base_log_scale",
        expected_shape=state.shape,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        linear_response = base_value + loading_value * state
    if not np.all(np.isfinite(linear_response)):
        raise ValueError("scale response: values must be finite before softplus")
    response = np.asarray(
        np.logaddexp(0.0, linear_response) + floor_value,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(response)) or not np.all(response > 0.0):
        raise ValueError("scale response: values must be strictly positive and finite")
    return _read_only_copy(response)


def _rollout_ar_path(
    regimes: NDArray[np.int64],
    coefficients: NDArray[np.float64],
    innovations: NDArray[np.float64],
    *,
    initial_state: float,
    field_name: str,
) -> NDArray[np.float64]:
    states = np.empty(regimes.size + 1, dtype=np.float64)
    states[0] = initial_state
    _replay_future_in_place(
        states,
        regimes,
        coefficients,
        innovations,
        origin_index=0,
        field_name=field_name,
    )
    return states


def _replay_future_in_place(
    states: NDArray[np.float64],
    regimes: NDArray[np.int64],
    coefficients: NDArray[np.float64],
    innovations: NDArray[np.float64],
    *,
    origin_index: int,
    field_name: str,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        for time_index in range(origin_index, regimes.size):
            regime = int(regimes[time_index])
            states[time_index + 1] = (
                coefficients[regime] * states[time_index] + innovations[time_index]
            )
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{field_name}: replayed state values must be finite")


def _copy_and_validate_coefficients(
    trend_ar_coefficients: object,
    scale_ar_coefficients: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    trend_coefficients = _copy_float64_vector(
        trend_ar_coefficients,
        field_name="trend_ar_coefficients",
    )
    scale_coefficients = _copy_float64_vector(
        scale_ar_coefficients,
        field_name="scale_ar_coefficients",
    )
    if trend_coefficients.shape != scale_coefficients.shape:
        raise ValueError(
            "trend_ar_coefficients and scale_ar_coefficients: expected matching "
            f"shape [R], got {trend_coefficients.shape} and {scale_coefficients.shape}"
        )
    for field_name, coefficients in (
        ("trend_ar_coefficients", trend_coefficients),
        ("scale_ar_coefficients", scale_coefficients),
    ):
        if np.any(np.abs(coefficients) >= 1.0):
            raise ValueError(f"{field_name}: every coefficient must satisfy strict |AR| < 1")
    return trend_coefficients, scale_coefficients


def _validate_regime_labels(
    regime_sequence: NDArray[np.int64],
    *,
    number_of_regimes: int,
) -> None:
    if np.any(regime_sequence < 0) or np.any(regime_sequence >= number_of_regimes):
        raise ValueError(
            f"regime_sequence: every label must index coefficients in [0, {number_of_regimes - 1}]"
        )


def _copy_regime_sequence(
    value: object,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise TypeError("regime_sequence: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"regime_sequence: expected dtype int64, got {value.dtype}")
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"regime_sequence: expected non-empty shape [T], got shape {value.shape}")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(
            f"regime_sequence: expected shape {expected_shape}, got shape {value.shape}"
        )
    return _read_only_copy(value)


def _copy_float64_vector(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    array = _require_float64_array(value, field_name=field_name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field_name}: expected non-empty shape [N], got shape {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{field_name}: expected shape {expected_shape}, got shape {array.shape}")
    return _read_only_copy(array)


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


def _scale_parameter(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, ...],
) -> float | NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        array = _require_float64_array(value, field_name=field_name)
        if array.shape != expected_shape:
            raise ValueError(
                f"{field_name}: expected scalar or shape {expected_shape}, got shape {array.shape}"
            )
        return array
    return _finite_real(value, field_name=field_name)


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise TypeError(f"{field_name}: expected a finite real scalar")
    real_value = float(value)
    if not np.isfinite(real_value):
        raise ValueError(f"{field_name}: expected a finite real scalar, got {real_value}")
    return real_value


def _read_only_copy(array: NDArray[np.generic]) -> NDArray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result
