"""Future-blind missingness over explicitly supplied stochastic arrays.

``True`` always means observed and ``False`` always means missing, matching
``WindowBatch.*_observed_mask``. This module has no random-number generator:
callers pre-generate either elementwise MCAR uniforms or complete block events.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

_MODES: Final = ("none", "mcar", "block")


def generate_missing_mask(
    mode: str,
    shape: tuple[int, ...],
    uniforms: NDArray[np.float64] | None,
    block_starts: NDArray[np.int64] | None,
    block_lengths: NDArray[np.int64] | None,
    rate: float,
) -> NDArray[np.bool_]:
    """Generate a read-only boolean observation mask without sampling.

    Args:
        mode: Exactly one canonical lowercase value: ``"none"``, ``"mcar"``,
            or ``"block"``. Case drift and unknown modes fail.
        shape: Non-empty tuple of positive integer dimensions. Axis 0 is causal
            time; every remaining axis is an aligned feature/value axis.
        uniforms: For MCAR, a finite float64 array with exactly ``shape`` and
            values in ``[0, 1)``. An element is observed exactly when
            ``uniforms >= rate``. It must be ``None`` for other modes.
        block_starts: For block mode, an int64 vector ``[E]`` of event start
            indices on time axis 0. Each event is a whole-feature-vector outage
            and therefore broadcasts over every remaining axis. Starts must lie
            in ``[0, shape[0])``. It must be ``None`` outside block mode.
        block_lengths: For block mode, an int64 vector ``[E]`` aligned one-to-one
            with ``block_starts``. Lengths are strictly positive. Event ``e``
            masks ``[start_e:min(start_e + length_e, shape[0]), ...]``; only the
            right time boundary is clipped. It must be ``None`` otherwise.
        rate: Finite probability in ``[0, 1]``. It is the MCAR elementwise
            missing probability. For block mode it is validated but inactive
            because the pre-generated starts and lengths already encode the
            stochastic realization. For ``none`` it must equal zero.

    Returns:
        An independently owned, read-only ``numpy.bool_`` array with ``shape``.
        ``True`` means observed and ``False`` means missing.

    Raises:
        TypeError: If scalar, array, shape, or dtype contracts are violated.
        ValueError: If modes, ranges, shapes, or mode-specific inputs are invalid.

    Causality:
        The function accepts no observations or labels. MCAR positions depend only
        on their same-time supplied uniform. A block affects only indices at or
        after its supplied start, so changing events that start in the future
        cannot alter an already generated prefix.
    """

    selected_mode = _validate_mode(mode)
    output_shape = _validate_shape(shape)
    missing_rate = _validate_rate(rate)

    if selected_mode == "none":
        _require_absent(uniforms, field_name="uniforms", mode=selected_mode)
        _require_absent(block_starts, field_name="block_starts", mode=selected_mode)
        _require_absent(block_lengths, field_name="block_lengths", mode=selected_mode)
        if missing_rate != 0.0:
            raise ValueError(f"rate: expected 0 for mode 'none', got {missing_rate}")
        return _read_only_copy(np.ones(output_shape, dtype=np.bool_))

    if selected_mode == "mcar":
        _require_absent(block_starts, field_name="block_starts", mode=selected_mode)
        _require_absent(block_lengths, field_name="block_lengths", mode=selected_mode)
        supplied_uniforms = _require_float64_array(uniforms, field_name="uniforms")
        if supplied_uniforms.shape != output_shape:
            raise ValueError(
                f"uniforms: expected shape {output_shape}, got shape {supplied_uniforms.shape}"
            )
        if np.any(supplied_uniforms < 0.0) or np.any(supplied_uniforms >= 1.0):
            raise ValueError("uniforms: every value must lie in [0, 1)")
        return _read_only_copy(np.asarray(supplied_uniforms >= missing_rate, dtype=np.bool_))

    _require_absent(uniforms, field_name="uniforms", mode=selected_mode)
    starts = _require_block_events(block_starts, field_name="block_starts")
    lengths = _require_block_events(block_lengths, field_name="block_lengths")
    if starts.shape != lengths.shape:
        raise ValueError(
            "block_starts and block_lengths: expected the same shape [E], "
            f"got {starts.shape} and {lengths.shape}"
        )
    time_length = output_shape[0]
    if np.any(starts < 0) or np.any(starts >= time_length):
        raise ValueError(
            f"block_starts: every start must lie in [0, {time_length - 1}], got {starts.tolist()}"
        )
    if np.any(lengths <= 0):
        raise ValueError(
            f"block_lengths: every length must be strictly positive, got {lengths.tolist()}"
        )

    mask = np.ones(output_shape, dtype=np.bool_)
    for start_value, length_value in zip(starts, lengths, strict=True):
        start = int(start_value)
        stop = min(start + int(length_value), time_length)
        mask[start:stop, ...] = False
    return _read_only_copy(mask)


def apply_observation_mask(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    fill_value: float = 0.0,
) -> NDArray[np.float64]:
    """Apply ``True``-means-observed semantics without modifying complete truth.

    ``values`` must be a non-empty, finite float64 NumPy array of rank at least
    one. ``mask`` must be a NumPy bool array with exactly the same shape.
    ``fill_value`` must be a finite real scalar; the default ``0.0`` is the
    standardized finite fill required by the Stage 1 ``WindowBatch`` mapping.

    The result is an independently owned, read-only float64 copy. Observed values
    are preserved and exactly the positions selected by ``~mask`` are replaced.
    NaN and infinity are rejected even at missing positions because ``WindowBatch``
    validates every numeric value before considering its mask.
    """

    complete = _require_float64_array(values, field_name="values")
    if complete.ndim == 0:
        raise ValueError("values: expected rank at least 1")
    if complete.size == 0:
        raise ValueError(f"values: expected a non-empty array, got shape {complete.shape}")
    observed = _require_bool_array(mask, field_name="mask")
    if observed.shape != complete.shape:
        raise ValueError(f"mask: expected shape {complete.shape}, got shape {observed.shape}")
    fill = _finite_real(fill_value, field_name="fill_value")

    result = np.array(complete, dtype=np.float64, copy=True, order="C")
    result[~observed] = fill
    return _read_only_copy(result)


def _validate_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("mode: expected one of 'none', 'mcar', or 'block'")
    if value not in _MODES:
        raise ValueError(f"mode: expected one of 'none', 'mcar', or 'block'; got {value!r}")
    return value


def _validate_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("shape: expected a tuple of positive integers")
    if not value:
        raise ValueError("shape: expected rank at least 1")
    dimensions: list[int] = []
    for axis, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int | np.integer):
            raise TypeError(f"shape[{axis}]: expected a positive integer")
        integer_dimension = int(dimension)
        if integer_dimension <= 0:
            raise ValueError(f"shape[{axis}]: expected a positive integer, got {integer_dimension}")
        dimensions.append(integer_dimension)
    return tuple(dimensions)


def _validate_rate(value: object) -> float:
    result = _finite_real(value, field_name="rate")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"rate: expected a value in [0, 1], got {result}")
    return result


def _require_absent(value: object, *, field_name: str, mode: str) -> None:
    if value is not None:
        raise ValueError(f"{field_name}: expected None for mode {mode!r}")


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


def _require_block_events(
    value: object,
    *,
    field_name: str,
) -> NDArray[np.int64]:
    if value is None:
        raise ValueError(f"{field_name}: required for mode 'block'")
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype int64")
    if value.dtype != np.dtype(np.int64):
        raise TypeError(f"{field_name}: expected dtype int64, got {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{field_name}: expected shape [E], got shape {value.shape}")
    return value


def _require_bool_array(
    value: object,
    *,
    field_name: str,
) -> NDArray[np.bool_]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected a numpy.ndarray with dtype bool")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError(f"{field_name}: expected dtype bool, got {value.dtype}")
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
