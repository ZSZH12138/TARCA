from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.data.synthetic.missingness as missingness_module  # noqa: E402
from tarca.contracts import WindowBatch  # noqa: E402
from tarca.data.synthetic.missingness import (  # noqa: E402
    apply_observation_mask,
    generate_missing_mask,
)


def test_missingness_public_functions_are_importable() -> None:
    assert callable(generate_missing_mask)
    assert callable(apply_observation_mask)


def _mcar_mask(
    uniforms: object,
    *,
    shape: object = (2, 3),
    rate: object = 0.5,
    block_starts: object = None,
    block_lengths: object = None,
) -> np.ndarray:
    return generate_missing_mask(
        "mcar",
        shape,  # type: ignore[arg-type]
        uniforms,  # type: ignore[arg-type]
        block_starts,  # type: ignore[arg-type]
        block_lengths,  # type: ignore[arg-type]
        rate,  # type: ignore[arg-type]
    )


def _block_mask(
    block_starts: object,
    block_lengths: object,
    *,
    shape: object = (6, 3),
    uniforms: object = None,
    rate: object = 0.25,
) -> np.ndarray:
    return generate_missing_mask(
        "block",
        shape,  # type: ignore[arg-type]
        uniforms,  # type: ignore[arg-type]
        block_starts,  # type: ignore[arg-type]
        block_lengths,  # type: ignore[arg-type]
        rate,  # type: ignore[arg-type]
    )


def test_none_returns_a_fully_observed_read_only_mask_without_random_arrays() -> None:
    mask = generate_missing_mask(
        "none",
        (2, 3, 4),
        None,
        None,
        None,
        0.0,
    )

    assert mask.shape == (2, 3, 4)
    assert mask.dtype == np.bool_
    assert np.all(mask)
    assert not mask.flags.writeable


@pytest.mark.parametrize(
    ("uniforms", "block_starts", "block_lengths", "rate", "message"),
    [
        (np.zeros((2, 2), dtype=np.float64), None, None, 0.0, "uniforms"),
        (None, np.zeros(0, dtype=np.int64), None, 0.0, "block_starts"),
        (None, None, np.zeros(0, dtype=np.int64), 0.0, "block_lengths"),
        (None, None, None, 0.1, "rate"),
    ],
)
def test_none_rejects_contradictory_stochastic_configuration(
    uniforms: object,
    block_starts: object,
    block_lengths: object,
    rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_missing_mask(
            "none",
            (2, 2),
            uniforms,  # type: ignore[arg-type]
            block_starts,  # type: ignore[arg-type]
            block_lengths,  # type: ignore[arg-type]
            rate,
        )


@pytest.mark.parametrize("mode", ["MCAR", "Block", "NONE", "unknown", "", None, 1])
def test_generate_missing_mask_rejects_unknown_or_noncanonical_modes(mode: object) -> None:
    with pytest.raises((TypeError, ValueError), match="mode"):
        generate_missing_mask(
            mode,  # type: ignore[arg-type]
            (2, 2),
            None,
            None,
            None,
            0.0,
        )


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3],
        (),
        (0, 3),
        (-1, 3),
        (2.0, 3),
        (True, 3),
        "2x3",
    ],
)
def test_generate_missing_mask_rejects_invalid_shape(shape: object) -> None:
    with pytest.raises((TypeError, ValueError), match="shape"):
        generate_missing_mask(
            "none",
            shape,  # type: ignore[arg-type]
            None,
            None,
            None,
            0.0,
        )


def test_generate_missing_mask_accepts_numpy_integer_shape_dimensions() -> None:
    mask = generate_missing_mask(
        "none",
        (np.int64(2), np.int32(3)),
        None,
        None,
        None,
        0.0,
    )

    assert mask.shape == (2, 3)


@pytest.mark.parametrize("rate", [True, "0.1", np.nan, np.inf, -0.01, 1.01])
def test_generate_missing_mask_rejects_invalid_rates(rate: object) -> None:
    with pytest.raises((TypeError, ValueError), match="rate"):
        _mcar_mask(
            np.full((2, 3), 0.5, dtype=np.float64),
            rate=rate,
        )


def test_mcar_uses_elementwise_uniform_threshold_with_true_meaning_observed() -> None:
    uniforms = np.array(
        [[0.0, 0.49, 0.5], [0.75, 0.25, np.nextafter(1.0, 0.0)]],
        dtype=np.float64,
    )
    before = uniforms.copy()

    mask = _mcar_mask(uniforms)

    expected = np.array(
        [[False, False, True], [True, False, True]],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(uniforms, before)
    assert mask.shape == uniforms.shape
    assert mask.dtype == np.bool_
    assert not mask.flags.writeable
    assert not np.shares_memory(mask, uniforms)


def test_mcar_is_deterministic_for_identical_supplied_uniforms() -> None:
    uniforms = np.array(
        [[0.01, 0.2, 0.9], [0.7, 0.15, 0.3]],
        dtype=np.float64,
    )

    first = _mcar_mask(uniforms, rate=0.2)
    second = _mcar_mask(uniforms.copy(), rate=0.2)

    assert first.tobytes() == second.tobytes()


def test_mcar_realized_rate_matches_an_even_explicit_grid() -> None:
    sample_count = 20_000
    uniforms = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count

    mask = _mcar_mask(
        uniforms.reshape(200, 100),
        shape=(200, 100),
        rate=0.15,
    )

    assert float(np.mean(~mask)) == pytest.approx(0.15, abs=1.0 / sample_count)


@pytest.mark.parametrize(
    ("uniforms", "message"),
    [
        (None, "uniforms"),
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "uniforms"),
        (np.zeros((2, 3), dtype=np.float32), "uniforms"),
        (np.zeros((3, 2), dtype=np.float64), "shape"),
        (np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5]]), "uniforms"),
        (np.array([[0.1, 0.2, -0.1], [0.3, 0.4, 0.5]]), "uniforms"),
        (np.array([[0.1, 0.2, 1.0], [0.3, 0.4, 0.5]]), "uniforms"),
    ],
)
def test_mcar_rejects_invalid_uniform_arrays(uniforms: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _mcar_mask(uniforms)


@pytest.mark.parametrize(
    ("block_starts", "block_lengths", "message"),
    [
        (np.zeros(0, dtype=np.int64), None, "block_starts"),
        (None, np.zeros(0, dtype=np.int64), "block_lengths"),
    ],
)
def test_mcar_rejects_block_event_arrays(
    block_starts: object,
    block_lengths: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _mcar_mask(
            np.full((2, 3), 0.5, dtype=np.float64),
            block_starts=block_starts,
            block_lengths=block_lengths,
        )


def test_mcar_prefix_is_unchanged_when_only_future_uniforms_change() -> None:
    prefix_length = 4
    base_uniforms = np.full((8, 2), 0.9, dtype=np.float64)
    changed_uniforms = base_uniforms.copy()
    changed_uniforms[prefix_length:] = 0.1

    base = _mcar_mask(base_uniforms, shape=(8, 2), rate=0.5)
    changed = _mcar_mask(changed_uniforms, shape=(8, 2), rate=0.5)

    np.testing.assert_array_equal(base[:prefix_length], changed[:prefix_length])
    assert not np.array_equal(base[prefix_length:], changed[prefix_length:])


def test_block_events_mask_whole_feature_vectors_and_clip_only_at_right_boundary() -> None:
    starts = np.array([1, 4], dtype=np.int64)
    lengths = np.array([2, 99], dtype=np.int64)
    starts_before = starts.copy()
    lengths_before = lengths.copy()

    mask = _block_mask(starts, lengths, rate=0.73)

    expected = np.array(
        [
            [True, True, True],
            [False, False, False],
            [False, False, False],
            [True, True, True],
            [False, False, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(starts, starts_before)
    np.testing.assert_array_equal(lengths, lengths_before)
    assert not mask.flags.writeable


def test_block_events_broadcast_over_every_axis_after_time() -> None:
    mask = _block_mask(
        np.array([2], dtype=np.int64),
        np.array([1], dtype=np.int64),
        shape=(5, 2, 3),
    )

    assert np.all(mask[:2])
    assert not np.any(mask[2])
    assert np.all(mask[3:])


def test_empty_block_event_arrays_produce_a_fully_observed_mask() -> None:
    mask = _block_mask(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        shape=(4,),
        rate=1.0,
    )

    np.testing.assert_array_equal(mask, np.ones(4, dtype=np.bool_))


def test_block_rate_is_validated_but_explicit_events_make_it_inactive() -> None:
    starts = np.array([1], dtype=np.int64)
    lengths = np.array([2], dtype=np.int64)

    rate_zero = _block_mask(starts, lengths, rate=0.0)
    rate_one = _block_mask(starts, lengths, rate=1.0)

    assert rate_zero.tobytes() == rate_one.tobytes()


@pytest.mark.parametrize(
    ("starts", "lengths", "error_type", "message"),
    [
        (None, np.ones(1, dtype=np.int64), ValueError, "block_starts"),
        (np.zeros(1, dtype=np.int64), None, ValueError, "block_lengths"),
        ([0], np.ones(1, dtype=np.int64), TypeError, "block_starts"),
        (np.zeros(1, dtype=np.int32), np.ones(1, dtype=np.int64), TypeError, "int64"),
        (np.zeros((1, 1), dtype=np.int64), np.ones(1, dtype=np.int64), ValueError, "shape"),
        (np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.float64), TypeError, "int64"),
        (np.zeros(1, dtype=np.int64), np.ones((1, 1), dtype=np.int64), ValueError, "shape"),
        (np.array([0, 1], dtype=np.int64), np.ones(1, dtype=np.int64), ValueError, "same"),
        (np.array([-1], dtype=np.int64), np.ones(1, dtype=np.int64), ValueError, "start"),
        (np.array([6], dtype=np.int64), np.ones(1, dtype=np.int64), ValueError, "start"),
        (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), ValueError, "positive"),
        (np.zeros(1, dtype=np.int64), -np.ones(1, dtype=np.int64), ValueError, "positive"),
    ],
)
def test_block_rejects_invalid_event_encoding(
    starts: object,
    lengths: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _block_mask(starts, lengths)


def test_block_rejects_uniforms_because_events_already_encode_randomness() -> None:
    with pytest.raises(ValueError, match="uniforms"):
        _block_mask(
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            uniforms=np.array([0.1], dtype=np.float64),
        )


def test_block_prefix_is_unchanged_when_only_future_events_change() -> None:
    prefix_length = 4
    base = _block_mask(
        np.array([1, 5], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
        shape=(8, 2),
    )
    changed = _block_mask(
        np.array([1, 6], dtype=np.int64),
        np.array([2, 2], dtype=np.int64),
        shape=(8, 2),
    )

    np.testing.assert_array_equal(base[:prefix_length], changed[:prefix_length])
    assert not np.array_equal(base[prefix_length:], changed[prefix_length:])


def test_apply_observation_mask_preserves_observed_values_and_fills_missing_values() -> None:
    values = np.array(
        [[1.5, -0.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=np.float64,
    )
    mask = np.array(
        [[True, False, True], [False, True, False]],
        dtype=np.bool_,
    )
    values_before = values.copy()
    mask_before = mask.copy()

    masked = apply_observation_mask(values, mask, fill_value=-7.5)

    expected = np.array(
        [[1.5, -7.5, 3.0], [-7.5, 5.0, -7.5]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert masked.shape == values.shape
    assert masked.dtype == np.float64
    assert np.all(np.isfinite(masked))
    assert not masked.flags.writeable
    assert not np.shares_memory(masked, values)
    assert not np.shares_memory(masked, mask)


def test_apply_observation_mask_uses_finite_zero_by_default() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([False, True, False], dtype=np.bool_)

    masked = apply_observation_mask(values, mask)

    np.testing.assert_array_equal(masked, np.array([0.0, 2.0, 0.0]))


@pytest.mark.parametrize(
    ("values", "error_type", "message"),
    [
        ([1.0], TypeError, "values"),
        (np.array([1.0], dtype=np.float32), TypeError, "float64"),
        (np.array(1.0, dtype=np.float64), ValueError, "rank"),
        (np.empty(0, dtype=np.float64), ValueError, "non-empty"),
        (np.array([np.nan], dtype=np.float64), ValueError, "finite"),
        (np.array([np.inf], dtype=np.float64), ValueError, "finite"),
    ],
)
def test_apply_observation_mask_rejects_invalid_complete_values(
    values: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        apply_observation_mask(
            values,  # type: ignore[arg-type]
            np.ones(1, dtype=np.bool_),
        )


@pytest.mark.parametrize(
    ("mask", "error_type", "message"),
    [
        ([True, False], TypeError, "mask"),
        (np.ones(2, dtype=np.int64), TypeError, "bool"),
        (np.ones((2, 1), dtype=np.bool_), ValueError, "shape"),
    ],
)
def test_apply_observation_mask_rejects_invalid_masks(
    mask: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        apply_observation_mask(
            np.array([1.0, 2.0], dtype=np.float64),
            mask,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("fill_value", [True, "0", np.nan, np.inf, -np.inf, np.array(0.0)])
def test_apply_observation_mask_rejects_nonfinite_or_nonscalar_fill(
    fill_value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="fill_value"):
        apply_observation_mask(
            np.array([1.0], dtype=np.float64),
            np.array([False], dtype=np.bool_),
            fill_value=fill_value,  # type: ignore[arg-type]
        )


def test_masked_values_and_observed_mask_satisfy_window_batch_contract() -> None:
    complete = np.array(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=np.float64,
    )
    uniforms = np.array(
        [[[0.9, 0.1], [0.2, 0.8]]],
        dtype=np.float64,
    )
    mask = _mcar_mask(uniforms, shape=complete.shape, rate=0.5)
    masked = apply_observation_mask(complete, mask)
    feature_start = datetime(2026, 1, 1, tzinfo=UTC)
    feature_end = feature_start + timedelta(hours=1)
    prediction_start = feature_end + timedelta(hours=1)

    batch = WindowBatch(
        x=torch.from_numpy(masked.copy()),
        y=None,
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=torch.from_numpy(mask.copy()),
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=("masked-window",),
        input_feature_names=("x0", "x1"),
        target_names=(),
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=(feature_start,),
        feature_end=(feature_end,),
        prediction_start=(prediction_start,),
        label_end=(prediction_start,),
        forecast_time=((prediction_start,),),
        metadata={"missingness": "mcar"},
    )

    assert bool(torch.isfinite(batch.x).all())
    assert torch.equal(batch.x_observed_mask, torch.from_numpy(mask.copy()))
    assert torch.equal(batch.x[~batch.x_observed_mask], torch.zeros(2))


class _ForbiddenRandomAccess:
    def __getattr__(self, attribute_name: str) -> object:
        raise AssertionError(f"missingness code accessed RNG attribute {attribute_name!r}")


def test_missingness_operations_never_access_numpy_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(missingness_module.np, "random", _ForbiddenRandomAccess())
    uniforms = np.array([[0.1, 0.9]], dtype=np.float64)
    mcar = _mcar_mask(uniforms, shape=(1, 2), rate=0.5)
    block = _block_mask(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        shape=(1, 2),
    )
    none = generate_missing_mask("none", (1, 2), None, None, None, 0.0)

    apply_observation_mask(np.ones((1, 2), dtype=np.float64), mcar)
    assert np.array_equal(block, np.zeros((1, 2), dtype=np.bool_))
    assert np.array_equal(none, np.ones((1, 2), dtype=np.bool_))
