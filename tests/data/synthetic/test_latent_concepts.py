from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.data.synthetic.latent_concepts as latent_concepts_module  # noqa: E402
from tarca.data.synthetic.latent_concepts import (  # noqa: E402
    LatentConceptPath,
    generate_latent_concepts,
    replace_concept_at_origin,
    scale_function,
)

PATH_ARRAY_FIELDS = (
    "trend",
    "scale",
    "trend_innovations",
    "scale_innovations",
    "regime_sequence",
    "trend_ar_coefficients",
    "scale_ar_coefficients",
)


def _latent_inputs() -> dict[str, object]:
    return {
        "regime_sequence": np.array([0, 1, 0], dtype=np.int64),
        "trend_ar_coefficients": np.array([0.5, -0.25], dtype=np.float64),
        "scale_ar_coefficients": np.array([-0.5, 0.25], dtype=np.float64),
        "trend_innovations": np.array([1.0, 2.0, 4.0], dtype=np.float64),
        "scale_innovations": np.array([-1.0, 0.5, -2.0], dtype=np.float64),
        "initial_trend": 3.0,
        "initial_scale": -3.0,
    }


def _generate_path() -> LatentConceptPath:
    return generate_latent_concepts(**_latent_inputs())  # type: ignore[arg-type]


def test_generate_latent_concepts_locks_shape_index_and_regime_specific_ar() -> None:
    inputs = _latent_inputs()
    array_snapshots = {
        name: value.copy() for name, value in inputs.items() if isinstance(value, np.ndarray)
    }

    path = generate_latent_concepts(**inputs)  # type: ignore[arg-type]

    assert np.array_equal(
        path.trend,
        np.array([3.0, 2.5, 1.375, 4.6875], dtype=np.float64),
    )
    assert np.array_equal(
        path.scale,
        np.array([-3.0, 0.5, 0.625, -2.3125], dtype=np.float64),
    )
    assert path.trend.shape == (4,)
    assert path.scale.shape == (4,)
    assert path.trend_innovations.shape == (3,)
    assert path.scale_innovations.shape == (3,)
    assert path.regime_sequence.shape == (3,)
    assert path.initial_trend == 3.0
    assert path.initial_scale == -3.0
    assert path.trend.dtype == np.float64
    assert path.scale.dtype == np.float64
    assert all(np.all(np.isfinite(array)) for array in (path.trend, path.scale))
    assert np.any(path.scale <= 0.0)
    for name, snapshot in array_snapshots.items():
        assert np.array_equal(inputs[name], snapshot)


def test_latent_path_owns_read_only_copies_and_is_frozen() -> None:
    inputs = _latent_inputs()
    source_arrays = tuple(value for value in inputs.values() if isinstance(value, np.ndarray))

    path = generate_latent_concepts(**inputs)  # type: ignore[arg-type]

    output_arrays = (
        path.trend,
        path.scale,
        path.trend_innovations,
        path.scale_innovations,
        path.regime_sequence,
        path.trend_ar_coefficients,
        path.scale_ar_coefficients,
    )
    assert all(not array.flags.writeable for array in output_arrays)
    assert all(
        not np.shares_memory(output, source) for output in output_arrays for source in source_arrays
    )
    with pytest.raises(ValueError, match="read-only"):
        path.trend[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        path.initial_trend = 0.0  # type: ignore[misc]


def test_direct_latent_path_construction_also_deep_freezes_arrays() -> None:
    generated = _generate_path()
    source_trend = generated.trend.copy()

    reconstructed = LatentConceptPath(
        trend=source_trend,
        scale=generated.scale,
        trend_innovations=generated.trend_innovations,
        scale_innovations=generated.scale_innovations,
        regime_sequence=generated.regime_sequence,
        trend_ar_coefficients=generated.trend_ar_coefficients,
        scale_ar_coefficients=generated.scale_ar_coefficients,
        initial_trend=generated.initial_trend,
        initial_scale=generated.initial_scale,
    )

    assert np.array_equal(reconstructed.trend, source_trend)
    assert not np.shares_memory(reconstructed.trend, source_trend)
    assert not reconstructed.trend.flags.writeable


def test_generation_is_bitwise_deterministic_for_identical_supplied_inputs() -> None:
    first = _generate_path()
    second = _generate_path()

    for field_name in (
        "trend",
        "scale",
        "trend_innovations",
        "scale_innovations",
        "regime_sequence",
        "trend_ar_coefficients",
        "scale_ar_coefficients",
    ):
        assert np.array_equal(
            getattr(first, field_name),
            getattr(second, field_name),
        )


def test_trend_and_scale_states_read_only_their_own_innovations() -> None:
    base_inputs = _latent_inputs()
    changed_trend_inputs = _latent_inputs()
    changed_scale_inputs = _latent_inputs()
    changed_trend_inputs["trend_innovations"] = np.array(
        [9.0, 2.0, 4.0],
        dtype=np.float64,
    )
    changed_scale_inputs["scale_innovations"] = np.array(
        [-1.0, 7.0, -2.0],
        dtype=np.float64,
    )

    base = generate_latent_concepts(**base_inputs)  # type: ignore[arg-type]
    changed_trend = generate_latent_concepts(  # type: ignore[arg-type]
        **changed_trend_inputs
    )
    changed_scale = generate_latent_concepts(  # type: ignore[arg-type]
        **changed_scale_inputs
    )

    assert not np.array_equal(changed_trend.trend, base.trend)
    assert np.array_equal(changed_trend.scale, base.scale)
    assert np.array_equal(changed_trend.scale_innovations, base.scale_innovations)
    assert not np.array_equal(changed_scale.scale, base.scale)
    assert np.array_equal(changed_scale.trend, base.trend)
    assert np.array_equal(changed_scale.trend_innovations, base.trend_innovations)


def test_scale_function_matches_softplus_response_and_keeps_state_unmodified() -> None:
    scale_state = np.array([-2.0, 0.0, 3.0], dtype=np.float64)
    before = scale_state.copy()

    response = scale_function(
        scale_state,
        0.125,
        2.0,
        base_log_scale=-0.5,
    )

    assert np.allclose(
        response,
        np.array(
            [
                0.1360477448485938,
                0.5990769841801067,
                5.629078443270571,
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-15,
    )
    assert response.shape == scale_state.shape
    assert response.dtype == np.float64
    assert np.all(response > 0.0)
    assert np.all(np.isfinite(response))
    assert not response.flags.writeable
    assert np.array_equal(scale_state, before)


def test_scale_function_supports_exact_shape_regime_schedules() -> None:
    scale_state = np.array([-1.0, 2.0], dtype=np.float64)
    loading = np.array([2.0, -0.5], dtype=np.float64)
    base_log_scale = np.array([0.0, 1.0], dtype=np.float64)

    response = scale_function(
        scale_state,
        0.25,
        loading,
        base_log_scale=base_log_scale,
    )

    assert np.allclose(
        response,
        np.array([0.3769280110429725, 0.9431471805599453], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert not response.flags.writeable


def test_scale_function_stays_finite_and_positive_for_extreme_finite_states() -> None:
    response = scale_function(
        np.array([-1e308, 1e308], dtype=np.float64),
        1e-12,
        1.0,
        base_log_scale=0.0,
    )

    assert np.all(response > 0.0)
    assert np.all(np.isfinite(response))
    assert response[0] == 1e-12
    assert response[1] == 1e308


@pytest.mark.parametrize(
    ("scale_state", "floor", "loading", "base_log_scale", "error_match"),
    [
        ([0.0], 0.1, 1.0, 0.0, "scale_state"),
        (np.array([0.0], dtype=np.float32), 0.1, 1.0, 0.0, "scale_state"),
        (np.array([np.nan], dtype=np.float64), 0.1, 1.0, 0.0, "scale_state"),
        (np.array([0.0], dtype=np.float64), 0.0, 1.0, 0.0, "floor"),
        (np.array([0.0], dtype=np.float64), -0.1, 1.0, 0.0, "floor"),
        (np.array([0.0], dtype=np.float64), np.inf, 1.0, 0.0, "floor"),
        (np.array([0.0], dtype=np.float64), 0.1, np.inf, 0.0, "loading"),
        (np.array([0.0], dtype=np.float64), 0.1, True, 0.0, "loading"),
        (
            np.array([0.0], dtype=np.float64),
            0.1,
            np.array([1.0], dtype=np.float32),
            0.0,
            "loading",
        ),
        (
            np.array([0.0, 1.0], dtype=np.float64),
            0.1,
            np.ones(3, dtype=np.float64),
            0.0,
            "loading",
        ),
        (
            np.array([0.0], dtype=np.float64),
            0.1,
            1.0,
            np.array([np.inf], dtype=np.float64),
            "base_log_scale",
        ),
    ],
)
def test_scale_function_rejects_invalid_dtype_shape_values_and_floor(
    scale_state: object,
    floor: object,
    loading: object,
    base_log_scale: object,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        scale_function(
            scale_state,  # type: ignore[arg-type]
            floor,  # type: ignore[arg-type]
            loading,  # type: ignore[arg-type]
            base_log_scale=base_log_scale,  # type: ignore[arg-type]
        )


def test_scale_function_rejects_a_nonfinite_linear_response() -> None:
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match=r"response.*finite"):
            scale_function(
                np.array([1e308], dtype=np.float64),
                0.1,
                2.0,
                base_log_scale=0.0,
            )


@pytest.mark.parametrize(
    ("field_name", "replacement", "error_match"),
    [
        (
            "regime_sequence",
            np.array([0, 1, 0], dtype=np.int32),
            "regime_sequence",
        ),
        (
            "regime_sequence",
            np.array([[0, 1, 0]], dtype=np.int64),
            "regime_sequence",
        ),
        (
            "regime_sequence",
            np.array([0, 2, 0], dtype=np.int64),
            "regime_sequence",
        ),
        (
            "trend_ar_coefficients",
            np.array([1.0, 0.2], dtype=np.float64),
            "trend_ar_coefficients",
        ),
        (
            "scale_ar_coefficients",
            np.array([-1.0, 0.2], dtype=np.float64),
            "scale_ar_coefficients",
        ),
        (
            "trend_ar_coefficients",
            np.array([0.2, np.nan], dtype=np.float64),
            "trend_ar_coefficients",
        ),
        (
            "scale_ar_coefficients",
            np.array([0.2], dtype=np.float64),
            "coefficients",
        ),
        (
            "trend_innovations",
            np.array([1.0, 2.0], dtype=np.float64),
            "trend_innovations",
        ),
        (
            "scale_innovations",
            np.array([-1.0, 0.5, -2.0], dtype=np.float32),
            "scale_innovations",
        ),
        ("initial_trend", np.inf, "initial_trend"),
        ("initial_scale", True, "initial_scale"),
    ],
)
def test_generate_latent_concepts_rejects_invalid_inputs(
    field_name: str,
    replacement: object,
    error_match: str,
) -> None:
    inputs = _latent_inputs()
    inputs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=error_match):
        generate_latent_concepts(**inputs)  # type: ignore[arg-type]


def test_generate_latent_concepts_rejects_nonfinite_rollout_output() -> None:
    inputs = _latent_inputs()
    inputs["trend_ar_coefficients"] = np.array([0.9, 0.8], dtype=np.float64)
    inputs["trend_innovations"] = np.array([1e308, 0.0, 0.0], dtype=np.float64)
    inputs["initial_trend"] = 1e308

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match=r"trend.*finite"):
            generate_latent_concepts(**inputs)  # type: ignore[arg-type]


def test_trend_replacement_changes_only_current_state_then_naturally_evolves() -> None:
    base = _generate_path()

    replaced = replace_concept_at_origin(
        base,
        concept="trend",
        origin_index=1,
        source_value=10.0,
    )

    assert np.array_equal(
        replaced.trend,
        np.array([3.0, 10.0, -0.5, 3.75], dtype=np.float64),
    )
    assert np.array_equal(replaced.scale, base.scale)
    assert np.array_equal(replaced.scale_innovations, base.scale_innovations)
    assert np.array_equal(replaced.regime_sequence, base.regime_sequence)
    assert replaced.trend[2] != 10.0
    assert replaced.trend[3] != 10.0
    assert np.array_equal(
        base.trend,
        np.array([3.0, 2.5, 1.375, 4.6875], dtype=np.float64),
    )


def test_scale_replacement_changes_only_scale_and_preserves_trend_truth() -> None:
    base = _generate_path()

    replaced = replace_concept_at_origin(
        base,
        concept="scale",
        origin_index=1,
        source_value=8.0,
    )

    assert np.array_equal(
        replaced.scale,
        np.array([-3.0, 8.0, 2.5, -3.25], dtype=np.float64),
    )
    assert np.array_equal(replaced.trend, base.trend)
    assert np.array_equal(replaced.trend_innovations, base.trend_innovations)
    assert np.array_equal(replaced.regime_sequence, base.regime_sequence)


@pytest.mark.parametrize("concept", ["trend", "scale"])
def test_source_equal_to_base_is_bitwise_identical(concept: str) -> None:
    base = _generate_path()
    source_value = float(getattr(base, concept)[1])

    replaced = replace_concept_at_origin(
        base,
        concept=concept,  # type: ignore[arg-type]
        origin_index=1,
        source_value=source_value,
    )

    for field_name in PATH_ARRAY_FIELDS:
        replaced_array = getattr(replaced, field_name)
        base_array = getattr(base, field_name)
        assert replaced_array.shape == base_array.shape
        assert replaced_array.dtype == base_array.dtype
        assert replaced_array.tobytes() == base_array.tobytes()


@pytest.mark.parametrize("concept", ["trend", "scale"])
def test_source_equal_to_negative_zero_preserves_every_bit(concept: str) -> None:
    inputs = _latent_inputs()
    inputs["initial_trend"] = -0.0
    inputs["initial_scale"] = -0.0
    base = generate_latent_concepts(**inputs)  # type: ignore[arg-type]

    replaced = replace_concept_at_origin(
        base,
        concept=concept,  # type: ignore[arg-type]
        origin_index=0,
        source_value=float(getattr(base, concept)[0]),
    )

    for field_name in PATH_ARRAY_FIELDS:
        assert getattr(replaced, field_name).tobytes() == getattr(base, field_name).tobytes()
    assert np.float64(replaced.initial_trend).tobytes() == np.float64(base.initial_trend).tobytes()
    assert np.float64(replaced.initial_scale).tobytes() == np.float64(base.initial_scale).tobytes()


def test_replacing_the_initial_origin_updates_the_selected_initial_state() -> None:
    base = _generate_path()

    replaced = replace_concept_at_origin(
        base,
        concept="trend",
        origin_index=0,
        source_value=-2.0,
    )

    assert replaced.initial_trend == -2.0
    assert replaced.initial_scale == base.initial_scale
    assert np.array_equal(
        replaced.trend,
        np.array([-2.0, 0.0, 2.0, 5.0], dtype=np.float64),
    )
    assert np.array_equal(replaced.scale, base.scale)


@pytest.mark.parametrize(
    ("concept", "source_value"),
    [("trend", 10.0), ("scale", 8.0)],
)
def test_replacement_preserves_all_input_bits_and_returns_read_only_arrays(
    concept: str,
    source_value: float,
) -> None:
    base = _generate_path()
    input_snapshot = {
        field_name: getattr(base, field_name).tobytes() for field_name in PATH_ARRAY_FIELDS
    }

    replaced = replace_concept_at_origin(
        base,
        concept=concept,  # type: ignore[arg-type]
        origin_index=1,
        source_value=source_value,
    )

    assert all(
        getattr(base, field_name).tobytes() == input_snapshot[field_name]
        for field_name in PATH_ARRAY_FIELDS
    )
    assert all(
        not getattr(replaced, field_name).flags.writeable for field_name in PATH_ARRAY_FIELDS
    )


@pytest.mark.parametrize(
    ("concept", "origin_index", "source_value", "error_match"),
    [
        ("unknown", 1, 0.0, "concept"),
        ("trend", -1, 0.0, "origin_index"),
        ("trend", 3, 0.0, "origin_index"),
        ("trend", True, 0.0, "origin_index"),
        ("scale", 1, np.inf, "source_value"),
        ("scale", 1, True, "source_value"),
    ],
)
def test_replace_concept_at_origin_rejects_invalid_requests(
    concept: object,
    origin_index: object,
    source_value: object,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        replace_concept_at_origin(
            _generate_path(),
            concept=concept,  # type: ignore[arg-type]
            origin_index=origin_index,  # type: ignore[arg-type]
            source_value=source_value,  # type: ignore[arg-type]
        )


def test_latent_operations_do_not_touch_global_numpy_rng_state() -> None:
    original_state = np.random.get_state()
    try:
        np.random.seed(811)
        before = np.random.get_state()

        path = _generate_path()
        replace_concept_at_origin(
            path,
            concept="trend",
            origin_index=1,
            source_value=4.0,
        )
        scale_function(
            path.scale,
            0.1,
            0.5,
            base_log_scale=-0.2,
        )

        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original_state)


class _ForbiddenRandomAccess:
    def __getattr__(self, attribute_name: str) -> object:
        raise AssertionError(f"latent concept code accessed RNG attribute {attribute_name!r}")


def test_latent_operations_never_access_any_numpy_rng_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_concepts_module.np,
        "random",
        _ForbiddenRandomAccess(),
    )

    path = _generate_path()
    replace_concept_at_origin(
        path,
        concept="scale",
        origin_index=1,
        source_value=2.0,
    )
    scale_function(
        path.scale,
        0.1,
        0.5,
        base_log_scale=-0.2,
    )
