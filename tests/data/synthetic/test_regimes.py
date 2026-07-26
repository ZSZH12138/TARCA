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

from tarca.data.synthetic.regimes import (  # noqa: E402
    RandomStream,
    build_regime_parameter_schedule,
    compute_stationary_distribution,
    make_unseen_parameter_shift,
    regime_persistence_statistics,
    resolve_regime_parameters,
    sample_regime_sequence,
    spawn_random_streams,
    validate_transition_matrix,
)

EXPECTED_STREAM_NAMES = (
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


def _transition_matrix() -> np.ndarray:
    return np.array([[0.75, 0.25], [0.4, 0.6]], dtype=np.float64)


def _parameters() -> dict[int, dict[str, int | float | np.ndarray]]:
    return {
        0: {
            "autoregression": np.array([0.2, 0.3], dtype=np.float64),
            "delay": 1,
            "noise_scale": 0.5,
        },
        1: {
            "autoregression": np.array([0.6, 0.7], dtype=np.float64),
            "delay": 2,
            "noise_scale": 0.8,
        },
    }


def test_spawn_random_streams_uses_the_ten_exact_seedsequence_children() -> None:
    streams = spawn_random_streams(20260726)
    expected_children = np.random.SeedSequence(20260726).spawn(10)
    expected_raw_samples = tuple(
        int(np.random.default_rng(child).bit_generator.random_raw()) for child in expected_children
    )

    assert tuple(streams) == EXPECTED_STREAM_NAMES
    assert tuple(stream.spawn_key for stream in streams.values()) == tuple(
        (index,) for index in range(10)
    )
    assert (
        tuple(int(stream.generator.bit_generator.random_raw()) for stream in streams.values())
        == expected_raw_samples
    )
    assert all(isinstance(stream, RandomStream) for stream in streams.values())


def test_spawn_random_streams_reproduces_each_child_stream() -> None:
    first = spawn_random_streams(1729)
    second = spawn_random_streams(1729)

    for name in EXPECTED_STREAM_NAMES:
        assert np.array_equal(
            first[name].generator.standard_normal(16),
            second[name].generator.standard_normal(16),
        )


def test_consuming_one_child_stream_does_not_advance_another() -> None:
    consumed = spawn_random_streams(314159)
    untouched = spawn_random_streams(314159)

    consumed["regime_transitions"].generator.random(1_000)

    assert np.array_equal(
        consumed["trend_innovations"].generator.random(32),
        untouched["trend_innovations"].generator.random(32),
    )


def test_random_stream_registry_and_records_are_immutable() -> None:
    streams = spawn_random_streams(7)
    stream = streams["regime_transitions"]

    with pytest.raises(TypeError):
        streams["extra"] = stream  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        stream.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("root_seed", [True, -1, 1.5, "7"])
def test_spawn_random_streams_rejects_invalid_root_seeds(root_seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="root_seed"):
        spawn_random_streams(root_seed)  # type: ignore[arg-type]


def test_validate_transition_matrix_accepts_a_finite_float64_stochastic_matrix() -> None:
    transition = _transition_matrix()
    before = transition.copy()

    assert validate_transition_matrix(transition) is None
    assert np.array_equal(transition, before)


@pytest.mark.parametrize(
    "transition",
    [
        [[0.8, 0.2], [0.1, 0.9]],
        np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        np.array([0.8, 0.2], dtype=np.float64),
        np.ones((2, 3), dtype=np.float64) / 3.0,
        np.empty((0, 0), dtype=np.float64),
        np.array([[1.1, -0.1], [0.1, 0.9]], dtype=np.float64),
        np.array([[0.8, np.nan], [0.1, 0.9]], dtype=np.float64),
        np.array([[0.8, np.inf], [0.1, 0.9]], dtype=np.float64),
        np.array([[0.8, 0.3], [0.1, 0.9]], dtype=np.float64),
    ],
)
def test_validate_transition_matrix_rejects_malformed_probabilities(
    transition: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="transition"):
        validate_transition_matrix(transition)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "initial_probabilities",
    [
        [0.5, 0.5],
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([-0.1, 1.1], dtype=np.float64),
        np.array([np.nan, 0.0], dtype=np.float64),
        np.array([0.4, 0.4], dtype=np.float64),
    ],
)
def test_sample_regime_sequence_rejects_invalid_initial_distributions(
    initial_probabilities: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="initial_probabilities"):
        sample_regime_sequence(
            _transition_matrix(),
            initial_probabilities,  # type: ignore[arg-type]
            np.array([0.1, 0.2], dtype=np.float64),
        )


@pytest.mark.parametrize(
    "uniforms",
    [
        [0.1, 0.2],
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([[0.1, 0.2]], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([-0.1, 0.2], dtype=np.float64),
        np.array([0.1, 1.0], dtype=np.float64),
        np.array([0.1, np.nan], dtype=np.float64),
    ],
)
def test_sample_regime_sequence_rejects_invalid_uniforms(uniforms: object) -> None:
    with pytest.raises((TypeError, ValueError), match="uniforms"):
        sample_regime_sequence(
            _transition_matrix(),
            np.array([0.3, 0.7], dtype=np.float64),
            uniforms,  # type: ignore[arg-type]
        )


def test_sample_regime_sequence_uses_only_the_supplied_uniforms() -> None:
    transition = _transition_matrix()
    initial_probabilities = np.array([0.3, 0.7], dtype=np.float64)
    uniforms = np.array([0.2, 0.8, 0.1, 0.5, 0.95], dtype=np.float64)
    transition_before = transition.copy()
    initial_before = initial_probabilities.copy()
    uniforms_before = uniforms.copy()

    sequence = sample_regime_sequence(transition, initial_probabilities, uniforms)

    assert np.array_equal(sequence, np.array([0, 1, 0, 0, 1], dtype=np.int64))
    assert sequence.dtype == np.int64
    assert not sequence.flags.writeable
    assert np.array_equal(transition, transition_before)
    assert np.array_equal(initial_probabilities, initial_before)
    assert np.array_equal(uniforms, uniforms_before)


def test_sample_regime_sequence_does_not_touch_the_legacy_global_rng() -> None:
    original_state = np.random.get_state()
    try:
        np.random.seed(811)
        before = np.random.get_state()

        sample_regime_sequence(
            _transition_matrix(),
            np.array([0.3, 0.7], dtype=np.float64),
            np.array([0.2, 0.8, 0.1], dtype=np.float64),
        )

        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original_state)


def test_compute_stationary_distribution_returns_the_unique_fixed_point() -> None:
    transition = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)

    stationary = compute_stationary_distribution(transition)

    assert np.allclose(stationary, np.array([2.0 / 3.0, 1.0 / 3.0]))
    assert np.allclose(stationary @ transition, stationary)
    assert stationary.dtype == np.float64
    assert not stationary.flags.writeable


def test_compute_stationary_distribution_rejects_a_nonunique_solution() -> None:
    with pytest.raises(ValueError, match="unique stationary"):
        compute_stationary_distribution(np.eye(2, dtype=np.float64))


def test_regime_persistence_statistics_reports_literal_run_lengths() -> None:
    statistics = regime_persistence_statistics(np.array([0, 0, 1, 1, 1, 0], dtype=np.int64))

    assert dict(statistics) == {
        "number_of_runs": 3,
        "number_of_transitions": 2,
        "mean_dwell_time": 2.0,
        "median_dwell_time": 2.0,
        "maximum_dwell_time": 3,
    }
    with pytest.raises(TypeError):
        statistics["mean_dwell_time"] = 9.0  # type: ignore[index]


def test_high_diagonal_transition_has_longer_mean_dwell_time() -> None:
    uniforms = np.random.default_rng(909).random(10_000, dtype=np.float64)
    initial = np.array([0.5, 0.5], dtype=np.float64)
    low_persistence = sample_regime_sequence(
        np.array([[0.55, 0.45], [0.45, 0.55]], dtype=np.float64),
        initial,
        uniforms,
    )
    high_persistence = sample_regime_sequence(
        np.array([[0.98, 0.02], [0.02, 0.98]], dtype=np.float64),
        initial,
        uniforms,
    )

    low_statistics = regime_persistence_statistics(low_persistence)
    high_statistics = regime_persistence_statistics(high_persistence)

    assert high_statistics["mean_dwell_time"] > 10 * low_statistics["mean_dwell_time"]


@pytest.mark.parametrize(
    "sequence",
    [
        [0, 1],
        np.array([0, 1], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([0, -1], dtype=np.int64),
    ],
)
def test_regime_persistence_statistics_rejects_invalid_sequences(sequence: object) -> None:
    with pytest.raises((TypeError, ValueError), match="regime_sequence"):
        regime_persistence_statistics(sequence)  # type: ignore[arg-type]


def test_resolve_regime_parameters_returns_deeply_immutable_copies() -> None:
    parameters = _parameters()
    source_array = parameters[0]["autoregression"]
    assert isinstance(source_array, np.ndarray)

    resolved = resolve_regime_parameters(parameters)

    assert tuple(resolved) == (0, 1)
    assert np.array_equal(resolved[0]["autoregression"], source_array)
    resolved_array = resolved[0]["autoregression"]
    assert isinstance(resolved_array, np.ndarray)
    assert not np.shares_memory(resolved_array, source_array)
    assert not resolved_array.flags.writeable
    with pytest.raises(TypeError):
        resolved[0]["delay"] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved[2] = resolved[0]  # type: ignore[index]


def test_build_regime_parameter_schedule_matches_every_regime_label() -> None:
    sequence = np.array([1, 1, 0, 1, 0], dtype=np.int64)

    schedule = build_regime_parameter_schedule(sequence, _parameters())

    assert len(schedule) == len(sequence)
    assert [entry["delay"] for entry in schedule] == [2, 2, 1, 2, 1]
    assert np.array_equal(
        schedule[2]["autoregression"],
        np.array([0.2, 0.3], dtype=np.float64),
    )


def test_build_regime_parameter_schedule_fails_on_an_unknown_label() -> None:
    with pytest.raises(ValueError, match=r"regime_sequence.*2"):
        build_regime_parameter_schedule(
            np.array([0, 2], dtype=np.int64),
            _parameters(),
        )


def test_make_unseen_parameter_shift_preserves_labels_and_base_parameters() -> None:
    base = _parameters()
    base_autoregression = base[0]["autoregression"]
    assert isinstance(base_autoregression, np.ndarray)
    before = base_autoregression.copy()

    unseen = make_unseen_parameter_shift(
        base,
        {
            "autoregression": np.array([0.05, -0.05], dtype=np.float64),
            "delay": 1,
        },
    )

    assert tuple(unseen) == tuple(base)
    assert np.allclose(unseen[0]["autoregression"], np.array([0.25, 0.25]))
    assert np.allclose(unseen[1]["autoregression"], np.array([0.65, 0.65]))
    assert unseen[0]["delay"] == 2
    assert unseen[1]["delay"] == 3
    assert unseen[0]["noise_scale"] == base[0]["noise_scale"]
    assert np.array_equal(base_autoregression, before)
    shifted_array = unseen[0]["autoregression"]
    assert isinstance(shifted_array, np.ndarray)
    assert not shifted_array.flags.writeable


def test_integer_parameters_reject_fractional_unseen_shifts() -> None:
    with pytest.raises(TypeError, match=r"delay.*integer parameter"):
        make_unseen_parameter_shift(_parameters(), {"delay": 0.5})


def test_resolve_regime_parameters_rejects_mixed_scalar_kinds() -> None:
    with pytest.raises(ValueError, match=r"delay.*signature"):
        resolve_regime_parameters(
            {
                0: {"delay": 1},
                1: {"delay": 2.0},
            }
        )


def test_parameter_builders_fail_fast_on_invalid_labels_values_and_shifts() -> None:
    with pytest.raises(ValueError, match="regime labels"):
        resolve_regime_parameters({1: {"delay": 1}})
    with pytest.raises(ValueError, match="finite"):
        resolve_regime_parameters({0: {"noise_scale": np.nan}})
    with pytest.raises(ValueError, match="unknown parameter"):
        make_unseen_parameter_shift(_parameters(), {"missing": 1.0})
    with pytest.raises(ValueError, match="shape"):
        make_unseen_parameter_shift(
            _parameters(),
            {"autoregression": np.ones(3, dtype=np.float64)},
        )
