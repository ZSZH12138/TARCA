from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from tarca.data.synthetic.nonlinear_var import (
    RegimeDynamics,
    SyntheticTrajectory,
    deterministic_transition,
    generate_regime_dynamics,
    rollout_nonlinear_var,
    scale_to_spectral_radius,
    spectral_radius,
)


def _dynamics(
    *,
    regime_label: int = 0,
    dimension: int = 1,
    exogenous_dimension: int = 0,
    linear_value: float = 0.0,
    nonlinear_value: float = 0.0,
    nonlinear_strength: float = 0.0,
    exogenous_value: float = 0.0,
    linear_matrix: np.ndarray | None = None,
    base_log_scale: float = 0.0,
    scale_loading: float = 0.0,
    nonlinear_delay: int = 0,
    trend_delay: int = 0,
) -> RegimeDynamics:
    resolved_linear = (
        np.eye(dimension, dtype=np.float64) * linear_value
        if linear_matrix is None
        else linear_matrix
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(resolved_linear))))
    return RegimeDynamics(
        regime_label=regime_label,
        linear_matrix=resolved_linear,
        nonlinear_matrix=np.eye(dimension, dtype=np.float64) * nonlinear_value,
        exogenous_matrix=np.full(
            (dimension, exogenous_dimension),
            exogenous_value,
            dtype=np.float64,
        ),
        nonlinear_strength=nonlinear_strength,
        base_log_scale=base_log_scale,
        scale_loading=scale_loading,
        nonlinear_delay=nonlinear_delay,
        trend_delay=trend_delay,
        raw_spectral_radius=radius,
        spectral_scale_factor=1.0,
        final_spectral_radius=radius,
    )


def _rollout_inputs(
    dynamics_schedule: tuple[RegimeDynamics, ...],
    *,
    initial_history: np.ndarray | None = None,
    trend_history: np.ndarray | None = None,
    trend_path: np.ndarray | None = None,
    scale_path: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
    exogenous_inputs: np.ndarray | None = None,
    observation_innovations: np.ndarray | None = None,
    shocks: np.ndarray | None = None,
    trend_loading: np.ndarray | None = None,
    observation_scale_floor: float = 0.25,
    burn_in: int = 0,
) -> dict[str, object]:
    steps = len(dynamics_schedule)
    dimension = dynamics_schedule[0].linear_matrix.shape[0]
    exogenous_dimension = dynamics_schedule[0].exogenous_matrix.shape[1]
    return {
        "initial_history": (
            np.zeros((1, dimension), dtype=np.float64)
            if initial_history is None
            else initial_history
        ),
        "trend_history": (
            np.empty(0, dtype=np.float64) if trend_history is None else trend_history
        ),
        "trend_path": (np.zeros(steps, dtype=np.float64) if trend_path is None else trend_path),
        "scale_path": (np.zeros(steps, dtype=np.float64) if scale_path is None else scale_path),
        "dynamics_schedule": dynamics_schedule,
        "regime_labels": (
            np.array(
                [dynamics.regime_label for dynamics in dynamics_schedule],
                dtype=np.int64,
            )
            if regime_labels is None
            else regime_labels
        ),
        "exogenous_inputs": (
            np.zeros((steps, exogenous_dimension), dtype=np.float64)
            if exogenous_inputs is None
            else exogenous_inputs
        ),
        "observation_innovations": (
            np.zeros((steps, dimension), dtype=np.float64)
            if observation_innovations is None
            else observation_innovations
        ),
        "shocks": (np.zeros((steps, dimension), dtype=np.float64) if shocks is None else shocks),
        "trend_loading": (
            np.zeros(dimension, dtype=np.float64) if trend_loading is None else trend_loading
        ),
        "observation_scale_floor": observation_scale_floor,
        "burn_in": burn_in,
    }


def test_spectral_radius_is_hand_computable_and_does_not_mutate_input() -> None:
    matrix = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float64)
    before = matrix.copy()

    assert spectral_radius(matrix) == pytest.approx(0.5)
    np.testing.assert_array_equal(matrix, before)
    assert spectral_radius(np.zeros((2, 2), dtype=np.float64)) == 0.0


@pytest.mark.parametrize(
    ("matrix", "error_type", "message"),
    [
        ([[1.0]], TypeError, "matrix"),
        (np.array([[1.0]], dtype=np.float32), TypeError, "float64"),
        (np.array([1.0], dtype=np.float64), ValueError, r"\[D, D\]"),
        (np.ones((1, 2), dtype=np.float64), ValueError, "square"),
        (np.array([[np.nan]], dtype=np.float64), ValueError, "finite"),
    ],
)
def test_spectral_radius_rejects_malformed_matrices(
    matrix: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        spectral_radius(matrix)  # type: ignore[arg-type]


def test_stability_scaling_records_exact_evidence_once() -> None:
    candidate = np.array([[1.0, 0.0], [0.0, -0.5]], dtype=np.float64)
    before = candidate.copy()

    scaled, raw_radius, factor, final_radius = scale_to_spectral_radius(
        candidate,
        target=0.8,
    )

    assert raw_radius == pytest.approx(1.0)
    assert factor == pytest.approx(0.8)
    assert final_radius == pytest.approx(0.8)
    np.testing.assert_allclose(scaled, np.array([[0.8, 0.0], [0.0, -0.4]]))
    np.testing.assert_array_equal(candidate, before)
    assert not scaled.flags.writeable

    rounding_candidate = np.array([[1.2819022270597127]], dtype=np.float64)
    rounding_result = scale_to_spectral_radius(rounding_candidate)
    assert rounding_result[3] <= 0.85
    assert spectral_radius(rounding_result[0]) <= 0.85


def test_stability_scaling_preserves_stable_and_zero_candidates() -> None:
    stable = np.array([[0.4]], dtype=np.float64)
    zero = np.zeros((1, 1), dtype=np.float64)

    stable_result = scale_to_spectral_radius(stable)
    zero_result = scale_to_spectral_radius(zero)

    np.testing.assert_array_equal(stable_result[0], stable)
    assert stable_result[1:] == pytest.approx((0.4, 1.0, 0.4))
    np.testing.assert_array_equal(zero_result[0], zero)
    assert zero_result[1:] == pytest.approx((0.0, 1.0, 0.0))
    assert not np.shares_memory(stable_result[0], stable)
    assert not np.shares_memory(zero_result[0], zero)


@pytest.mark.parametrize("target", [0.0, -0.1, 0.8500001, np.inf, np.nan, True])
def test_stability_scaling_rejects_invalid_target(target: object) -> None:
    with pytest.raises((TypeError, ValueError), match="target"):
        scale_to_spectral_radius(
            np.array([[0.5]], dtype=np.float64),
            target=target,  # type: ignore[arg-type]
        )


def test_generate_regime_dynamics_scales_only_linear_candidates_and_records_graph() -> None:
    linear_candidates = np.array([[[1.0]], [[0.4]]], dtype=np.float64)
    nonlinear_matrices = np.array([[[2.0]], [[0.0]]], dtype=np.float64)
    exogenous_matrices = np.array([[[3.0]], [[4.0]]], dtype=np.float64)
    nonlinear_strengths = np.array([0.5, 0.0], dtype=np.float64)
    base_log_scales = np.array([-1.0, 1.0], dtype=np.float64)
    scale_loadings = np.array([0.25, -0.5], dtype=np.float64)
    nonlinear_delays = np.array([1, 0], dtype=np.int64)
    trend_delays = np.array([2, 0], dtype=np.int64)
    inputs_before = tuple(
        array.copy()
        for array in (
            linear_candidates,
            nonlinear_matrices,
            exogenous_matrices,
            nonlinear_strengths,
            base_log_scales,
            scale_loadings,
            nonlinear_delays,
            trend_delays,
        )
    )

    result = generate_regime_dynamics(
        linear_candidates=linear_candidates,
        nonlinear_matrices=nonlinear_matrices,
        exogenous_matrices=exogenous_matrices,
        nonlinear_strengths=nonlinear_strengths,
        base_log_scales=base_log_scales,
        scale_loadings=scale_loadings,
        nonlinear_delays=nonlinear_delays,
        trend_delays=trend_delays,
        target=0.8,
    )

    assert isinstance(result, tuple)
    assert [item.regime_label for item in result] == [0, 1]
    assert result[0].raw_spectral_radius == pytest.approx(1.0)
    assert result[0].spectral_scale_factor == pytest.approx(0.8)
    assert result[0].final_spectral_radius == pytest.approx(0.8)
    np.testing.assert_allclose(result[0].linear_matrix, np.array([[0.8]]))
    np.testing.assert_array_equal(result[0].nonlinear_matrix, np.array([[2.0]]))
    np.testing.assert_array_equal(result[0].true_graph, np.array([[True]]))
    assert result[1].spectral_scale_factor == 1.0
    np.testing.assert_array_equal(result[1].true_graph, np.array([[True]]))
    for original, before in zip(
        (
            linear_candidates,
            nonlinear_matrices,
            exogenous_matrices,
            nonlinear_strengths,
            base_log_scales,
            scale_loadings,
            nonlinear_delays,
            trend_delays,
        ),
        inputs_before,
        strict=True,
    ):
        np.testing.assert_array_equal(original, before)
    for dynamics in result:
        assert not dynamics.linear_matrix.flags.writeable
        assert not dynamics.nonlinear_matrix.flags.writeable
        assert not dynamics.exogenous_matrix.flags.writeable
        assert not dynamics.true_graph.flags.writeable


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        (
            {"linear_candidates": np.zeros((1, 1, 1), dtype=np.float32)},
            TypeError,
            "linear_candidates.*float64",
        ),
        (
            {"nonlinear_matrices": np.zeros((1, 2, 2), dtype=np.float64)},
            ValueError,
            "nonlinear_matrices.*shape",
        ),
        (
            {"exogenous_matrices": np.zeros((2, 1, 0), dtype=np.float64)},
            ValueError,
            "exogenous_matrices.*shape",
        ),
        (
            {"nonlinear_delays": np.array([-1], dtype=np.int64)},
            ValueError,
            "nonlinear_delays.*non-negative",
        ),
        (
            {"trend_delays": np.zeros(1, dtype=np.float64)},
            TypeError,
            "trend_delays.*int64",
        ),
        (
            {"base_log_scales": np.array([np.inf], dtype=np.float64)},
            ValueError,
            "base_log_scales.*finite",
        ),
        (
            {"true_graphs": np.zeros((1, 2, 2), dtype=np.bool_)},
            ValueError,
            "true_graphs.*shape",
        ),
        ({"true_graphs": np.ones((1, 1, 1), dtype=np.bool_)}, ValueError, "true_graph"),
    ],
)
def test_generate_regime_dynamics_rejects_invalid_candidates(
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "linear_candidates": np.zeros((1, 1, 1), dtype=np.float64),
        "nonlinear_matrices": np.zeros((1, 1, 1), dtype=np.float64),
        "exogenous_matrices": np.zeros((1, 1, 0), dtype=np.float64),
        "nonlinear_strengths": np.zeros(1, dtype=np.float64),
        "base_log_scales": np.zeros(1, dtype=np.float64),
        "scale_loadings": np.zeros(1, dtype=np.float64),
        "nonlinear_delays": np.zeros(1, dtype=np.int64),
        "trend_delays": np.zeros(1, dtype=np.int64),
    }
    arguments.update(replacement)

    with pytest.raises(error_type, match=message):
        generate_regime_dynamics(**arguments)  # type: ignore[arg-type]


def test_regime_dynamics_is_frozen_deep_copied_and_direct_construction_never_scales() -> None:
    linear_matrix = np.array([[0.5]], dtype=np.float64)
    dynamics = _dynamics(linear_value=0.5, linear_matrix=linear_matrix)
    linear_matrix_before = linear_matrix.copy()

    with pytest.raises(FrozenInstanceError):
        dynamics.regime_label = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        dynamics.linear_matrix[0, 0] = 0.0
    linear_matrix[0, 0] = 0.25
    np.testing.assert_array_equal(dynamics.linear_matrix, linear_matrix_before)

    unstable = np.array([[0.9]], dtype=np.float64)
    with pytest.raises(ValueError, match="stability_target"):
        RegimeDynamics(
            regime_label=0,
            linear_matrix=unstable,
            nonlinear_matrix=np.zeros((1, 1), dtype=np.float64),
            exogenous_matrix=np.zeros((1, 0), dtype=np.float64),
            nonlinear_strength=0.0,
            base_log_scale=0.0,
            scale_loading=0.0,
            nonlinear_delay=0,
            trend_delay=0,
            raw_spectral_radius=0.9,
            spectral_scale_factor=1.0,
            final_spectral_radius=0.9,
        )
    np.testing.assert_array_equal(unstable, np.array([[0.9]]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("regime_label", -1, "regime_label"),
        ("nonlinear_delay", -1, "nonlinear_delay"),
        ("trend_delay", -1, "trend_delay"),
        ("raw_spectral_radius", 0.6, "spectral evidence"),
        ("spectral_scale_factor", 0.5, "spectral evidence"),
        ("final_spectral_radius", 0.4, "final_spectral_radius"),
        ("stability_target", 0.9, "stability_target"),
    ],
)
def test_regime_dynamics_rejects_invalid_labels_delays_targets_and_evidence(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "regime_label": 0,
        "linear_matrix": np.array([[0.5]], dtype=np.float64),
        "nonlinear_matrix": np.zeros((1, 1), dtype=np.float64),
        "exogenous_matrix": np.zeros((1, 0), dtype=np.float64),
        "nonlinear_strength": 0.0,
        "base_log_scale": 0.0,
        "scale_loading": 0.0,
        "nonlinear_delay": 0,
        "trend_delay": 0,
        "raw_spectral_radius": 0.5,
        "spectral_scale_factor": 1.0,
        "final_spectral_radius": 0.5,
        "stability_target": 0.85,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        RegimeDynamics(**arguments)  # type: ignore[arg-type]


def test_deterministic_transition_matches_fixed_regime_linear_subcase() -> None:
    next_state = deterministic_transition(
        state_history=np.array([[2.0, -4.0]], dtype=np.float64),
        trend_history=np.array([0.0], dtype=np.float64),
        scale_state=0.0,
        exogenous_input=np.empty(0, dtype=np.float64),
        observation_innovation=np.zeros(2, dtype=np.float64),
        shock=np.zeros(2, dtype=np.float64),
        dynamics=_dynamics(dimension=2, linear_value=0.5),
        trend_loading=np.zeros(2, dtype=np.float64),
        observation_scale_floor=0.25,
    )

    np.testing.assert_array_equal(next_state, np.array([1.0, -2.0]))
    assert next_state.dtype == np.float64
    assert not next_state.flags.writeable


def test_deterministic_transition_uses_exact_nonlinear_and_trend_lag_indices() -> None:
    next_state = deterministic_transition(
        state_history=np.array([[0.25], [5.0], [9.0]], dtype=np.float64),
        trend_history=np.array([2.0, 7.0, 11.0], dtype=np.float64),
        scale_state=0.0,
        exogenous_input=np.empty(0, dtype=np.float64),
        observation_innovation=np.zeros(1, dtype=np.float64),
        shock=np.zeros(1, dtype=np.float64),
        dynamics=_dynamics(
            nonlinear_value=2.0,
            nonlinear_strength=3.0,
            nonlinear_delay=2,
            trend_delay=2,
        ),
        trend_loading=np.array([4.0], dtype=np.float64),
        observation_scale_floor=0.25,
    )

    expected = 3.0 * np.tanh(2.0 * 0.25) + 4.0 * 2.0
    np.testing.assert_allclose(next_state, np.array([expected]))


def test_zero_delay_uses_current_state_and_concept_for_earliest_next_step() -> None:
    next_state = deterministic_transition(
        state_history=np.array([[1.0], [3.0]], dtype=np.float64),
        trend_history=np.array([2.0, 5.0], dtype=np.float64),
        scale_state=0.0,
        exogenous_input=np.empty(0, dtype=np.float64),
        observation_innovation=np.zeros(1, dtype=np.float64),
        shock=np.zeros(1, dtype=np.float64),
        dynamics=_dynamics(
            nonlinear_value=1.0,
            nonlinear_strength=2.0,
            nonlinear_delay=0,
            trend_delay=0,
        ),
        trend_loading=np.array([3.0], dtype=np.float64),
        observation_scale_floor=0.25,
    )

    expected = 2.0 * np.tanh(3.0) + 15.0
    np.testing.assert_allclose(next_state, np.array([expected]))


def test_deterministic_transition_adds_exogenous_scale_noise_and_shock_components() -> None:
    floor = 0.25
    next_state = deterministic_transition(
        state_history=np.zeros((1, 2), dtype=np.float64),
        trend_history=np.array([0.0], dtype=np.float64),
        scale_state=10.0,
        exogenous_input=np.array([3.0], dtype=np.float64),
        observation_innovation=np.array([2.0, -1.0], dtype=np.float64),
        shock=np.array([5.0, 7.0], dtype=np.float64),
        dynamics=_dynamics(
            dimension=2,
            exogenous_dimension=1,
            exogenous_value=2.0,
            base_log_scale=0.0,
            scale_loading=0.0,
        ),
        trend_loading=np.zeros(2, dtype=np.float64),
        observation_scale_floor=floor,
    )

    scale = np.log(2.0) + floor
    expected = np.array([6.0 + 2.0 * scale + 5.0, 6.0 - scale + 7.0])
    np.testing.assert_allclose(next_state, expected)


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        (
            {"state_history": np.zeros((1, 1), dtype=np.float32)},
            TypeError,
            "state_history.*float64",
        ),
        (
            {"state_history": np.empty((0, 1), dtype=np.float64)},
            ValueError,
            "state_history",
        ),
        (
            {"trend_history": np.empty(0, dtype=np.float64)},
            ValueError,
            "trend_history",
        ),
        ({"observation_scale_floor": 0.0}, ValueError, "observation_scale_floor"),
        (
            {"observation_innovation": np.array([np.nan], dtype=np.float64)},
            ValueError,
            "observation_innovation.*finite",
        ),
    ],
)
def test_deterministic_transition_rejects_invalid_or_insufficient_inputs(
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "state_history": np.array([[1.0], [2.0]], dtype=np.float64),
        "trend_history": np.array([1.0], dtype=np.float64),
        "scale_state": 0.0,
        "exogenous_input": np.empty(0, dtype=np.float64),
        "observation_innovation": np.zeros(1, dtype=np.float64),
        "shock": np.zeros(1, dtype=np.float64),
        "dynamics": _dynamics(nonlinear_delay=1),
        "trend_loading": np.zeros(1, dtype=np.float64),
        "observation_scale_floor": 0.25,
    }
    arguments.update(replacement)

    with pytest.raises(error_type, match=message):
        deterministic_transition(**arguments)  # type: ignore[arg-type]


def test_rollout_matches_hand_computable_recurrence_and_burn_in_alignment() -> None:
    dynamics = _dynamics(linear_value=0.5)
    inputs = _rollout_inputs(
        (dynamics, dynamics, dynamics),
        initial_history=np.array([[4.0]], dtype=np.float64),
        burn_in=1,
    )

    trajectory = rollout_nonlinear_var(**inputs)

    np.testing.assert_array_equal(trajectory.full_values, np.array([[2.0], [1.0], [0.5]]))
    np.testing.assert_array_equal(trajectory.values, np.array([[1.0], [0.5]]))
    np.testing.assert_array_equal(
        trajectory.values,
        trajectory.full_values[trajectory.burn_in :],
    )
    assert trajectory.burn_in == 1


def test_rollout_uses_per_step_dynamics_even_when_regime_label_is_unchanged() -> None:
    first = _dynamics(regime_label=0, linear_value=0.5)
    shifted = _dynamics(regime_label=0, linear_value=0.25)
    stopped = _dynamics(regime_label=0, linear_value=0.0)
    inputs = _rollout_inputs(
        (first, shifted, stopped),
        initial_history=np.array([[8.0]], dtype=np.float64),
        trend_path=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        regime_labels=np.zeros(3, dtype=np.int64),
        trend_loading=np.ones(1, dtype=np.float64),
    )

    trajectory = rollout_nonlinear_var(**inputs)
    future_changed = {**inputs, "trend_path": np.array([1.0, 200.0, 300.0])}

    np.testing.assert_array_equal(trajectory.full_values, np.array([[5.0], [3.25], [3.0]]))
    assert rollout_nonlinear_var(**future_changed).full_values[0, 0] == 5.0
    assert trajectory.dynamics_schedule == (first, shifted, stopped)


def test_rollout_factual_self_replay_is_bitwise_identical() -> None:
    first = _dynamics(
        regime_label=0,
        dimension=2,
        exogenous_dimension=1,
        linear_value=0.25,
        nonlinear_value=0.5,
        nonlinear_strength=0.75,
        exogenous_value=0.2,
        base_log_scale=-0.5,
        scale_loading=0.3,
        nonlinear_delay=1,
        trend_delay=1,
    )
    second = _dynamics(
        regime_label=1,
        dimension=2,
        exogenous_dimension=1,
        linear_value=-0.1,
        nonlinear_value=0.2,
        nonlinear_strength=-0.4,
        exogenous_value=-0.1,
        base_log_scale=0.25,
        scale_loading=-0.2,
        nonlinear_delay=0,
        trend_delay=0,
    )
    inputs = _rollout_inputs(
        (first, second, first),
        initial_history=np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float64),
        trend_history=np.array([0.1], dtype=np.float64),
        trend_path=np.array([0.2, -0.3, 0.4], dtype=np.float64),
        scale_path=np.array([0.5, -0.25, 0.75], dtype=np.float64),
        exogenous_inputs=np.array([[1.0], [2.0], [-1.0]], dtype=np.float64),
        observation_innovations=np.array(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
            dtype=np.float64,
        ),
        shocks=np.array([[0.0, 1.0], [0.5, 0.0], [-0.25, 0.75]], dtype=np.float64),
        trend_loading=np.array([0.4, -0.2], dtype=np.float64),
        burn_in=1,
    )

    first_rollout = rollout_nonlinear_var(**inputs)
    replay = rollout_nonlinear_var(
        initial_history=first_rollout.initial_history,
        trend_history=first_rollout.trend_history,
        trend_path=first_rollout.trend_path,
        scale_path=first_rollout.scale_path,
        dynamics_schedule=first_rollout.dynamics_schedule,
        regime_labels=first_rollout.regime_labels,
        exogenous_inputs=first_rollout.exogenous_inputs,
        observation_innovations=first_rollout.observation_innovations,
        shocks=first_rollout.shocks,
        trend_loading=first_rollout.trend_loading,
        observation_scale_floor=first_rollout.observation_scale_floor,
        burn_in=first_rollout.burn_in,
    )

    assert first_rollout.full_values.tobytes() == replay.full_values.tobytes()
    assert first_rollout.values.tobytes() == replay.values.tobytes()


def test_rollout_preserves_inputs_and_returns_frozen_read_only_truth() -> None:
    dynamics = _dynamics()
    inputs = _rollout_inputs(
        (dynamics, dynamics),
        trend_path=np.array([1.0, 2.0], dtype=np.float64),
        scale_path=np.array([3.0, 4.0], dtype=np.float64),
        observation_innovations=np.array([[0.1], [0.2]], dtype=np.float64),
        shocks=np.array([[0.3], [0.4]], dtype=np.float64),
    )
    array_inputs = {
        name: value.copy() for name, value in inputs.items() if isinstance(value, np.ndarray)
    }

    trajectory = rollout_nonlinear_var(**inputs)

    for name, before in array_inputs.items():
        np.testing.assert_array_equal(inputs[name], before)
    for field_name in (
        "values",
        "full_values",
        "initial_history",
        "trend_history",
        "trend_path",
        "scale_path",
        "regime_labels",
        "exogenous_inputs",
        "observation_innovations",
        "shocks",
        "trend_loading",
    ):
        assert not getattr(trajectory, field_name).flags.writeable
    with pytest.raises(FrozenInstanceError):
        trajectory.burn_in = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        trajectory.full_values[0, 0] = 1.0


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    [
        (
            {"regime_labels": np.zeros(2, dtype=np.float64)},
            TypeError,
            "regime_labels.*int64",
        ),
        (
            {"regime_labels": np.array([0, -1], dtype=np.int64)},
            ValueError,
            "regime_labels.*non-negative",
        ),
        (
            {"regime_labels": np.array([0, 1], dtype=np.int64)},
            ValueError,
            "regime_labels.*dynamics_schedule",
        ),
        (
            {"trend_path": np.zeros(1, dtype=np.float64)},
            ValueError,
            "trend_path.*shape",
        ),
        (
            {"scale_path": np.zeros((2, 1), dtype=np.float64)},
            ValueError,
            "scale_path.*shape",
        ),
        (
            {"exogenous_inputs": np.zeros((2, 1), dtype=np.float64)},
            ValueError,
            "exogenous_inputs.*shape",
        ),
        (
            {"observation_innovations": np.zeros((1, 1), dtype=np.float64)},
            ValueError,
            "observation_innovations.*shape",
        ),
        ({"dynamics_schedule": ()}, ValueError, "dynamics_schedule"),
        ({"burn_in": 2}, ValueError, "burn_in"),
    ],
)
def test_rollout_rejects_shape_dtype_label_and_burn_in_errors(
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    dynamics = _dynamics()
    inputs = _rollout_inputs((dynamics, dynamics))
    inputs.update(replacement)

    with pytest.raises(error_type, match=message):
        rollout_nonlinear_var(**inputs)


def test_rollout_requires_enough_explicit_state_and_trend_history() -> None:
    dynamics = _dynamics(nonlinear_delay=2, trend_delay=2)
    inputs = _rollout_inputs(
        (dynamics,),
        initial_history=np.zeros((2, 1), dtype=np.float64),
        trend_history=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match=r"initial_history.*nonlinear_delay"):
        rollout_nonlinear_var(**inputs)

    inputs["initial_history"] = np.zeros((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match=r"trend_history.*trend_delay"):
        rollout_nonlinear_var(**inputs)


def test_rollout_fails_fast_when_finite_inputs_explode() -> None:
    dynamics = _dynamics(exogenous_dimension=1, exogenous_value=1e308)
    inputs = _rollout_inputs(
        (dynamics,),
        exogenous_inputs=np.array([[1e308]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match=r"transition output.*finite"):
        rollout_nonlinear_var(**inputs)


def test_rollout_and_generation_never_access_module_random_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tarca.data.synthetic.nonlinear_var as nonlinear_var_module

    class RandomTrap:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"unexpected random access: {name}")

    monkeypatch.setattr(nonlinear_var_module.np, "random", RandomTrap())
    generated = generate_regime_dynamics(
        linear_candidates=np.array([[[0.5]]], dtype=np.float64),
        nonlinear_matrices=np.zeros((1, 1, 1), dtype=np.float64),
        exogenous_matrices=np.zeros((1, 1, 0), dtype=np.float64),
        nonlinear_strengths=np.zeros(1, dtype=np.float64),
        base_log_scales=np.zeros(1, dtype=np.float64),
        scale_loadings=np.zeros(1, dtype=np.float64),
        nonlinear_delays=np.zeros(1, dtype=np.int64),
        trend_delays=np.zeros(1, dtype=np.int64),
    )
    validated_fields: list[str] = []
    original_validator = nonlinear_var_module._require_float64_array

    def track_validation(value: object, *, field_name: str) -> np.ndarray:
        validated_fields.append(field_name)
        return original_validator(value, field_name=field_name)

    monkeypatch.setattr(nonlinear_var_module, "_require_float64_array", track_validation)
    trajectory = rollout_nonlinear_var(**_rollout_inputs(generated))

    np.testing.assert_array_equal(trajectory.full_values, np.zeros((1, 1)))
    assert "state_history" not in validated_fields
    assert validated_fields.count("trend_history") == 1


def test_synthetic_trajectory_rejects_tampered_direct_construction() -> None:
    dynamics = _dynamics(linear_value=0.5)
    trajectory = rollout_nonlinear_var(
        **_rollout_inputs(
            (dynamics,),
            initial_history=np.array([[2.0]], dtype=np.float64),
        )
    )

    replay = {field.name: getattr(trajectory, field.name) for field in fields(trajectory)}
    tampered = np.array([[99.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"full_values.*recurrence"):
        SyntheticTrajectory(**(replay | {"values": tampered, "full_values": tampered}))
    with pytest.raises(ValueError, match=r"values.*full_values"):
        SyntheticTrajectory(**(replay | {"values": tampered}))
