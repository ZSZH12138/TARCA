from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.data.synthetic.counterfactual_oracle as oracle_module  # noqa: E402
from tarca.contracts import InterventionPair, RegimeRelation, SplitPartition  # noqa: E402
from tarca.data.synthetic.counterfactual_oracle import (  # noqa: E402
    CounterfactualIntervention,
    FutureNoiseBank,
    MonteCarloOracleResult,
    PairedCounterfactualResult,
    estimate_effect_delay,
    monte_carlo_oracle,
    replay_paired_counterfactual,
)
from tarca.data.synthetic.nonlinear_var import (  # noqa: E402
    RegimeDynamics,
    SyntheticTrajectory,
    rollout_nonlinear_var,
)


def _dynamics(
    *,
    regime_label: int = 0,
    linear: float = 0.0,
    nonlinear_delay: int = 0,
    trend_delay: int = 0,
    exogenous_dimension: int = 0,
    exogenous_loading: float = 0.0,
    base_log_scale: float = 0.0,
    scale_loading: float = 0.0,
) -> RegimeDynamics:
    radius = abs(linear)
    return RegimeDynamics(
        regime_label=regime_label,
        linear_matrix=np.array([[linear]], dtype=np.float64),
        nonlinear_matrix=np.zeros((1, 1), dtype=np.float64),
        exogenous_matrix=np.full(
            (1, exogenous_dimension),
            exogenous_loading,
            dtype=np.float64,
        ),
        nonlinear_strength=0.0,
        base_log_scale=base_log_scale,
        scale_loading=scale_loading,
        nonlinear_delay=nonlinear_delay,
        trend_delay=trend_delay,
        raw_spectral_radius=radius,
        spectral_scale_factor=1.0,
        final_spectral_radius=radius,
    )


def _bank(
    horizon: int,
    *,
    regime_path: np.ndarray | None = None,
    regime_uniforms: np.ndarray | None = None,
    observation_innovations: np.ndarray | None = None,
    exogenous_inputs: np.ndarray | None = None,
    shocks: np.ndarray | None = None,
    trend_innovations: np.ndarray | None = None,
    scale_innovations: np.ndarray | None = None,
) -> FutureNoiseBank:
    path = np.zeros(horizon, dtype=np.int64) if regime_path is None else regime_path
    trend = np.zeros(horizon, dtype=np.float64) if trend_innovations is None else trend_innovations
    scale = np.zeros(horizon, dtype=np.float64) if scale_innovations is None else scale_innovations
    exogenous = (
        np.empty((horizon, 0), dtype=np.float64) if exogenous_inputs is None else exogenous_inputs
    )
    observation = (
        np.zeros((horizon, 1), dtype=np.float64)
        if observation_innovations is None
        else observation_innovations
    )
    supplied_shocks = np.zeros((horizon, 1), dtype=np.float64) if shocks is None else shocks
    return FutureNoiseBank(
        regime_uniforms=regime_uniforms,
        regime_path=path,
        trend_innovations=trend,
        scale_innovations=scale,
        exogenous_inputs=exogenous,
        observation_innovations=observation,
        shocks=supplied_shocks,
    )


def _replay_arguments(
    schedule: tuple[RegimeDynamics, ...],
    bank: FutureNoiseBank,
    *,
    intervention: CounterfactualIntervention | None = None,
    current_trend: float = 0.0,
    current_scale: float = 0.0,
    initial_history: np.ndarray | None = None,
    trend_history: np.ndarray | None = None,
    causal_delay: int | None = None,
    allocation_metadata: InterventionPair | None = None,
) -> dict[str, object]:
    maximum_state_delay = max(item.nonlinear_delay for item in schedule)
    maximum_trend_delay = max(item.trend_delay for item in schedule)
    state = (
        np.zeros((maximum_state_delay + 1, 1), dtype=np.float64)
        if initial_history is None
        else initial_history
    )
    past_trend = (
        np.zeros(maximum_trend_delay, dtype=np.float64) if trend_history is None else trend_history
    )
    delay = schedule[0].trend_delay if causal_delay is None else causal_delay
    return {
        "initial_history": state,
        "trend_history": past_trend,
        "current_trend": current_trend,
        "current_scale": current_scale,
        "trend_ar_coefficients": np.array([0.0], dtype=np.float64),
        "scale_ar_coefficients": np.array([0.0], dtype=np.float64),
        "dynamics_schedule": schedule,
        "noise_bank": bank,
        "intervention": intervention,
        "trend_loading": np.ones(1, dtype=np.float64),
        "observation_scale_floor": 0.1,
        "causal_delay": delay,
        "allocation_metadata": allocation_metadata,
    }


def _allocation(*, concept: str, concept_delta: float) -> InterventionPair:
    return InterventionPair.build(
        partition=SplitPartition.TEST,
        base_window_id="base-window",
        source_window_id="source-window",
        concept_name=concept,
        regime_relation=RegimeRelation.SAME,
        matching_distance=0.25,
        concept_delta=concept_delta,
    )


def test_future_noise_bank_covers_every_explicit_stochastic_input() -> None:
    uniforms = np.array([0.1, 0.8], dtype=np.float64)
    regimes = np.array([0, 1], dtype=np.int64)
    trend = np.array([0.2, -0.3], dtype=np.float64)
    scale = np.array([-0.4, 0.5], dtype=np.float64)
    exogenous = np.array([[1.0], [2.0]], dtype=np.float64)
    observation = np.array([[3.0], [4.0]], dtype=np.float64)
    shocks = np.array([[5.0], [6.0]], dtype=np.float64)
    sources = (uniforms, regimes, trend, scale, exogenous, observation, shocks)
    snapshots = tuple(array.copy() for array in sources)

    bank = FutureNoiseBank(
        regime_uniforms=uniforms,
        regime_path=regimes,
        trend_innovations=trend,
        scale_innovations=scale,
        exogenous_inputs=exogenous,
        observation_innovations=observation,
        shocks=shocks,
    )

    assert bank.horizon == 2
    for field_name, source, snapshot in zip(
        (
            "regime_uniforms",
            "regime_path",
            "trend_innovations",
            "scale_innovations",
            "exogenous_inputs",
            "observation_innovations",
            "shocks",
        ),
        sources,
        snapshots,
        strict=True,
    ):
        stored = getattr(bank, field_name)
        assert stored is not None
        assert stored.tobytes() == snapshot.tobytes()
        assert not stored.flags.writeable
        assert not np.shares_memory(stored, source)
        np.testing.assert_array_equal(source, snapshot)
    with pytest.raises(FrozenInstanceError):
        bank.trend_innovations = trend  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("allocation", None, ValueError),
        ("regime_uniforms", np.array([0.1, 1.0], dtype=np.float64), ValueError),
        ("regime_path", np.zeros(2, dtype=np.float64), TypeError),
        ("trend_innovations", np.zeros(2, dtype=np.float32), TypeError),
        ("scale_innovations", np.zeros(1, dtype=np.float64), ValueError),
        ("observation_innovations", np.zeros((2, 0), dtype=np.float64), ValueError),
        ("shocks", np.zeros((2, 2), dtype=np.float64), ValueError),
        ("exogenous_inputs", np.zeros((1, 0), dtype=np.float64), ValueError),
    ],
)
def test_future_noise_bank_rejects_invalid_shapes_dtypes_and_values(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "regime_uniforms": np.array([0.1, 0.9], dtype=np.float64),
        "regime_path": np.zeros(2, dtype=np.int64),
        "trend_innovations": np.zeros(2, dtype=np.float64),
        "scale_innovations": np.zeros(2, dtype=np.float64),
        "exogenous_inputs": np.empty((2, 0), dtype=np.float64),
        "observation_innovations": np.zeros((2, 1), dtype=np.float64),
        "shocks": np.zeros((2, 1), dtype=np.float64),
    }
    if field_name == "allocation":
        arguments["regime_path"] = None
        message = "regime_path"
    else:
        arguments[field_name] = value
        message = field_name

    with pytest.raises(error_type, match=message):
        FutureNoiseBank(**arguments)  # type: ignore[arg-type]


def test_fixed_regime_linear_shock_free_replay_matches_analytic_effect() -> None:
    schedule = tuple(_dynamics(linear=0.5, trend_delay=1) for _ in range(3))
    bank = _bank(3)
    intervention = CounterfactualIntervention(concept="trend", source_value=2.0)

    result = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            bank,
            intervention=intervention,
            initial_history=np.array([[4.0]], dtype=np.float64),
            trend_history=np.array([0.0], dtype=np.float64),
            causal_delay=1,
        )
    )

    np.testing.assert_array_equal(
        result.factual_path.full_values,
        np.array([[2.0], [1.0], [0.5]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result.counterfactual_path.full_values,
        np.array([[2.0], [3.0], [1.5]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result.effect,
        np.array([[0.0], [2.0], [1.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(result.horizon_index, np.array([1, 2, 3]))
    assert result.causal_delay == 1
    assert estimate_effect_delay(result.effect) == 1


def test_first_trend_effect_is_exactly_at_delta_plus_one() -> None:
    delta = 2
    schedule = tuple(_dynamics(trend_delay=delta) for _ in range(4))

    result = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            _bank(4),
            intervention=CounterfactualIntervention(
                concept="trend",
                source_value=3.0,
            ),
            causal_delay=delta,
        )
    )
    nonzero_horizons = result.horizon_index[np.any(result.effect != 0.0, axis=1)]

    assert nonzero_horizons[0] == delta + 1
    assert estimate_effect_delay(result.effect) == delta


def test_no_intervention_and_source_equals_base_have_bitwise_zero_effect() -> None:
    schedule = tuple(_dynamics(linear=0.25) for _ in range(3))
    bank = _bank(
        3,
        observation_innovations=np.array([[0.1], [-0.2], [0.3]], dtype=np.float64),
        shocks=np.array([[0.0], [0.5], [-0.25]], dtype=np.float64),
        trend_innovations=np.array([0.2, -0.1, 0.4], dtype=np.float64),
        scale_innovations=np.array([-0.3, 0.5, 0.1], dtype=np.float64),
    )

    no_intervention = replay_paired_counterfactual(
        **_replay_arguments(schedule, bank, current_trend=-0.0)
    )
    equal_source = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            bank,
            current_trend=-0.0,
            intervention=CounterfactualIntervention(
                concept="trend",
                source_value=-0.0,
            ),
        )
    )

    for result in (no_intervention, equal_source):
        assert result.effect.tobytes() == np.zeros((3, 1), dtype=np.float64).tobytes()
        assert (
            result.factual_path.full_values.tobytes()
            == result.counterfactual_path.full_values.tobytes()
        )


def test_factual_self_replay_and_every_paired_bank_array_are_exact() -> None:
    schedule = (
        _dynamics(exogenous_dimension=1, exogenous_loading=0.5),
        _dynamics(linear=0.25, exogenous_dimension=1, exogenous_loading=-0.5),
    )
    bank = _bank(
        2,
        regime_uniforms=np.array([0.2, 0.8], dtype=np.float64),
        exogenous_inputs=np.array([[1.0], [2.0]], dtype=np.float64),
        observation_innovations=np.array([[0.3], [-0.4]], dtype=np.float64),
        shocks=np.array([[0.5], [0.25]], dtype=np.float64),
        trend_innovations=np.array([0.1, 0.2], dtype=np.float64),
        scale_innovations=np.array([-0.1, 0.3], dtype=np.float64),
    )
    result = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            bank,
            intervention=CounterfactualIntervention(
                concept="trend",
                source_value=1.0,
            ),
        )
    )

    replay = rollout_nonlinear_var(
        initial_history=result.factual_path.initial_history,
        trend_history=result.factual_path.trend_history,
        trend_path=result.factual_path.trend_path,
        scale_path=result.factual_path.scale_path,
        dynamics_schedule=result.factual_path.dynamics_schedule,
        regime_labels=result.factual_path.regime_labels,
        exogenous_inputs=result.factual_path.exogenous_inputs,
        observation_innovations=result.factual_path.observation_innovations,
        shocks=result.factual_path.shocks,
        trend_loading=result.factual_path.trend_loading,
        observation_scale_floor=result.factual_path.observation_scale_floor,
    )

    assert replay.full_values.tobytes() == result.factual_path.full_values.tobytes()
    for trajectory in (result.factual_path, result.counterfactual_path):
        assert trajectory.exogenous_inputs.tobytes() == result.noise_bank.exogenous_inputs.tobytes()
        assert (
            trajectory.observation_innovations.tobytes()
            == result.noise_bank.observation_innovations.tobytes()
        )
        assert trajectory.shocks.tobytes() == result.noise_bank.shocks.tobytes()
    for concepts in (result.factual_concepts, result.counterfactual_concepts):
        assert concepts.trend_innovations.tobytes() == result.noise_bank.trend_innovations.tobytes()
        assert concepts.scale_innovations.tobytes() == result.noise_bank.scale_innovations.tobytes()


def test_trend_intervention_leaves_scale_latent_truth_bitwise_unchanged() -> None:
    result = replay_paired_counterfactual(
        **_replay_arguments(
            tuple(_dynamics() for _ in range(3)),
            _bank(
                3,
                scale_innovations=np.array([0.2, -0.4, 0.8], dtype=np.float64),
            ),
            current_scale=-1.0,
            intervention=CounterfactualIntervention(
                concept="trend",
                source_value=2.0,
            ),
        )
    )

    assert result.factual_concepts.scale.tobytes() == result.counterfactual_concepts.scale.tobytes()
    assert (
        result.factual_concepts.scale_innovations.tobytes()
        == result.counterfactual_concepts.scale_innovations.tobytes()
    )


def test_mixed_delay_shifted_schedule_retains_origin_delay_and_per_step_truth() -> None:
    schedule = (_dynamics(linear=0.5, trend_delay=2), _dynamics(linear=0.25), _dynamics())
    result = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            _bank(3),
            initial_history=np.array([[8.0]], dtype=np.float64),
        )
    )

    np.testing.assert_array_equal(
        result.factual_path.full_values,
        np.array([[4.0], [1.0], [0.0]], dtype=np.float64),
    )
    assert result.factual_path.dynamics_schedule == schedule
    assert result.causal_delay == 2


def test_correct_delay_and_concept_controls_are_strictly_better() -> None:
    horizon = 4
    true_schedule = tuple(_dynamics(trend_delay=2) for _ in range(horizon))
    wrong_schedule = tuple(_dynamics(trend_delay=1) for _ in range(horizon))
    bank = _bank(horizon)
    correct = replay_paired_counterfactual(
        **_replay_arguments(
            true_schedule,
            bank,
            intervention=CounterfactualIntervention("trend", 2.0),
            causal_delay=2,
        )
    )
    wrong_delay = replay_paired_counterfactual(
        **_replay_arguments(
            wrong_schedule,
            bank,
            intervention=CounterfactualIntervention("trend", 2.0),
            causal_delay=1,
        )
    )
    random_concept = replay_paired_counterfactual(
        **_replay_arguments(
            true_schedule,
            bank,
            intervention=CounterfactualIntervention("scale", 2.0),
            causal_delay=2,
        )
    )
    target = np.array([[0.0], [0.0], [2.0], [0.0]], dtype=np.float64)

    correct_error = float(np.sqrt(np.mean((correct.effect - target) ** 2)))
    wrong_error = float(np.sqrt(np.mean((wrong_delay.effect - target) ** 2)))
    random_error = float(np.sqrt(np.mean((random_concept.effect - target) ** 2)))
    assert correct_error == 0.0
    assert wrong_error > correct_error
    assert random_error > correct_error


def test_scale_oracle_changes_std_more_than_mean_in_symmetric_fixed_case() -> None:
    innovations = (-2.0, -1.0, 1.0, 2.0)
    banks = tuple(
        _bank(
            1,
            observation_innovations=np.array([[innovation]], dtype=np.float64),
        )
        for innovation in innovations
    )
    schedule = (_dynamics(scale_loading=1.0),)
    arguments = _replay_arguments(
        schedule,
        banks[0],
        intervention=CounterfactualIntervention("scale", 2.0),
    )
    arguments.pop("noise_bank")
    result = monte_carlo_oracle(
        noise_banks=banks,
        quantiles=np.array([0.25, 0.5, 0.75], dtype=np.float64),
        **arguments,
    )

    assert result.mean_effect[0, 0] == pytest.approx(0.0, abs=1e-15)
    assert result.std_effect[0, 0] > abs(result.mean_effect[0, 0])
    assert result.quantile_effects.shape == (3, 1, 1)
    assert np.all(np.diff(result.factual_quantiles[:, 0, 0]) >= 0.0)
    assert np.all(np.diff(result.counterfactual_quantiles[:, 0, 0]) >= 0.0)
    for paired in result.paired_results:
        assert (
            paired.factual_concepts.trend.tobytes()
            == paired.counterfactual_concepts.trend.tobytes()
        )
        assert (
            paired.factual_concepts.trend_innovations.tobytes()
            == paired.counterfactual_concepts.trend_innovations.tobytes()
        )


def test_monte_carlo_retains_samples_and_deterministic_ordered_quantiles() -> None:
    banks = tuple(
        _bank(
            2,
            observation_innovations=np.full((2, 1), value, dtype=np.float64),
        )
        for value in (-1.0, 0.0, 2.0)
    )
    schedule = tuple(_dynamics() for _ in range(2))
    common = _replay_arguments(
        schedule,
        banks[0],
        intervention=CounterfactualIntervention("trend", 1.5),
    )
    common.pop("noise_bank")
    quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float64)

    first = monte_carlo_oracle(
        noise_banks=banks,
        quantiles=quantiles,
        **common,
    )
    second = monte_carlo_oracle(
        noise_banks=banks,
        quantiles=quantiles,
        **common,
    )

    assert len(first.paired_results) == 3
    assert first.factual_paths.shape == (3, 2, 1)
    assert first.counterfactual_paths.shape == (3, 2, 1)
    assert first.sample_effects.shape == (3, 2, 1)
    assert first.mean_effect.shape == (2, 1)
    assert first.std_effect.shape == (2, 1)
    np.testing.assert_array_equal(first.quantiles, quantiles)
    assert np.all(np.diff(first.quantiles) > 0.0)
    for field_name in (
        "factual_paths",
        "counterfactual_paths",
        "sample_effects",
        "mean_effect",
        "std_effect",
        "quantiles",
        "factual_quantiles",
        "counterfactual_quantiles",
        "quantile_effects",
        "horizon_index",
    ):
        first_array = getattr(first, field_name)
        assert first_array.tobytes() == getattr(second, field_name).tobytes()
        assert not first_array.flags.writeable


def test_allocation_metadata_survives_paired_and_aggregate_results() -> None:
    allocation = _allocation(concept="trend", concept_delta=2.0)
    intervention = CounterfactualIntervention("trend", 2.0)
    schedule = (_dynamics(),)
    bank = _bank(1)

    paired = replay_paired_counterfactual(
        **_replay_arguments(
            schedule,
            bank,
            intervention=intervention,
            allocation_metadata=allocation,
        )
    )
    aggregate_args = _replay_arguments(
        schedule,
        bank,
        intervention=intervention,
        allocation_metadata=allocation,
    )
    aggregate_args.pop("noise_bank")
    aggregate = monte_carlo_oracle(
        noise_banks=(bank,),
        quantiles=np.array([0.5], dtype=np.float64),
        **aggregate_args,
    )

    assert paired.allocation_metadata == allocation
    assert aggregate.allocation_metadata == allocation
    assert aggregate.paired_results[0].allocation_metadata == allocation


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    [
        ("dynamics_schedule", (), ValueError, "dynamics_schedule"),
        (
            "dynamics_schedule",
            (_dynamics(regime_label=1), _dynamics(regime_label=1)),
            ValueError,
            "regime_path.*dynamics_schedule",
        ),
        ("causal_delay", 2, ValueError, "causal_delay"),
        ("causal_delay", 1, ValueError, "causal_delay.*schedule"),
        (
            "trend_ar_coefficients",
            np.array([0.0], dtype=np.float32),
            TypeError,
            "trend_ar_coefficients",
        ),
    ],
)
def test_replay_rejects_invalid_schedules_delays_and_dtypes(
    field_name: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    arguments = _replay_arguments((_dynamics(), _dynamics()), _bank(2))
    arguments[field_name] = value

    with pytest.raises(error_type, match=message):
        replay_paired_counterfactual(**arguments)


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.5],
        np.array([0.5], dtype=np.float32),
        np.array([], dtype=np.float64),
        np.array([[0.5]], dtype=np.float64),
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([0.75, 0.25], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    ],
)
def test_monte_carlo_rejects_invalid_quantiles(quantiles: object) -> None:
    bank = _bank(1)
    arguments = _replay_arguments((_dynamics(),), bank)
    arguments.pop("noise_bank")

    with pytest.raises((TypeError, ValueError), match="quantiles"):
        monte_carlo_oracle(
            noise_banks=(bank,),
            quantiles=quantiles,  # type: ignore[arg-type]
            **arguments,
        )


def test_oracle_rejects_empty_or_incompatible_banks() -> None:
    arguments = _replay_arguments((_dynamics(),), _bank(1))
    arguments.pop("noise_bank")

    with pytest.raises(ValueError, match="noise_banks"):
        monte_carlo_oracle(
            noise_banks=(),
            quantiles=np.array([0.5], dtype=np.float64),
            **arguments,
        )
    with pytest.raises(ValueError, match="horizon"):
        monte_carlo_oracle(
            noise_banks=(_bank(1), _bank(2)),
            quantiles=np.array([0.5], dtype=np.float64),
            **arguments,
        )


def test_records_reject_tampered_direct_construction() -> None:
    bank = _bank(2)
    pair = replay_paired_counterfactual(
        **_replay_arguments(
            (_dynamics(), _dynamics()),
            bank,
            intervention=CounterfactualIntervention("trend", 1.0),
            allocation_metadata=_allocation(concept="trend", concept_delta=1.0),
        )
    )
    pair_fields = {field.name: getattr(pair, field.name) for field in fields(pair)}
    tampered_effect = pair.effect.copy()
    tampered_effect[0, 0] += 1.0

    with pytest.raises(ValueError, match="effect"):
        PairedCounterfactualResult(**(pair_fields | {"effect": tampered_effect}))
    forged_intervention = object.__new__(CounterfactualIntervention)
    object.__setattr__(forged_intervention, "concept", "invalid")
    object.__setattr__(forged_intervention, "source_value", 1.0)
    with pytest.raises(ValueError, match=r"intervention.concept"):
        PairedCounterfactualResult(**(pair_fields | {"intervention": forged_intervention}))
    wrong_allocation = _allocation(concept="scale", concept_delta=1.0)
    with pytest.raises(ValueError, match="concept_name"):
        PairedCounterfactualResult(**(pair_fields | {"allocation_metadata": wrong_allocation}))

    oracle_args = _replay_arguments(
        (_dynamics(), _dynamics()),
        bank,
        intervention=CounterfactualIntervention("trend", 1.0),
    )
    oracle_args.pop("noise_bank")
    result = monte_carlo_oracle(
        noise_banks=(bank,),
        quantiles=np.array([0.5], dtype=np.float64),
        **oracle_args,
    )
    result_fields = {field.name: getattr(result, field.name) for field in fields(result)}
    tampered_mean = result.mean_effect.copy()
    tampered_mean[0, 0] += 1.0

    with pytest.raises(ValueError, match="mean_effect"):
        MonteCarloOracleResult(**(result_fields | {"mean_effect": tampered_mean}))
    for field_name in (
        "sample_effects",
        "std_effect",
        "factual_quantiles",
        "quantile_effects",
    ):
        tampered = getattr(result, field_name).copy()
        tampered.flat[0] += 1.0
        with pytest.raises(ValueError, match=field_name):
            MonteCarloOracleResult(**(result_fields | {field_name: tampered}))


def test_paired_result_revalidates_a_forged_noise_bank() -> None:
    pair = replay_paired_counterfactual(**_replay_arguments((_dynamics(),), _bank(1)))
    forged = object.__new__(FutureNoiseBank)
    for field in fields(pair.noise_bank):
        value = getattr(pair.noise_bank, field.name)
        object.__setattr__(forged, field.name, value)
    object.__setattr__(
        forged,
        "observation_innovations",
        np.zeros((1, 1), dtype=np.float32),
    )
    pair_fields = {field.name: getattr(pair, field.name) for field in fields(pair)}

    with pytest.raises(TypeError, match=r"observation_innovations.*float64"):
        PairedCounterfactualResult(**(pair_fields | {"noise_bank": forged}))
    wrong_path = FutureNoiseBank(
        regime_uniforms=None,
        regime_path=np.ones(1, dtype=np.int64),
        trend_innovations=pair.noise_bank.trend_innovations,
        scale_innovations=pair.noise_bank.scale_innovations,
        exogenous_inputs=pair.noise_bank.exogenous_inputs,
        observation_innovations=pair.noise_bank.observation_innovations,
        shocks=pair.noise_bank.shocks,
    )
    with pytest.raises(ValueError, match=r"regime_path.*labels"):
        PairedCounterfactualResult(**(pair_fields | {"noise_bank": wrong_path}))
    forged_path = object.__new__(SyntheticTrajectory)
    for field in fields(pair.factual_path):
        object.__setattr__(forged_path, field.name, getattr(pair.factual_path, field.name))
    object.__setattr__(forged_path, "full_values", np.ones((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="full_values"):
        PairedCounterfactualResult(**(pair_fields | {"factual_path": forged_path}))


def test_oracle_never_mutates_inputs_or_accesses_any_numpy_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = (_dynamics(), _dynamics())
    bank = _bank(
        2,
        trend_innovations=np.array([0.1, 0.2], dtype=np.float64),
        observation_innovations=np.array([[0.3], [0.4]], dtype=np.float64),
    )
    initial_history = np.array([[1.0]], dtype=np.float64)
    trend_loading = np.array([1.0], dtype=np.float64)
    before = tuple(
        array.tobytes() for array in (initial_history, trend_loading, bank.trend_innovations)
    )

    class RandomTrap:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"oracle accessed numpy RNG attribute {name!r}")

    monkeypatch.setattr(oracle_module.np, "random", RandomTrap())
    arguments = _replay_arguments(
        schedule,
        bank,
        intervention=CounterfactualIntervention("trend", 1.0),
        initial_history=initial_history,
    )
    arguments["trend_loading"] = trend_loading
    replay_paired_counterfactual(**arguments)
    arguments.pop("noise_bank")
    monte_carlo_oracle(
        noise_banks=(bank,),
        quantiles=np.array([0.5], dtype=np.float64),
        **arguments,
    )

    assert before == tuple(
        array.tobytes() for array in (initial_history, trend_loading, bank.trend_innovations)
    )


@pytest.mark.parametrize(
    ("effect", "error_type"),
    [
        ([0.0, 1.0], TypeError),
        (np.array([0.0, 1.0], dtype=np.float32), TypeError),
        (np.array([], dtype=np.float64), ValueError),
        (np.array([0.0, np.nan], dtype=np.float64), ValueError),
    ],
)
def test_estimate_effect_delay_rejects_invalid_effects(
    effect: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="effect"):
        estimate_effect_delay(effect)  # type: ignore[arg-type]
