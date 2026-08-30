from __future__ import annotations

import torch

from tarca.e01.estimators import (
    analytic_delayed_effect,
    paired_difference,
    simulate_delayed_effects,
)


def test_paired_difference_is_exact_zero_for_identical_rollouts() -> None:
    factual = torch.tensor([[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]], dtype=torch.float64)

    effect = paired_difference(factual, factual.clone())

    assert torch.equal(effect, torch.zeros_like(factual))


def test_analytic_delayed_effect_starts_at_the_true_lag() -> None:
    effect = analytic_delayed_effect(horizon=6, true_lag=3, delta=2.0, decay=0.5)

    assert torch.equal(
        effect,
        torch.tensor([0.0, 0.0, 2.0, 1.0, 0.5, 0.25], dtype=torch.float64),
    )


def test_correct_mc_estimator_uses_nested_common_random_numbers() -> None:
    first = simulate_delayed_effects(
        seed=11,
        sample_count=64,
        horizon=8,
        true_lag=3,
        wrong_lag=6,
        delta=1.0,
        condition="CORRECT_SCM",
        device="cpu",
        batch_size=7,
    )
    second = simulate_delayed_effects(
        seed=11,
        sample_count=128,
        horizon=8,
        true_lag=3,
        wrong_lag=6,
        delta=1.0,
        condition="CORRECT_SCM",
        device="cpu",
        batch_size=31,
    )

    assert torch.equal(first.values, second.values[:64])
    assert first.group_ids == tuple(f"base-{index:06d}" for index in range(64))


def test_identity_is_bitwise_zero_and_controls_change_the_effect_shape() -> None:
    arguments = {
        "seed": 23,
        "sample_count": 32,
        "horizon": 8,
        "true_lag": 3,
        "wrong_lag": 6,
        "delta": 1.0,
        "device": "cpu",
        "batch_size": 9,
    }
    identity = simulate_delayed_effects(condition="IDENTITY", **arguments)
    correct = simulate_delayed_effects(condition="CORRECT_SCM", **arguments)
    wrong_scm = simulate_delayed_effects(condition="WRONG_SCM", **arguments)
    wrong_lag = simulate_delayed_effects(condition="WRONG_LAG", **arguments)
    random = simulate_delayed_effects(condition="RANDOM_CONCEPT", **arguments)

    assert torch.equal(identity.values, torch.zeros((32, 8), dtype=torch.float64))
    assert int(torch.argmax(correct.values.abs().mean(dim=0)).item()) + 1 == 3
    assert int(torch.argmax(wrong_lag.values.abs().mean(dim=0)).item()) + 1 == 6
    assert not torch.equal(wrong_scm.values, correct.values)
    assert not torch.equal(random.values, correct.values)


def test_estimator_batching_does_not_change_the_scientific_sample_count() -> None:
    result = simulate_delayed_effects(
        seed=29,
        sample_count=33,
        horizon=6,
        true_lag=2,
        wrong_lag=5,
        delta=0.5,
        condition="CORRECT_SCM",
        device="cpu",
        batch_size=128,
    )

    assert result.values.shape == (33, 6)
