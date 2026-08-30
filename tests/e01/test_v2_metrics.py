from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarca.e01.estimators import analytic_delayed_effect
from tarca.e01.v2_carry_forward import verify_e01_b_carry_forward
from tarca.e01.v2_config import load_e01_v2_config
from tarca.e01.v2_metrics import (
    analyze_e01a_seed,
    curve_multipliers,
    evaluate_e01_v2_gate,
    normal_mean_interval,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_e01_v2_config(ROOT / "configs/e01/e01_v2.yaml")


def test_curve_multiplier_projection_is_hand_derived() -> None:
    truth = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64)
    values = torch.stack((truth * 0.5, truth * 1.5))

    multipliers = curve_multipliers(values, truth)

    assert torch.equal(multipliers, torch.tensor([0.5, 1.5], dtype=torch.float64))


def test_normal_interval_reports_sample_mcse_and_95_percent_bounds() -> None:
    interval = normal_mean_interval(
        torch.tensor([0.5, 1.5], dtype=torch.float64),
        confidence=0.95,
    )

    assert interval.point_estimate == pytest.approx(1.0)
    assert interval.mcse == pytest.approx(0.5)
    assert interval.half_width == pytest.approx(0.979981992270027, rel=1e-12)
    assert interval.lower == pytest.approx(1.0 - interval.half_width)
    assert interval.upper == pytest.approx(1.0 + interval.half_width)


def _passing_effects() -> dict[str, torch.Tensor]:
    world = CONFIG.worlds[0]
    truth = analytic_delayed_effect(
        horizon=CONFIG.horizons[-1],
        true_lag=world.true_lag,
        delta=world.intervention_delta,
        decay=world.decay,
    )
    count = CONFIG.sample_sizes[-1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    gains = 1.0 + 0.35 * torch.randn(count, 1, generator=generator, dtype=torch.float64)
    correct = gains * truth.reshape(1, -1)
    wrong_scm_curve = analytic_delayed_effect(
        horizon=CONFIG.horizons[-1],
        true_lag=world.true_lag,
        delta=world.intervention_delta * 0.55,
        decay=0.25,
    )
    wrong_lag_curve = analytic_delayed_effect(
        horizon=CONFIG.horizons[-1],
        true_lag=world.wrong_lag,
        delta=world.intervention_delta,
        decay=world.decay,
    )
    return {
        "CORRECT_SCM": correct,
        "WRONG_SCM": gains * wrong_scm_curve.reshape(1, -1),
        "WRONG_LAG": gains * wrong_lag_curve.reshape(1, -1),
        "RANDOM_CONCEPT": torch.zeros_like(correct),
        "IDENTITY": torch.zeros_like(correct),
    }


def test_seed_analysis_uses_calibrated_gates_and_keeps_old_ratio_diagnostic_only() -> None:
    report = analyze_e01a_seed(CONFIG, CONFIG.formal_seeds[0], _passing_effects())

    maximum = report["sample_size_statistics"]["8192"]
    assert maximum["covers_truth"] is True
    assert maximum["half_width"] <= CONFIG.gates.interval_half_width_max
    assert report["mcse_ratio"] <= CONFIG.gates.mcse_ratio_max
    assert report["recovered_lag"] == CONFIG.worlds[0].true_lag
    assert report["identity_bitwise_zero"] is True
    assert all(item["pass"] for item in report["controls"].values())
    assert "diagnostic_endpoint_error_ratio" in report
    assert report["seed_gate"]["status"] == "PASS"


def _aggregate_report(seed: int, *, coverage: bool = True, estimate: float = 1.0):
    return {
        "seed": seed,
        "world_id": "analytic_delayed_control_v1",
        "sample_size_statistics": {
            "32": {"point_estimate": 1.0, "mcse": 0.04},
            "8192": {
                "point_estimate": estimate,
                "mcse": 0.003,
                "lower": 0.994 if coverage else 1.001,
                "upper": 1.006 if coverage else 1.013,
                "half_width": 0.006,
                "covers_truth": coverage,
            },
        },
        "mcse_ratio": 0.075,
        "recovered_lag": 3,
        "identity_bitwise_zero": True,
        "controls": {
            control: {"pass": True, "win_fraction": 0.9, "gap_lower": 0.1}
            for control in ("WRONG_SCM", "WRONG_LAG", "RANDOM_CONCEPT")
        },
    }


def test_aggregate_gate_accepts_45_of_50_and_hash_verified_e01_b() -> None:
    reports = tuple(
        _aggregate_report(seed, coverage=index < 45)
        for index, seed in enumerate(CONFIG.formal_seeds)
    )
    carry_forward = verify_e01_b_carry_forward(ROOT, CONFIG)

    gate = evaluate_e01_v2_gate(CONFIG, reports, carry_forward)

    assert gate["coverage_seed_count"] == 45
    assert gate["mcse_ratio_seed_count"] == 50
    assert gate["interval_precision_seed_count"] == 50
    assert gate["directional_seed_counts"] == {
        "WRONG_SCM": 50,
        "WRONG_LAG": 50,
        "RANDOM_CONCEPT": 50,
    }
    assert gate["e01_a_status"] == "PASS"
    assert gate["e01_b_status"] == "PASS"
    assert gate["status"] == "PASS"


def test_aggregate_gate_fails_at_44_coverage_or_excess_bias() -> None:
    carry_forward = verify_e01_b_carry_forward(ROOT, CONFIG)
    low_coverage = tuple(
        _aggregate_report(seed, coverage=index < 44)
        for index, seed in enumerate(CONFIG.formal_seeds)
    )
    biased = tuple(_aggregate_report(seed, estimate=1.006) for seed in CONFIG.formal_seeds)

    coverage_gate = evaluate_e01_v2_gate(CONFIG, low_coverage, carry_forward)
    bias_gate = evaluate_e01_v2_gate(CONFIG, biased, carry_forward)

    assert coverage_gate["coverage_seed_count"] == 44
    assert coverage_gate["status"] == "FAIL"
    assert bias_gate["aggregate_multiplier_bias"] == pytest.approx(0.006)
    assert bias_gate["status"] == "FAIL"


def test_aggregate_gate_rejects_duplicate_or_missing_seed_reports() -> None:
    reports = tuple(_aggregate_report(seed) for seed in CONFIG.formal_seeds[:-1])

    with pytest.raises(ValueError, match="exactly the 50 frozen formal seeds"):
        evaluate_e01_v2_gate(
            CONFIG,
            reports,
            verify_e01_b_carry_forward(ROOT, CONFIG),
        )
