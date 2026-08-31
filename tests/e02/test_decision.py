from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tarca.e02.bootstrap import BootstrapInterval
from tarca.e02.config import load_e02_config
from tarca.e02.decision import E02Evidence, evaluate_e02
from tarca.e02.scoring import ScoreSummary

E02_CONFIG = load_e02_config(Path("configs/e02/e02_v1.yaml"))


def passing_summary() -> ScoreSummary:
    return ScoreSummary(
        trajectory_count=120,
        crps=0.97,
        nll=1.01,
        mae=0.99,
        baseline_crps=1.0,
        baseline_nll=1.0,
        baseline_mae=1.0,
        crps_skill=0.03,
        relative_nll=0.01,
        relative_mae=-0.01,
        coverage_levels=(0.5, 0.8, 0.9, 0.95),
        observed_coverage=(0.49, 0.79, 0.89, 0.94),
        coverage_error=0.01,
        regime_crps_skill=(("SEEN", 0.03), ("UNSEEN", 0.0)),
        regime_coverage_error=(("SEEN", 0.02), ("UNSEEN", 0.03)),
        secondary_horizon_skill=(("h7_12", -0.05), ("h13_24", -0.08)),
        data_seed_primary_skill=(
            (1729, 0.03),
            (2718, 0.02),
            (3141, 0.01),
            (5772, 0.01),
            (8111, -0.01),
        ),
    )


def passing_evidence() -> E02Evidence:
    return E02Evidence(
        e02_config_sha256=E02_CONFIG.scientific_hash(),
        stage2_freeze_receipt_sha256="a" * 64,
        score_summary=passing_summary(),
        bootstrap=BootstrapInterval(0.03, 0.001, 0.06, 0.90, 5000, 120, 10),
        completed_trajectories=120,
        failed_trajectory_ids=(),
        integrity_violation_ids=(),
        finite_probabilities=True,
        positive_scales=True,
        non_crossing_quantiles=True,
        better_than_last_value=True,
        better_than_seasonal_naive=True,
        positive_initializations=2,
    )


@pytest.mark.parametrize(
    ("skill", "ci_lower", "expected"),
    (
        (0.02, 0.0001, "PASS"),
        (0.019999, 0.0001, "INCONCLUSIVE"),
        (0.02, 0.0, "INCONCLUSIVE"),
        (-0.0001, 0.01, "FAIL"),
    ),
)
def test_primary_gate_boundaries(skill: float, ci_lower: float, expected: str) -> None:
    evidence = passing_evidence()
    evidence = replace(
        evidence,
        score_summary=replace(evidence.score_summary, crps_skill=skill),
        bootstrap=replace(evidence.bootstrap, estimate=skill, lower=ci_lower),
    )

    assert evaluate_e02(evidence, E02_CONFIG).outcome == expected


def test_guardrail_equality_boundaries_are_inclusive_except_seen_skill() -> None:
    evidence = passing_evidence()
    summary = replace(
        evidence.score_summary,
        relative_nll=0.05,
        relative_mae=0.05,
        coverage_error=0.05,
        regime_crps_skill=(("SEEN", 0.0001), ("UNSEEN", -0.05)),
        regime_coverage_error=(("SEEN", 0.10), ("UNSEEN", 0.10)),
        secondary_horizon_skill=(("h7_12", -0.10), ("h13_24", -0.10)),
    )

    assert evaluate_e02(replace(evidence, score_summary=summary), E02_CONFIG).outcome == "PASS"
    seen_zero = replace(summary, regime_crps_skill=(("SEEN", 0.0), ("UNSEEN", -0.05)))
    assert (
        evaluate_e02(replace(evidence, score_summary=seen_zero), E02_CONFIG).outcome
        == "FAIL"
    )


def test_integrity_failure_precedes_operational_incompleteness() -> None:
    incomplete = replace(
        passing_evidence(),
        completed_trajectories=119,
        score_summary=replace(passing_summary(), trajectory_count=119),
    )
    assert evaluate_e02(incomplete, E02_CONFIG).outcome == "NOT_EVALUABLE"

    corrupted = replace(incomplete, integrity_violation_ids=("prediction-hash-drift",))
    assert evaluate_e02(corrupted, E02_CONFIG).outcome == "FAIL"


def test_decision_contains_named_gate_results() -> None:
    decision = evaluate_e02(passing_evidence(), E02_CONFIG)

    assert decision.outcome == "PASS"
    assert len({gate.gate_id for gate in decision.gates}) == len(decision.gates)
    assert all(gate.passed for gate in decision.gates)

