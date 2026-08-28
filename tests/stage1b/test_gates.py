from __future__ import annotations

from dataclasses import replace

from tarca.stage1b.config import WorldRole
from tarca.stage1b.gates import (
    GateStatus,
    StructuralCheck,
    SuiteGateEvidence,
    TrajectoryComparison,
    WorldGateEvidence,
    evaluate_suite_gate,
    evaluate_world_gate,
)

REQUIRED_CHECKS = tuple(
    StructuralCheck(check_id=f"WQ-{index:02d}", passed=True, details="test")
    for index in range(1, 13)
)


def _world_evidence(
    *,
    world_id: str = "world-a",
    family_id: str = "family-a",
    winning_units: int = 28,
    total_units: int = 42,
    role: WorldRole = WorldRole.PRIMARY_MECHANISTIC,
) -> WorldGateEvidence:
    rows: list[TrajectoryComparison] = []
    seeds = (101, 103, 107)
    for unit in range(total_units):
        wins = unit < winning_units
        for horizon_group in ("h1_6", "h7_12", "h13_24"):
            rows.append(
                TrajectoryComparison(
                    seed=seeds[unit % len(seeds)],
                    neural_adapter="ITransformerReference",
                    trajectory_id=f"trajectory-{unit}",
                    regime_id="seen" if unit % 2 == 0 else "unseen",
                    horizon_group=horizon_group,
                    var_crps=1.0,
                    neural_crps=0.90 if wins else 1.02,
                    var_nll=1.1,
                    neural_nll=1.05,
                    var_mae=0.9,
                    neural_mae=0.87,
                    var_calibration_error=0.08,
                    neural_calibration_error=0.07,
                    scale_valid=True,
                )
            )
    return WorldGateEvidence(
        world_id=world_id,
        family_id=family_id,
        role=role,
        expected_seeds=seeds,
        structural_checks=REQUIRED_CHECKS,
        operable_adapters=("ITransformerReference",),
        comparisons=tuple(rows),
        unseen_regime_ids=("unseen",),
    )


def _evaluate(evidence: WorldGateEvidence):  # type: ignore[no-untyped-def]
    return evaluate_world_gate(
        evidence,
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.05,
        minimum_comparison_units=40,
        minimum_win_rate=0.65,
        minimum_skill_score=0.0,
        require_seen_and_unseen_majority=True,
    )


def test_world_gate_accepts_high_repeat_win_rate_despite_some_losses() -> None:
    decision = _evaluate(_world_evidence(winning_units=28, total_units=42))
    assert decision.status is GateStatus.PASS
    assert decision.selected_neural_adapter == "ITransformerReference"
    assert decision.comparison_unit_count == 42
    assert decision.win_rate == 28 / 42
    assert decision.skill_score > 0
    assert decision.bootstrap_interval is not None
    assert decision.bootstrap_interval.upper > 0


def test_world_gate_rejects_win_rate_below_preregistered_threshold() -> None:
    decision = _evaluate(_world_evidence(winning_units=27, total_units=42))
    assert decision.status is GateStatus.FAIL
    assert "win_rate" in decision.failed_checks


def test_world_gate_rejects_fewer_than_40_blind_units() -> None:
    decision = _evaluate(_world_evidence(winning_units=39, total_units=39))
    assert decision.status is GateStatus.FAIL
    assert "comparison_unit_count" in decision.failed_checks


def test_confirmation_gate_decides_only_h1_6_with_absolute_calibration() -> None:
    original = _world_evidence(winning_units=42, total_units=42)
    comparisons = tuple(
        replace(
            row,
            neural_crps=0.90 if row.horizon_group == "h1_6" else 1.20,
            neural_calibration_error=0.04,
        )
        for row in original.comparisons
    )
    evidence = replace(original, comparisons=comparisons)

    decision = evaluate_world_gate(
        evidence,
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.05,
        minimum_comparison_units=40,
        minimum_win_rate=0.65,
        minimum_skill_score=0.0,
        require_seen_and_unseen_majority=True,
        primary_horizon_group=(1, 6),
        calibration_guardrail_mode="ABSOLUTE",
        maximum_absolute_calibration_error=0.05,
    )

    assert decision.status is GateStatus.PASS
    assert decision.comparison_unit_count == 42


def test_world_gate_fails_before_scoring_when_structural_truth_is_missing() -> None:
    evidence = _world_evidence()
    incomplete = WorldGateEvidence(
        world_id=evidence.world_id,
        family_id=evidence.family_id,
        role=evidence.role,
        expected_seeds=evidence.expected_seeds,
        structural_checks=evidence.structural_checks[:-1],
        operable_adapters=evidence.operable_adapters,
        comparisons=evidence.comparisons,
        unseen_regime_ids=evidence.unseen_regime_ids,
    )
    decision = _evaluate(incomplete)
    assert decision.status is GateStatus.FAIL
    assert decision.failed_checks == ("structural_truth",)


def test_auxiliary_and_control_worlds_are_exempt_after_structural_checks() -> None:
    for role in (WorldRole.CONTROL_LINEAR, WorldRole.ORACLE_AUXILIARY, WorldRole.FORECAST_STRESS):
        assert _evaluate(_world_evidence(role=role)).status is GateStatus.EXEMPT


def test_suite_gate_needs_one_passing_primary_family_not_every_primary_world() -> None:
    passing = _evaluate(_world_evidence(world_id="pass", family_id="single-scale"))
    failing = _evaluate(
        _world_evidence(world_id="fail", family_id="two-scale", winning_units=10, total_units=42)
    )
    decision = evaluate_suite_gate(
        SuiteGateEvidence(world_decisions=(passing, failing)), minimum_primary_families=1
    )
    assert decision.status is GateStatus.PASS
    assert decision.passed_world_ids == ("pass",)
    assert decision.failed_world_ids == ("fail",)
