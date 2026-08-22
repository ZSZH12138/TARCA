from __future__ import annotations

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
    for index in range(1, 12)
)


def _world_evidence(
    *,
    world_id: str = "world-a",
    family_id: str = "family-a",
    seed_improvements: tuple[float, float, float] = (0.08, 0.06, 0.04),
) -> WorldGateEvidence:
    rows: list[TrajectoryComparison] = []
    for seed, improvement in zip((101, 103, 107), seed_improvements, strict=True):
        for trajectory in range(8):
            for horizon_group in ("h1_2", "h3_5", "h6_8"):
                rows.append(
                    TrajectoryComparison(
                        seed=seed,
                        neural_adapter="SmallITransformer",
                        trajectory_id=f"{seed}-{trajectory}",
                        regime_id="seen" if trajectory < 4 else "unseen",
                        horizon_group=horizon_group,
                        var_crps=1.0,
                        neural_crps=1.0 - improvement,
                        var_nll=1.1,
                        neural_nll=1.05,
                        var_mae=0.9,
                        neural_mae=0.85,
                        scale_valid=True,
                    )
                )
    return WorldGateEvidence(
        world_id=world_id,
        family_id=family_id,
        role=WorldRole.PRIMARY_MECHANISTIC,
        expected_seeds=(101, 103, 107),
        structural_checks=REQUIRED_CHECKS,
        operable_adapters=("SmallITransformer",),
        comparisons=tuple(rows),
    )


def test_world_gate_passes_stable_operable_neural_advantage() -> None:
    decision = evaluate_world_gate(
        _world_evidence(),
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )

    assert decision.status is GateStatus.PASS
    assert decision.selected_neural_adapter == "SmallITransformer"
    assert decision.bootstrap_interval is not None
    assert decision.bootstrap_interval.lower > 0


def test_world_gate_fails_if_one_seed_loses_to_var() -> None:
    decision = evaluate_world_gate(
        _world_evidence(seed_improvements=(0.08, 0.04, -0.01)),
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )

    assert decision.status is GateStatus.FAIL
    assert "seed_direction" in decision.failed_checks


def test_world_gate_fails_before_scoring_when_structural_truth_is_missing() -> None:
    evidence = _world_evidence()
    evidence = WorldGateEvidence(
        world_id=evidence.world_id,
        family_id=evidence.family_id,
        role=evidence.role,
        expected_seeds=evidence.expected_seeds,
        structural_checks=evidence.structural_checks[:-1],
        operable_adapters=evidence.operable_adapters,
        comparisons=evidence.comparisons,
    )

    decision = evaluate_world_gate(
        evidence,
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )

    assert decision.status is GateStatus.FAIL
    assert "structural_truth" in decision.failed_checks
    assert decision.selected_neural_adapter is None


def test_suite_gate_requires_two_independent_primary_families() -> None:
    first = evaluate_world_gate(
        _world_evidence(world_id="world-a", family_id="same-family"),
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )
    second = evaluate_world_gate(
        _world_evidence(world_id="world-b", family_id="same-family"),
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )

    decision = evaluate_suite_gate(
        SuiteGateEvidence(world_decisions=(first, second)), minimum_primary_families=2
    )

    assert decision.status is GateStatus.FAIL
    assert "independent_primary_families" in decision.failed_checks


def test_linear_control_is_exempt_from_neural_win_but_not_structural_checks() -> None:
    evidence = _world_evidence()
    control = WorldGateEvidence(
        world_id="control",
        family_id="linear",
        role=WorldRole.CONTROL_LINEAR,
        expected_seeds=evidence.expected_seeds,
        structural_checks=REQUIRED_CHECKS,
        operable_adapters=(),
        comparisons=(),
    )

    decision = evaluate_world_gate(
        control,
        bootstrap_replicates=2000,
        confidence_level=0.95,
        guardrail_relative_tolerance=0.02,
    )

    assert decision.status is GateStatus.EXEMPT
