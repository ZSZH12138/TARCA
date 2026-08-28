from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import torch

from tarca.stage1b.config import WorldRole
from tarca.stage1b.metrics import BootstrapInterval, paired_bootstrap_interval

REQUIRED_STRUCTURAL_CHECKS = frozenset(f"WQ-{index:02d}" for index in range(1, 13))


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    EXEMPT = "EXEMPT"


@dataclass(frozen=True, slots=True)
class StructuralCheck:
    check_id: str
    passed: bool
    details: str


@dataclass(frozen=True, slots=True)
class TrajectoryComparison:
    seed: int
    neural_adapter: str
    trajectory_id: str
    regime_id: str
    horizon_group: str
    var_crps: float
    neural_crps: float
    var_nll: float
    neural_nll: float
    var_mae: float
    neural_mae: float
    var_calibration_error: float
    neural_calibration_error: float
    scale_valid: bool


@dataclass(frozen=True, slots=True)
class WorldGateEvidence:
    world_id: str
    family_id: str
    role: WorldRole
    expected_seeds: tuple[int, int, int]
    structural_checks: tuple[StructuralCheck, ...]
    operable_adapters: tuple[str, ...]
    comparisons: tuple[TrajectoryComparison, ...]
    unseen_regime_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorldGateDecision:
    world_id: str
    family_id: str
    role: WorldRole
    status: GateStatus
    selected_neural_adapter: str | None
    failed_checks: tuple[str, ...]
    seed_crps_improvements: tuple[tuple[int, float], ...]
    bootstrap_interval: BootstrapInterval | None
    comparison_unit_count: int
    win_rate: float
    skill_score: float
    seen_win_rate: float
    unseen_win_rate: float


@dataclass(frozen=True, slots=True)
class SuiteGateEvidence:
    world_decisions: tuple[WorldGateDecision, ...]


@dataclass(frozen=True, slots=True)
class SuiteGateDecision:
    status: GateStatus
    passed_world_ids: tuple[str, ...]
    failed_world_ids: tuple[str, ...]
    primary_families: tuple[str, ...]
    failed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ComparisonUnit:
    seed: int
    trajectory_id: str
    regime_id: str
    var_crps: float
    neural_crps: float

    @property
    def improvement(self) -> float:
        return self.var_crps - self.neural_crps

    @property
    def won(self) -> bool:
        return self.improvement > 0.0


@dataclass(frozen=True, slots=True)
class _CandidateDecision:
    adapter: str
    failed_checks: tuple[str, ...]
    seed_improvements: tuple[tuple[int, float], ...]
    bootstrap_interval: BootstrapInterval
    comparison_unit_count: int
    win_rate: float
    skill_score: float
    seen_win_rate: float
    unseen_win_rate: float


def _mean(rows: tuple[TrajectoryComparison, ...], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


def _relative_guardrail(neural: float, baseline: float, tolerance: float) -> bool:
    return neural <= baseline + tolerance * max(abs(baseline), 1e-8)


def _bootstrap_seed(world_id: str, adapter: str) -> int:
    digest = hashlib.sha256(f"{world_id}|{adapter}|WQ-13-v2".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _comparison_units(rows: tuple[TrajectoryComparison, ...]) -> tuple[_ComparisonUnit, ...]:
    grouped: dict[tuple[int, str, str], list[TrajectoryComparison]] = {}
    for row in rows:
        grouped.setdefault((row.seed, row.trajectory_id, row.regime_id), []).append(row)
    return tuple(
        _ComparisonUnit(
            seed=seed,
            trajectory_id=trajectory_id,
            regime_id=regime_id,
            var_crps=sum(row.var_crps for row in group) / len(group),
            neural_crps=sum(row.neural_crps for row in group) / len(group),
        )
        for (seed, trajectory_id, regime_id), group in sorted(grouped.items())
    )


def _unit_win_rate(units: tuple[_ComparisonUnit, ...]) -> float:
    return 0.0 if not units else sum(unit.won for unit in units) / len(units)


def _evaluate_candidate(
    evidence: WorldGateEvidence,
    adapter: str,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    guardrail_relative_tolerance: float,
    minimum_comparison_units: int,
    minimum_win_rate: float,
    minimum_skill_score: float,
    require_seen_and_unseen_majority: bool,
    primary_horizon_group: tuple[int, int] | None,
    calibration_guardrail_mode: str,
    maximum_absolute_calibration_error: float | None,
) -> _CandidateDecision:
    horizon_label = (
        None
        if primary_horizon_group is None
        else f"h{primary_horizon_group[0]}_{primary_horizon_group[1]}"
    )
    rows = tuple(
        row
        for row in evidence.comparisons
        if row.neural_adapter == adapter
        and (horizon_label is None or row.horizon_group == horizon_label)
    )
    units = _comparison_units(rows)
    failed: set[str] = set()
    if len(units) < minimum_comparison_units:
        failed.add("comparison_unit_count")
    unexpected_seeds = {row.seed for row in rows} - set(evidence.expected_seeds)
    covered_seeds = {row.seed for row in rows}
    if unexpected_seeds:
        failed.add("seed_namespace")
    if covered_seeds != set(evidence.expected_seeds):
        failed.add("seed_coverage")

    seed_improvements = tuple(
        (
            seed,
            sum(unit.improvement for unit in units if unit.seed == seed)
            / max(1, sum(unit.seed == seed for unit in units)),
        )
        for seed in evidence.expected_seeds
    )
    win_rate = _unit_win_rate(units)
    if win_rate < minimum_win_rate:
        failed.add("win_rate")
    baseline_total = sum(unit.var_crps for unit in units)
    skill_score = (
        float("-inf")
        if baseline_total <= 0.0
        else 1.0 - sum(unit.neural_crps for unit in units) / baseline_total
    )
    if skill_score <= minimum_skill_score:
        failed.add("skill_score")

    if len(units) >= 2:
        interval = paired_bootstrap_interval(
            torch.tensor([unit.improvement for unit in units], dtype=torch.float64),
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=_bootstrap_seed(evidence.world_id, adapter),
        )
    else:
        interval = BootstrapInterval(
            estimate=float("-inf"),
            lower=float("-inf"),
            upper=float("-inf"),
            confidence_level=confidence_level,
            replicates=bootstrap_replicates,
            unit_count=len(units),
        )
    if interval.upper <= 0.0:
        failed.add("stable_overall_inferiority")

    unseen_ids = set(evidence.unseen_regime_ids)
    unseen_units = tuple(unit for unit in units if unit.regime_id in unseen_ids)
    seen_units = tuple(unit for unit in units if unit.regime_id not in unseen_ids)
    seen_win_rate = _unit_win_rate(seen_units)
    unseen_win_rate = _unit_win_rate(unseen_units)
    if unseen_ids and {row.seed for row in rows if row.regime_id in unseen_ids} != set(
        evidence.expected_seeds
    ):
        failed.add("unseen_regime_coverage")
    if require_seen_and_unseen_majority and (
        not seen_units or not unseen_units or seen_win_rate <= 0.5 or unseen_win_rate <= 0.5
    ):
        failed.add("seen_unseen_majority")

    if rows:
        for neural_field, var_field, check in (
            ("neural_nll", "var_nll", "nll_guardrail"),
            ("neural_mae", "var_mae", "mae_guardrail"),
        ):
            if not _relative_guardrail(
                _mean(rows, neural_field),
                _mean(rows, var_field),
                guardrail_relative_tolerance,
            ):
                failed.add(check)
        neural_calibration = _mean(rows, "neural_calibration_error")
        if calibration_guardrail_mode == "ABSOLUTE":
            if (
                maximum_absolute_calibration_error is None
                or neural_calibration > maximum_absolute_calibration_error
            ):
                failed.add("calibration_guardrail")
        elif not _relative_guardrail(
            neural_calibration,
            _mean(rows, "var_calibration_error"),
            guardrail_relative_tolerance,
        ):
            failed.add("calibration_guardrail")
        for regime_id in {row.regime_id for row in rows}:
            regime_rows = tuple(row for row in rows if row.regime_id == regime_id)
            if not _relative_guardrail(
                _mean(regime_rows, "neural_crps"),
                _mean(regime_rows, "var_crps"),
                guardrail_relative_tolerance,
            ):
                failed.add("worst_regime_guardrail")
    if not rows or not all(row.scale_valid for row in rows):
        failed.add("probabilistic_scale")
    if adapter not in evidence.operable_adapters:
        failed.add("mechanistic_operability")

    return _CandidateDecision(
        adapter=adapter,
        failed_checks=tuple(sorted(failed)),
        seed_improvements=seed_improvements,
        bootstrap_interval=interval,
        comparison_unit_count=len(units),
        win_rate=win_rate,
        skill_score=skill_score,
        seen_win_rate=seen_win_rate,
        unseen_win_rate=unseen_win_rate,
    )


def _structural_truth_passes(checks: tuple[StructuralCheck, ...]) -> bool:
    identifiers = tuple(check.check_id for check in checks)
    return (
        len(identifiers) == len(set(identifiers))
        and set(identifiers) == REQUIRED_STRUCTURAL_CHECKS
        and all(check.passed for check in checks)
    )


def _empty_decision(
    evidence: WorldGateEvidence, status: GateStatus, failed: tuple[str, ...]
) -> WorldGateDecision:
    return WorldGateDecision(
        world_id=evidence.world_id,
        family_id=evidence.family_id,
        role=evidence.role,
        status=status,
        selected_neural_adapter=None,
        failed_checks=failed,
        seed_crps_improvements=(),
        bootstrap_interval=None,
        comparison_unit_count=0,
        win_rate=0.0,
        skill_score=0.0,
        seen_win_rate=0.0,
        unseen_win_rate=0.0,
    )


def evaluate_world_gate(
    evidence: WorldGateEvidence,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    guardrail_relative_tolerance: float,
    minimum_comparison_units: int = 40,
    minimum_win_rate: float = 0.65,
    minimum_skill_score: float = 0.0,
    require_seen_and_unseen_majority: bool = True,
    primary_horizon_group: tuple[int, int] | None = None,
    calibration_guardrail_mode: str = "RELATIVE_TO_VAR",
    maximum_absolute_calibration_error: float | None = None,
) -> WorldGateDecision:
    if not _structural_truth_passes(evidence.structural_checks):
        return _empty_decision(evidence, GateStatus.FAIL, ("structural_truth",))
    if evidence.role is not WorldRole.PRIMARY_MECHANISTIC:
        return _empty_decision(evidence, GateStatus.EXEMPT, ())
    adapters = tuple(sorted({row.neural_adapter for row in evidence.comparisons}))
    if not adapters:
        return _empty_decision(evidence, GateStatus.FAIL, ("candidate_coverage",))
    candidates = tuple(
        _evaluate_candidate(
            evidence,
            adapter,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            guardrail_relative_tolerance=guardrail_relative_tolerance,
            minimum_comparison_units=minimum_comparison_units,
            minimum_win_rate=minimum_win_rate,
            minimum_skill_score=minimum_skill_score,
            require_seen_and_unseen_majority=require_seen_and_unseen_majority,
            primary_horizon_group=primary_horizon_group,
            calibration_guardrail_mode=calibration_guardrail_mode,
            maximum_absolute_calibration_error=maximum_absolute_calibration_error,
        )
        for adapter in adapters
    )
    passing = tuple(candidate for candidate in candidates if not candidate.failed_checks)
    selected = max(passing or candidates, key=lambda candidate: candidate.skill_score)
    status = GateStatus.PASS if passing else GateStatus.FAIL
    return WorldGateDecision(
        world_id=evidence.world_id,
        family_id=evidence.family_id,
        role=evidence.role,
        status=status,
        selected_neural_adapter=selected.adapter if status is GateStatus.PASS else None,
        failed_checks=selected.failed_checks,
        seed_crps_improvements=selected.seed_improvements,
        bootstrap_interval=selected.bootstrap_interval,
        comparison_unit_count=selected.comparison_unit_count,
        win_rate=selected.win_rate,
        skill_score=selected.skill_score,
        seen_win_rate=selected.seen_win_rate,
        unseen_win_rate=selected.unseen_win_rate,
    )


def evaluate_suite_gate(
    evidence: SuiteGateEvidence,
    *,
    minimum_primary_families: int,
) -> SuiteGateDecision:
    primary = tuple(
        decision
        for decision in evidence.world_decisions
        if decision.role is WorldRole.PRIMARY_MECHANISTIC
    )
    passed = tuple(decision for decision in primary if decision.status is GateStatus.PASS)
    failed = tuple(decision for decision in primary if decision.status is not GateStatus.PASS)
    families = tuple(sorted({decision.family_id for decision in passed}))
    failed_checks = (
        ("independent_primary_families",) if len(families) < minimum_primary_families else ()
    )
    return SuiteGateDecision(
        status=GateStatus.FAIL if failed_checks else GateStatus.PASS,
        passed_world_ids=tuple(decision.world_id for decision in passed),
        failed_world_ids=tuple(decision.world_id for decision in failed),
        primary_families=families,
        failed_checks=failed_checks,
    )
