from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import torch

from tarca.stage1b.config import WorldRole
from tarca.stage1b.metrics import BootstrapInterval, paired_bootstrap_interval

REQUIRED_STRUCTURAL_CHECKS = frozenset(f"WQ-{index:02d}" for index in range(1, 12))


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
class _CandidateDecision:
    adapter: str
    failed_checks: tuple[str, ...]
    seed_improvements: tuple[tuple[int, float], ...]
    bootstrap_interval: BootstrapInterval


def _mean(rows: tuple[TrajectoryComparison, ...], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


def _relative_guardrail(neural: float, baseline: float, tolerance: float) -> bool:
    return neural <= baseline + tolerance * max(abs(baseline), 1e-8)


def _bootstrap_seed(world_id: str, adapter: str) -> int:
    digest = hashlib.sha256(f"{world_id}|{adapter}|WQ-13".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _evaluate_candidate(
    evidence: WorldGateEvidence,
    adapter: str,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    guardrail_relative_tolerance: float,
) -> _CandidateDecision:
    rows = tuple(row for row in evidence.comparisons if row.neural_adapter == adapter)
    failed: set[str] = set()
    seed_improvements: list[tuple[int, float]] = []
    for seed in evidence.expected_seeds:
        seed_rows = tuple(row for row in rows if row.seed == seed)
        if not seed_rows:
            failed.add("seed_coverage")
            seed_improvements.append((seed, float("-inf")))
            continue
        improvement = _mean(seed_rows, "var_crps") - _mean(seed_rows, "neural_crps")
        seed_improvements.append((seed, improvement))
        if improvement <= 0:
            failed.add("seed_direction")
    unexpected_seeds = {row.seed for row in rows} - set(evidence.expected_seeds)
    if unexpected_seeds:
        failed.add("seed_namespace")

    trajectory_units: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        trajectory_units.setdefault((row.seed, row.trajectory_id), []).append(
            row.var_crps - row.neural_crps
        )
    unit_improvements = torch.tensor(
        [sum(values) / len(values) for values in trajectory_units.values()],
        dtype=torch.float64,
    )
    if unit_improvements.numel() < 2:
        unit_improvements = torch.tensor([float("-inf"), float("-inf")])
    interval = paired_bootstrap_interval(
        unit_improvements,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=_bootstrap_seed(evidence.world_id, adapter),
    )
    if interval.lower <= 0:
        failed.add("bootstrap_lower_bound")

    for horizon_group in {row.horizon_group for row in rows}:
        group_rows = tuple(row for row in rows if row.horizon_group == horizon_group)
        if _mean(group_rows, "var_crps") - _mean(group_rows, "neural_crps") <= 0:
            failed.add("horizon_consistency")

    if rows:
        if not _relative_guardrail(
            _mean(rows, "neural_nll"),
            _mean(rows, "var_nll"),
            guardrail_relative_tolerance,
        ):
            failed.add("nll_guardrail")
        if not _relative_guardrail(
            _mean(rows, "neural_mae"),
            _mean(rows, "var_mae"),
            guardrail_relative_tolerance,
        ):
            failed.add("mae_guardrail")
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
    if evidence.unseen_regime_ids:
        unseen_rows = tuple(row for row in rows if row.regime_id in evidence.unseen_regime_ids)
        unseen_seeds = {row.seed for row in unseen_rows}
        if unseen_seeds != set(evidence.expected_seeds):
            failed.add("unseen_regime_coverage")

    return _CandidateDecision(
        adapter=adapter,
        failed_checks=tuple(sorted(failed)),
        seed_improvements=tuple(seed_improvements),
        bootstrap_interval=interval,
    )


def _structural_truth_passes(checks: tuple[StructuralCheck, ...]) -> bool:
    identifiers = tuple(check.check_id for check in checks)
    return (
        len(identifiers) == len(set(identifiers))
        and set(identifiers) == REQUIRED_STRUCTURAL_CHECKS
        and all(check.passed for check in checks)
    )


def evaluate_world_gate(
    evidence: WorldGateEvidence,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    guardrail_relative_tolerance: float,
) -> WorldGateDecision:
    if not _structural_truth_passes(evidence.structural_checks):
        return WorldGateDecision(
            world_id=evidence.world_id,
            family_id=evidence.family_id,
            role=evidence.role,
            status=GateStatus.FAIL,
            selected_neural_adapter=None,
            failed_checks=("structural_truth",),
            seed_crps_improvements=(),
            bootstrap_interval=None,
        )
    if evidence.role is WorldRole.CONTROL_LINEAR:
        return WorldGateDecision(
            world_id=evidence.world_id,
            family_id=evidence.family_id,
            role=evidence.role,
            status=GateStatus.EXEMPT,
            selected_neural_adapter=None,
            failed_checks=(),
            seed_crps_improvements=(),
            bootstrap_interval=None,
        )
    adapters = tuple(sorted({row.neural_adapter for row in evidence.comparisons}))
    if not adapters:
        return WorldGateDecision(
            world_id=evidence.world_id,
            family_id=evidence.family_id,
            role=evidence.role,
            status=GateStatus.FAIL,
            selected_neural_adapter=None,
            failed_checks=("candidate_coverage",),
            seed_crps_improvements=(),
            bootstrap_interval=None,
        )
    candidates = tuple(
        _evaluate_candidate(
            evidence,
            adapter,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            guardrail_relative_tolerance=guardrail_relative_tolerance,
        )
        for adapter in adapters
    )
    passing = tuple(candidate for candidate in candidates if not candidate.failed_checks)
    if passing:
        selected = max(
            passing,
            key=lambda candidate: candidate.bootstrap_interval.estimate,
        )
        status = GateStatus.PASS
    else:
        selected = max(
            candidates,
            key=lambda candidate: candidate.bootstrap_interval.estimate,
        )
        status = GateStatus.FAIL
    return WorldGateDecision(
        world_id=evidence.world_id,
        family_id=evidence.family_id,
        role=evidence.role,
        status=status,
        selected_neural_adapter=selected.adapter if status is GateStatus.PASS else None,
        failed_checks=selected.failed_checks,
        seed_crps_improvements=selected.seed_improvements,
        bootstrap_interval=selected.bootstrap_interval,
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
    failed_checks: list[str] = []
    if failed or not primary:
        failed_checks.append("all_primary_worlds")
    if len(families) < minimum_primary_families:
        failed_checks.append("independent_primary_families")
    return SuiteGateDecision(
        status=GateStatus.FAIL if failed_checks else GateStatus.PASS,
        passed_world_ids=tuple(decision.world_id for decision in passed),
        failed_world_ids=tuple(decision.world_id for decision in failed),
        primary_families=families,
        failed_checks=tuple(failed_checks),
    )

