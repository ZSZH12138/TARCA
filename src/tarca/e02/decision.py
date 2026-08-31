from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from tarca.contracts import canonical_json_hash
from tarca.e02.bootstrap import BootstrapInterval
from tarca.e02.config import E02Config
from tarca.e02.scoring import ScoreSummary

E02Outcome = Literal["PASS", "FAIL", "INCONCLUSIVE", "NOT_EVALUABLE"]


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    observed: float | int | str | bool
    required: float | int | str | bool

    def __post_init__(self) -> None:
        if not self.gate_id.strip() or type(self.passed) is not bool:
            raise ValueError("E02 gate identity and result must be valid")


@dataclass(frozen=True, slots=True)
class E02Evidence:
    e02_config_sha256: str
    stage2_freeze_receipt_sha256: str
    score_summary: ScoreSummary
    bootstrap: BootstrapInterval
    completed_trajectories: int
    failed_trajectory_ids: tuple[str, ...]
    integrity_violation_ids: tuple[str, ...]
    finite_probabilities: bool
    positive_scales: bool
    non_crossing_quantiles: bool
    better_than_last_value: bool
    better_than_seasonal_naive: bool
    positive_initializations: int

    def __post_init__(self) -> None:
        for digest in (self.e02_config_sha256, self.stage2_freeze_receipt_sha256):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("E02 evidence identities must be lowercase SHA-256 values")
        if self.completed_trajectories < 0 or self.positive_initializations < 0:
            raise ValueError("E02 evidence counts must be nonnegative")
        for identifiers in (self.failed_trajectory_ids, self.integrity_violation_ids):
            if len(identifiers) != len(set(identifiers)) or any(
                not identifier.strip() for identifier in identifiers
            ):
                raise ValueError("E02 evidence event identities must be unique and nonblank")
        if any(
            type(value) is not bool
            for value in (
                self.finite_probabilities,
                self.positive_scales,
                self.non_crossing_quantiles,
                self.better_than_last_value,
                self.better_than_seasonal_naive,
            )
        ):
            raise ValueError("E02 evidence flags must be boolean")

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def evidence_sha256(self) -> str:
        return canonical_json_hash(self.payload())


@dataclass(frozen=True, slots=True)
class E02Decision:
    outcome: E02Outcome
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        gate_ids = tuple(gate.gate_id for gate in self.gates)
        if not gate_ids or len(gate_ids) != len(set(gate_ids)):
            raise ValueError("E02 decision gate IDs must be nonempty and unique")

    def payload(self) -> dict[str, object]:
        return {"outcome": self.outcome, "gates": [asdict(gate) for gate in self.gates]}

    def decision_sha256(self) -> str:
        return canonical_json_hash(self.payload())


def _finite_summary(summary: ScoreSummary, interval: BootstrapInterval) -> bool:
    values = (
        summary.crps,
        summary.nll,
        summary.mae,
        summary.baseline_crps,
        summary.baseline_nll,
        summary.baseline_mae,
        summary.crps_skill,
        summary.relative_nll,
        summary.relative_mae,
        summary.coverage_error,
        *summary.observed_coverage,
        *(value for _, value in summary.regime_crps_skill),
        *(value for _, value in summary.regime_coverage_error),
        *(value for _, value in summary.secondary_horizon_skill),
        *(value for _, value in summary.data_seed_primary_skill),
        interval.estimate,
        interval.lower,
        interval.upper,
    )
    return all(math.isfinite(value) for value in values)


def _named_value(values: tuple[tuple[str, float], ...], name: str) -> float | None:
    matches = tuple(value for candidate, value in values if candidate == name)
    return matches[0] if len(matches) == 1 else None


def evaluate_e02(evidence: E02Evidence, config: E02Config) -> E02Decision:
    summary = evidence.score_summary
    interval = evidence.bootstrap
    gate = config.gate
    results: list[GateResult] = []

    def add(
        gate_id: str,
        passed: bool,
        observed: float | int | str | bool,
        required: float | int | str | bool,
    ) -> None:
        results.append(GateResult(gate_id, passed, observed, required))

    add(
        "config_identity",
        evidence.e02_config_sha256 == config.scientific_hash(),
        evidence.e02_config_sha256,
        config.scientific_hash(),
    )
    add(
        "integrity_events",
        not evidence.integrity_violation_ids,
        len(evidence.integrity_violation_ids),
        0,
    )
    add(
        "failed_trajectories",
        len(evidence.failed_trajectory_ids) <= gate.allowed_failed_trajectories,
        len(evidence.failed_trajectory_ids),
        gate.allowed_failed_trajectories,
    )
    add(
        "summary_completion_matches",
        summary.trajectory_count == evidence.completed_trajectories,
        summary.trajectory_count,
        evidence.completed_trajectories,
    )
    add(
        "completed_trajectories",
        evidence.completed_trajectories == gate.required_completed_trajectories,
        evidence.completed_trajectories,
        gate.required_completed_trajectories,
    )
    add(
        "finite_probabilities",
        evidence.finite_probabilities and _finite_summary(summary, interval),
        evidence.finite_probabilities and _finite_summary(summary, interval),
        True,
    )
    add("positive_scales", evidence.positive_scales, evidence.positive_scales, True)
    add(
        "non_crossing_quantiles",
        evidence.non_crossing_quantiles,
        evidence.non_crossing_quantiles,
        True,
    )
    estimate_matches = math.isclose(
        interval.estimate,
        summary.crps_skill,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    add(
        "bootstrap_estimate_matches",
        estimate_matches,
        interval.estimate,
        summary.crps_skill,
    )
    add(
        "bootstrap_design",
        (
            interval.replicates == config.bootstrap.replicates
            and interval.units_per_replicate == gate.required_completed_trajectories
            and interval.stratum_count == 10
            and interval.confidence == config.bootstrap.confidence
        ),
        f"{interval.replicates}/{interval.units_per_replicate}/{interval.stratum_count}",
        "5000/120/10",
    )
    add(
        "primary_skill_nonnegative",
        summary.crps_skill >= 0.0,
        summary.crps_skill,
        ">=0",
    )
    add(
        "minimum_primary_crps_skill",
        summary.crps_skill >= gate.minimum_crps_skill,
        summary.crps_skill,
        gate.minimum_crps_skill,
    )
    add(
        "primary_ci_lower",
        interval.lower > gate.ci_lower_strictly_above,
        interval.lower,
        f">{gate.ci_lower_strictly_above}",
    )
    positive_seeds = sum(value > 0.0 for _, value in summary.data_seed_primary_skill)
    add(
        "positive_data_seeds",
        positive_seeds >= gate.minimum_positive_data_seeds
        and len(summary.data_seed_primary_skill) == gate.data_seed_count,
        positive_seeds,
        gate.minimum_positive_data_seeds,
    )
    add(
        "positive_initializations",
        evidence.positive_initializations >= gate.minimum_positive_initializations
        and evidence.positive_initializations <= gate.initialization_count,
        evidence.positive_initializations,
        gate.minimum_positive_initializations,
    )
    add(
        "better_than_last_value",
        evidence.better_than_last_value,
        evidence.better_than_last_value,
        True,
    )
    add(
        "better_than_seasonal_naive",
        evidence.better_than_seasonal_naive,
        evidence.better_than_seasonal_naive,
        True,
    )
    seen = _named_value(summary.regime_crps_skill, "SEEN")
    unseen = _named_value(summary.regime_crps_skill, "UNSEEN")
    add(
        "seen_regime_skill",
        seen is not None and seen > gate.seen_skill_strictly_above,
        seen if seen is not None else "MISSING",
        f">{gate.seen_skill_strictly_above}",
    )
    add(
        "unseen_regime_skill",
        unseen is not None and unseen >= gate.unseen_skill_floor,
        unseen if unseen is not None else "MISSING",
        gate.unseen_skill_floor,
    )
    add(
        "relative_nll",
        summary.relative_nll <= gate.relative_nll_tolerance,
        summary.relative_nll,
        gate.relative_nll_tolerance,
    )
    add(
        "relative_mae",
        summary.relative_mae <= gate.relative_mae_tolerance,
        summary.relative_mae,
        gate.relative_mae_tolerance,
    )
    add(
        "overall_coverage_error",
        summary.coverage_error <= gate.overall_coverage_error_max,
        summary.coverage_error,
        gate.overall_coverage_error_max,
    )
    max_regime_coverage = max(
        (value for _, value in summary.regime_coverage_error), default=math.inf
    )
    add(
        "regime_coverage_error",
        len(summary.regime_coverage_error) == 2
        and max_regime_coverage <= gate.regime_coverage_error_max,
        max_regime_coverage,
        gate.regime_coverage_error_max,
    )
    minimum_secondary = min(
        (value for _, value in summary.secondary_horizon_skill), default=-math.inf
    )
    add(
        "secondary_horizon_skill",
        len(summary.secondary_horizon_skill) == 2
        and minimum_secondary >= gate.secondary_horizon_skill_floor,
        minimum_secondary,
        gate.secondary_horizon_skill_floor,
    )

    hard_fail_ids = {
        "config_identity",
        "integrity_events",
        "failed_trajectories",
        "summary_completion_matches",
        "finite_probabilities",
        "positive_scales",
        "non_crossing_quantiles",
        "bootstrap_estimate_matches",
        "bootstrap_design",
        "primary_skill_nonnegative",
        "better_than_last_value",
        "better_than_seasonal_naive",
        "seen_regime_skill",
        "unseen_regime_skill",
        "relative_nll",
        "relative_mae",
        "overall_coverage_error",
        "regime_coverage_error",
        "secondary_horizon_skill",
    }
    hard_failure = any(
        not result.passed and result.gate_id in hard_fail_ids for result in results
    )
    completion = next(
        result for result in results if result.gate_id == "completed_trajectories"
    )
    if hard_failure:
        outcome: E02Outcome = "FAIL"
    elif not completion.passed:
        outcome = "NOT_EVALUABLE"
    elif all(result.passed for result in results):
        outcome = "PASS"
    elif summary.crps_skill >= 0.0:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "FAIL"
    return E02Decision(outcome=outcome, gates=tuple(results))
