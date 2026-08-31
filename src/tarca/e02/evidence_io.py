from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tarca.contracts import canonical_json_hash
from tarca.e02.bootstrap import BootstrapInterval
from tarca.e02.decision import E02Decision, E02Evidence, GateResult
from tarca.e02.receipt import build_e02_receipt
from tarca.e02.scoring import ScoreSummary, TrajectoryScore
from tarca.stage1b.evidence_io import write_canonical_json


def trajectory_score(value: dict[str, Any]) -> TrajectoryScore:
    return TrajectoryScore(
        trajectory_id=value["trajectory_id"],
        formal_seed=int(value["formal_seed"]),
        regime=value["regime"],
        origin_count=int(value["origin_count"]),
        horizon_count=int(value["horizon_count"]),
        variable_count=int(value["variable_count"]),
        crps=float(value["crps"]),
        nll=float(value["nll"]),
        mae=float(value["mae"]),
        coverage=tuple((float(level), float(metric)) for level, metric in value["coverage"]),
        horizon_crps=tuple(float(item) for item in value["horizon_crps"]),
        horizon_nll=tuple(float(item) for item in value["horizon_nll"]),
        horizon_mae=tuple(float(item) for item in value["horizon_mae"]),
        horizon_coverage=tuple(
            (float(level), tuple(float(item) for item in metrics))
            for level, metrics in value["horizon_coverage"]
        ),
    )


def score_set(
    payload: dict[str, Any], key: str = "scores"
) -> tuple[TrajectoryScore, ...]:
    return tuple(trajectory_score(item) for item in payload[key])


def score_summary(value: dict[str, Any]) -> ScoreSummary:
    return ScoreSummary(
        trajectory_count=int(value["trajectory_count"]),
        crps=float(value["crps"]),
        nll=float(value["nll"]),
        mae=float(value["mae"]),
        baseline_crps=float(value["baseline_crps"]),
        baseline_nll=float(value["baseline_nll"]),
        baseline_mae=float(value["baseline_mae"]),
        crps_skill=float(value["crps_skill"]),
        relative_nll=float(value["relative_nll"]),
        relative_mae=float(value["relative_mae"]),
        coverage_levels=tuple(float(item) for item in value["coverage_levels"]),
        observed_coverage=tuple(float(item) for item in value["observed_coverage"]),
        coverage_error=float(value["coverage_error"]),
        regime_crps_skill=tuple(
            (str(label), float(metric)) for label, metric in value["regime_crps_skill"]
        ),
        regime_coverage_error=tuple(
            (str(label), float(metric)) for label, metric in value["regime_coverage_error"]
        ),
        secondary_horizon_skill=tuple(
            (str(label), float(metric))
            for label, metric in value["secondary_horizon_skill"]
        ),
        data_seed_primary_skill=tuple(
            (int(seed), float(metric)) for seed, metric in value["data_seed_primary_skill"]
        ),
    )


def evidence(value: dict[str, Any]) -> E02Evidence:
    return E02Evidence(
        e02_config_sha256=value["e02_config_sha256"],
        stage2_freeze_receipt_sha256=value["stage2_freeze_receipt_sha256"],
        score_summary=score_summary(value["score_summary"]),
        bootstrap=BootstrapInterval(**value["bootstrap"]),
        completed_trajectories=int(value["completed_trajectories"]),
        failed_trajectory_ids=tuple(value["failed_trajectory_ids"]),
        integrity_violation_ids=tuple(value["integrity_violation_ids"]),
        finite_probabilities=bool(value["finite_probabilities"]),
        positive_scales=bool(value["positive_scales"]),
        non_crossing_quantiles=bool(value["non_crossing_quantiles"]),
        better_than_last_value=bool(value["better_than_last_value"]),
        better_than_seasonal_naive=bool(value["better_than_seasonal_naive"]),
        positive_initializations=int(value["positive_initializations"]),
    )


def decision(value: dict[str, Any]) -> E02Decision:
    return E02Decision(
        outcome=value["outcome"],
        gates=tuple(GateResult(**item) for item in value["gates"]),
    )


def write_final_e02(
    artifact_root: Path, evidence_value: E02Evidence, decision_value: E02Decision
) -> None:
    destination = artifact_root.resolve() / "frozen/v1"
    receipt = build_e02_receipt(decision_value, evidence_value)
    expected = {
        "e02_evidence.json": evidence_value.payload(),
        "e02_decision.json": decision_value.payload(),
        "e02_receipt.json": receipt.model_dump(mode="json"),
    }
    if destination.exists():
        observed = {
            name: json.loads((destination / name).read_text(encoding="utf-8"))
            for name in expected
        }
        if canonical_json_hash(observed) != canonical_json_hash(expected):
            raise ValueError("existing final E02 evidence does not match this run")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".e02-final-", dir=destination.parent))
    try:
        for name, value in expected.items():
            write_canonical_json(temporary / name, value, replace=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
