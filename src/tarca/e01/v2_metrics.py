from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import torch
from torch import Tensor

from tarca.e01.config import E01Condition
from tarca.e01.metrics import recover_lag
from tarca.e01.v2_carry_forward import E01BCarryForwardReceipt
from tarca.e01.v2_config import E01V2Config

_CONTROL_CONDITIONS: tuple[E01Condition, ...] = (
    "WRONG_SCM",
    "WRONG_LAG",
    "RANDOM_CONCEPT",
)


def _matrix(value: Tensor, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError(f"{label} must be a nonempty rank-2 tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite and floating")
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


def _vector(value: Tensor, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() <= 0:
        raise ValueError(f"{label} must be a nonempty rank-1 tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite and floating")
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


@dataclass(frozen=True, slots=True)
class MeanInterval:
    point_estimate: float
    mcse: float
    lower: float
    upper: float
    half_width: float
    confidence: float
    sample_count: int


def curve_multipliers(values: Tensor, truth: Tensor) -> Tensor:
    curves = _matrix(values, "effect values")
    reference = _vector(truth, "truth curve")
    if curves.shape[1] != reference.numel():
        raise ValueError("effect values and truth curve horizons must match")
    denominator = torch.dot(reference, reference)
    if float(denominator.item()) <= 0.0:
        raise ValueError("truth curve must contain a nonzero effect")
    return curves.mv(reference).div(denominator)


def normal_mean_interval(values: Tensor, confidence: float) -> MeanInterval:
    resolved = _vector(values, "interval values")
    if resolved.numel() < 2:
        raise ValueError("mean interval requires at least two samples")
    if not math.isfinite(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be inside (0.5, 1)")
    point = float(resolved.mean().item())
    mcse = float(resolved.std(correction=1).div(math.sqrt(resolved.numel())).item())
    critical = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    half_width = critical * mcse
    return MeanInterval(
        point_estimate=point,
        mcse=mcse,
        lower=point - half_width,
        upper=point + half_width,
        half_width=half_width,
        confidence=confidence,
        sample_count=resolved.numel(),
    )


def _serializable_interval(interval: MeanInterval, *, truth: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "point_estimate": interval.point_estimate,
        "mcse": interval.mcse,
        "lower": interval.lower,
        "upper": interval.upper,
        "half_width": interval.half_width,
        "confidence": interval.confidence,
        "sample_count": interval.sample_count,
    }
    if truth is not None:
        payload["covers_truth"] = interval.lower <= truth <= interval.upper
    return payload


def _control_evidence(
    correct: Tensor,
    control: Tensor,
    truth: Tensor,
    *,
    confidence: float,
    win_fraction_min: float,
) -> dict[str, Any]:
    correct_errors = (correct - truth.reshape(1, -1)).abs().mean(dim=1)
    control_errors = (control - truth.reshape(1, -1)).abs().mean(dim=1)
    gaps = control_errors - correct_errors
    interval = normal_mean_interval(gaps, confidence)
    win_fraction = float((gaps > 0.0).to(dtype=torch.float64).mean().item())
    passed = win_fraction >= win_fraction_min and interval.lower > 0.0
    return {
        "win_fraction": win_fraction,
        "gap_point_estimate": interval.point_estimate,
        "gap_mcse": interval.mcse,
        "gap_lower": interval.lower,
        "gap_upper": interval.upper,
        "confidence": confidence,
        "pass": passed,
    }


def analyze_e01a_seed(
    config: E01V2Config,
    seed: int,
    effects: Mapping[str, Tensor],
) -> dict[str, Any]:
    if seed not in config.formal_seeds:
        raise ValueError("seed is outside the frozen E01-v2 TEST set")
    if tuple(effects) != config.conditions:
        raise ValueError("effect mapping must contain every frozen condition in order")
    matrices = {
        condition: _matrix(effects[condition], condition) for condition in config.conditions
    }
    shapes = {tuple(value.shape) for value in matrices.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < config.sample_sizes[-1]:
        raise ValueError("condition tensors must share the full frozen sample shape")
    world = config.worlds[0]
    from tarca.e01.estimators import analytic_delayed_effect

    truth = analytic_delayed_effect(
        horizon=config.horizons[-1],
        true_lag=world.true_lag,
        delta=world.intervention_delta,
        decay=world.decay,
    )
    if next(iter(shapes))[1] != truth.numel():
        raise ValueError("condition tensors must cover every frozen horizon")
    correct = matrices["CORRECT_SCM"]
    statistics: dict[str, Any] = {}
    curve_errors: dict[int, float] = {}
    for sample_size in config.sample_sizes:
        prefix = correct[:sample_size]
        interval = normal_mean_interval(
            curve_multipliers(prefix, truth),
            config.gates.confidence,
        )
        curve_error = float((prefix.mean(dim=0) - truth).abs().mean().item())
        curve_errors[sample_size] = curve_error
        statistics[str(sample_size)] = {
            **_serializable_interval(interval, truth=1.0),
            "curve_mean_absolute_error": curve_error,
        }
    first = statistics[str(config.sample_sizes[0])]
    last = statistics[str(config.sample_sizes[-1])]
    first_mcse = float(first["mcse"])
    last_mcse = float(last["mcse"])
    mcse_ratio = (
        0.0
        if first_mcse == 0.0 and last_mcse == 0.0
        else math.inf
        if first_mcse == 0.0
        else last_mcse / first_mcse
    )
    first_error = curve_errors[config.sample_sizes[0]]
    last_error = curve_errors[config.sample_sizes[-1]]
    diagnostic_ratio = (
        0.0
        if first_error == 0.0 and last_error == 0.0
        else math.inf
        if first_error == 0.0
        else last_error / first_error
    )
    maximum = config.sample_sizes[-1]
    controls = {
        control: _control_evidence(
            correct[:maximum],
            matrices[control][:maximum],
            truth,
            confidence=config.gates.confidence,
            win_fraction_min=config.gates.control_win_fraction_min,
        )
        for control in _CONTROL_CONDITIONS
    }
    recovered = recover_lag(correct[:maximum].mean(dim=0))
    identity_zero = torch.equal(
        matrices["IDENTITY"][:maximum],
        torch.zeros_like(matrices["IDENTITY"][:maximum]),
    )
    seed_checks = {
        "coverage_pass": bool(last["covers_truth"]),
        "mcse_ratio_pass": mcse_ratio <= config.gates.mcse_ratio_max,
        "interval_precision_pass": float(last["half_width"])
        <= config.gates.interval_half_width_max,
        "lag_pass": abs(recovered - world.true_lag) <= config.gates.analytic_lag_tolerance_steps,
        "identity_pass": identity_zero,
        "controls_pass": all(item["pass"] for item in controls.values()),
    }
    return {
        "schema_version": "tarca-e01-a-seed-report-v2",
        "world_id": world.world_id,
        "seed": seed,
        "truth_multiplier": 1.0,
        "sample_size_statistics": statistics,
        "mcse_ratio": mcse_ratio,
        "diagnostic_endpoint_error_ratio": diagnostic_ratio,
        "diagnostic_endpoint_error_ratio_is_gate": False,
        "recovered_lag": recovered,
        "identity_bitwise_zero": identity_zero,
        "controls": controls,
        "seed_gate": {
            **seed_checks,
            "status": "PASS" if all(seed_checks.values()) else "FAIL",
        },
    }


def evaluate_e01_v2_gate(
    config: E01V2Config,
    reports: Sequence[Mapping[str, Any]],
    carry_forward: E01BCarryForwardReceipt,
) -> dict[str, Any]:
    if not isinstance(carry_forward, E01BCarryForwardReceipt):
        raise TypeError("E01-B requires a verified carry-forward receipt")
    seeds = tuple(int(report.get("seed", -1)) for report in reports)
    if len(reports) != 50 or set(seeds) != set(config.formal_seeds) or len(set(seeds)) != 50:
        raise ValueError("reports must contain exactly the 50 frozen formal seeds")
    maximum = str(config.sample_sizes[-1])
    coverage = 0
    mcse = 0
    precision = 0
    lag = 0
    identity = 0
    directional = {control: 0 for control in ("WRONG_SCM", "WRONG_LAG", "RANDOM_CONCEPT")}
    estimates: list[float] = []
    for report in reports:
        statistics = report.get("sample_size_statistics")
        if not isinstance(statistics, Mapping) or maximum not in statistics:
            raise ValueError("seed report is missing maximum-sample statistics")
        final = statistics[maximum]
        if not isinstance(final, Mapping):
            raise ValueError("maximum-sample statistics are invalid")
        estimate = float(final["point_estimate"])
        estimates.append(estimate)
        coverage += int(final.get("covers_truth") is True)
        mcse += int(float(report["mcse_ratio"]) <= config.gates.mcse_ratio_max)
        precision += int(float(final["half_width"]) <= config.gates.interval_half_width_max)
        lag += int(
            abs(int(report["recovered_lag"]) - config.worlds[0].true_lag)
            <= config.gates.analytic_lag_tolerance_steps
        )
        identity += int(report.get("identity_bitwise_zero") is True)
        controls = report.get("controls")
        if not isinstance(controls, Mapping):
            raise ValueError("seed report controls are invalid")
        for control in directional:
            evidence = controls.get(control)
            directional[control] += int(
                isinstance(evidence, Mapping) and evidence.get("pass") is True
            )
    aggregate_bias = abs(sum(estimates) / len(estimates) - 1.0)
    required = config.gates.required_seed_count
    checks = {
        "coverage": coverage >= required,
        "mcse_ratio": mcse >= required,
        "interval_precision": precision >= required,
        "aggregate_bias": aggregate_bias <= config.gates.aggregate_multiplier_bias_max,
        "lag": lag >= required,
        "identity": identity >= required,
        "controls": all(count >= required for count in directional.values()),
    }
    e01_a_status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "gate_freeze_id": config.gates.gate_freeze_id,
        "criteria": config.gates.model_dump(mode="json"),
        "coverage_seed_count": coverage,
        "mcse_ratio_seed_count": mcse,
        "interval_precision_seed_count": precision,
        "lag_seed_count": lag,
        "identity_seed_count": identity,
        "directional_seed_counts": directional,
        "aggregate_multiplier_bias": aggregate_bias,
        "e01_a_checks": checks,
        "e01_a_status": e01_a_status,
        "e01_b_status": carry_forward.status,
        "e01_b_carry_forward_receipt_sha256": carry_forward.receipt_sha256,
        "status": "PASS" if e01_a_status == "PASS" and carry_forward.status == "PASS" else "FAIL",
    }
