"""Public synthetic-SCM validation and CPU-only E01 engineering smoke."""

from __future__ import annotations

import ctypes
import math
import os
import stat
import time
from dataclasses import fields
from statistics import NormalDist

import numpy as np
import torch

from tarca.contracts import WindowBatch

from . import _validation_core as validation_core
from . import counterfactual_oracle as oracle
from ._validation_core import (
    ConvergencePoint,
    E01EngineeringSmokeReport,
    EngineeringSmokeStatus,
    ValidationIssue,
    ValidationMetric,
    ValidationReport,
    ValidationStatus,
)
from .dataset_builder import PersistedSyntheticDataset, SyntheticDataset
from .nonlinear_var import RegimeDynamics

__all__ = [
    "ConvergencePoint",
    "E01EngineeringSmokeReport",
    "EngineeringSmokeStatus",
    "ValidationIssue",
    "ValidationMetric",
    "ValidationReport",
    "ValidationStatus",
    "run_e01_engineering_smoke",
    "validate_synthetic_dataset",
]

_MC_SIZES = (32, 64, 128, 256)
_QUANTILES = np.array([0.1, 0.5, 0.9], dtype=np.float64)
_LIMITS = {"runtime": 1800.0, "memory": 4 * 1024**3, "output": 2 * 1024**3}
_EASY_CONFIG_HASH = "sha256:0236e94ac2ff6ef9523fd86f80c5c576278dd0817f36cb40a4cc0c839f40601c"
_FLOAT_METRICS = (
    "trend_mean_rmse scale_mean_rmse scale_std_relative_error "
    "conditional_variance_relative_error quantile_normalized_rmse "
    "convergence_log_error_slope correct_signature_distance "
    "wrong_delay_signature_distance wrong_scale_signature_distance "
    "random_concept_signature_distance estimator_variance".split()
)


def validate_synthetic_dataset(
    dataset: SyntheticDataset,
    *,
    persisted: PersistedSyntheticDataset | None = None,
) -> ValidationReport:
    """Validate in-memory truth and, when supplied, its exact nine-file publication."""

    return validation_core.validate_synthetic_dataset(dataset, persisted=persisted)


def _signature_distance(
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    differences = tuple(
        np.asarray(first).ravel() - np.asarray(second).ravel()
        for first, second in zip(candidate, target, strict=True)
    )
    return math.sqrt(
        sum(float(value @ value) for value in differences)
        / sum(value.size for value in differences)
    )


_total_error = lambda mean, std, variance, quantile: math.sqrt(  # noqa: E731
    (mean**2 + std**2 + variance**2 + quantile**2) / 4.0
)


def _analytic_dynamics(delay: int, loading: float) -> RegimeDynamics:
    zero = np.zeros((1, 1), dtype=np.float64)
    return RegimeDynamics(
        regime_label=0,
        linear_matrix=zero,
        nonlinear_matrix=zero,
        exogenous_matrix=np.zeros((1, 0), dtype=np.float64),
        nonlinear_strength=0.0,
        base_log_scale=0.0,
        scale_loading=loading,
        nonlinear_delay=0,
        trend_delay=delay,
        raw_spectral_radius=0.0,
        spectral_scale_factor=1.0,
        final_spectral_radius=0.0,
        stability_target=0.85,
        true_graph=np.zeros((1, 1), dtype=np.bool_),
    )


def _bank(observation: np.ndarray) -> oracle.FutureNoiseBank:
    horizon = observation.size
    return oracle.FutureNoiseBank(
        regime_uniforms=None,
        regime_path=np.zeros(horizon, dtype=np.int64),
        trend_innovations=np.zeros(horizon, dtype=np.float64),
        scale_innovations=np.zeros(horizon, dtype=np.float64),
        exogenous_inputs=np.zeros((horizon, 0), dtype=np.float64),
        observation_innovations=observation.reshape(horizon, 1),
        shocks=np.zeros((horizon, 1), dtype=np.float64),
    )


def _mc(
    banks: tuple[oracle.FutureNoiseBank, ...],
    dynamics: RegimeDynamics,
    intervention: oracle.CounterfactualIntervention,
) -> oracle.MonteCarloOracleResult:
    horizon = banks[0].horizon
    return oracle.monte_carlo_oracle(
        initial_history=np.zeros((1, 1), dtype=np.float64),
        trend_history=np.zeros(dynamics.trend_delay, dtype=np.float64),
        current_trend=0.0,
        current_scale=0.0,
        trend_ar_coefficients=np.zeros(1, dtype=np.float64),
        scale_ar_coefficients=np.zeros(1, dtype=np.float64),
        dynamics_schedules=tuple((dynamics,) * horizon for _ in banks),
        noise_banks=banks,
        intervention=intervention,
        trend_loading=np.ones(1, dtype=np.float64),
        observation_scale_floor=0.01,
        causal_delay=dynamics.trend_delay,
        quantiles=_QUANTILES,
    )


_signature = lambda result: (  # noqa: E731
    result.mean_effect,
    result.std_effect,
    result.quantile_effects,
)
_combine = lambda first, second: tuple(  # noqa: E731
    np.concatenate((left.ravel(), right.ravel())) for left, right in zip(first, second, strict=True)
)


def _memory_estimate(dataset: SyntheticDataset, pairs: int) -> int:
    total = sum(value.nbytes for value in dataset.truth.values())
    for split in dataset.physical_splits:
        for item in fields(WindowBatch):
            value = getattr(split.batch, item.name)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return 2 * (total + pairs * 256 * 20 * 8)


def _available_memory() -> int:
    if os.name != "nt":
        return 0

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("remaining", ctypes.c_ulonglong * 5),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return 0
    return int(status.available_physical) if success else 0


def _output_size(persisted: PersistedSyntheticDataset | None) -> int:
    if persisted is None:
        return 0
    total = 0
    reparse_marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for path in persisted.files.values():
        try:
            info = path.lstat()
        except OSError:
            continue
        is_reparse = stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0) & reparse_marker
        )
        if stat.S_ISREG(info.st_mode) and not is_reparse:
            total += info.st_size
    return total


def _smoke_report(
    dataset: SyntheticDataset,
    validation: ValidationReport,
    issues: tuple[ValidationIssue, ...],
    pairs: int,
    persisted: PersistedSyntheticDataset | None,
    started: float,
    values: dict[str, object] | None = None,
) -> E01EngineeringSmokeReport:
    output = _output_size(persisted)
    configured_delay = dataset.config.true_delay
    true_delay = (
        configured_delay
        if isinstance(configured_delay, int)
        else int(np.min(dataset.truth["resolved_true_delay"]))
    )
    payload: dict[str, object] = {name: 0.0 for name in _FLOAT_METRICS}
    payload.update(
        {
            "status": "ENGINEERING_SMOKE_PASS" if not issues else "ENGINEERING_SMOKE_FAIL",
            "validation": validation,
            "convergence": tuple(ConvergencePoint(size, 0, 0, 0, 0, 0, 0) for size in _MC_SIZES),
            "true_delay": true_delay,
            "estimated_delay": 0,
            "delay_absolute_error": true_delay,
            "pair_count": pairs,
            "mc_sample_sizes": _MC_SIZES,
            "runtime_seconds": time.perf_counter() - started,
            "additional_memory_estimate_bytes": _memory_estimate(dataset, pairs),
            "memory_estimation_method": (
                "deterministic byte accounting for truth+tensors+analytic arrays "
                "with 2x safety factor"
            ),
            "output_size_bytes": output,
            "logical_cpu_count": os.cpu_count() or 1,
            "available_memory_bytes": _available_memory(),
            "gpu_available": bool(torch.cuda.is_available()),
            "gpu_used": False,
            "root_seed": dataset.config.root_seed,
            "git_commit": dataset.synthetic_provenance.git_commit,
            "config_hash": validation.config_hash,
            "data_hash": validation.data_hash,
            "issues": issues,
        }
    )
    payload.update(values or {})
    return E01EngineeringSmokeReport(**payload)  # type: ignore[arg-type]


def _scale_convergence(
    dataset: SyntheticDataset, pairs: int, delta: float, variance_target: float
) -> tuple[tuple[ConvergencePoint, ...], tuple[oracle.MonteCarloOracleResult, ...]]:
    child = np.random.SeedSequence(dataset.config.root_seed).spawn(10)[8]
    samples = tuple(np.random.default_rng(seed).normal(size=256) for seed in child.spawn(pairs))
    bank_sets = tuple(tuple(_bank(np.array([value])) for value in sample) for sample in samples)
    quantiles = delta * np.array([NormalDist().inv_cdf(float(q)) for q in _QUANTILES])
    target, dynamics = quantiles[:, None, None], _analytic_dynamics(0, 1.0)
    intervention = oracle.CounterfactualIntervention("scale", 1.0)
    points, final = [], ()
    for size in _MC_SIZES:
        components, current = [], []
        for banks in bank_sets:
            result = _mc(banks[:size], dynamics, intervention)
            current.append(result)
            observed_variance = np.var(result.counterfactual_paths) - np.var(result.factual_paths)
            components.append(
                (
                    float(np.sqrt(np.mean(result.mean_effect**2))),
                    abs(result.std_effect.item() - delta) / abs(delta),
                    abs(observed_variance - variance_target) / abs(variance_target),
                    float(np.sqrt(np.mean((result.quantile_effects - target) ** 2)) / abs(delta)),
                )
            )
        averaged = np.mean(np.asarray(components), axis=0)
        totals = np.array([_total_error(*item) for item in components])
        points.append(
            ConvergencePoint(
                size,
                *map(float, averaged),
                _total_error(*map(float, averaged)),
                float(np.var(totals)),
            )
        )
        if size == _MC_SIZES[-1]:
            final = tuple(current)
    return tuple(points), final


def run_e01_engineering_smoke(
    dataset: SyntheticDataset,
    *,
    persisted: PersistedSyntheticDataset | None = None,
    mc_sample_sizes: tuple[int, ...] = _MC_SIZES,
    pair_count: int | None = None,
) -> E01EngineeringSmokeReport:
    """Run the fixed easy/CPU engineering smoke; never emit a formal E01 claim."""

    if not isinstance(dataset, SyntheticDataset):
        raise TypeError("dataset: expected SyntheticDataset")
    if not isinstance(mc_sample_sizes, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) for size in mc_sample_sizes
    ):
        raise TypeError("mc_sample_sizes: expected tuple[int, ...]")
    if mc_sample_sizes != _MC_SIZES:
        raise ValueError("mc_sample_sizes: expected exactly (32, 64, 128, 256)")
    pairs = dataset.config.oracle_pairs_smoke if pair_count is None else pair_count
    if isinstance(pairs, bool) or not isinstance(pairs, int):
        raise TypeError("pair_count: expected int or None")
    if not 1 <= pairs <= 16:
        raise ValueError("pair_count: expected [1, 16]")
    started = time.perf_counter()
    validation = validate_synthetic_dataset(dataset, persisted=persisted)
    early = list(validation.issues)
    if dataset.config.name != "synthetic_easy":
        validation_core._add(early, "e01.easy_only", "config.name", "synthetic_easy only")
    elif validation.config_hash != _EASY_CONFIG_HASH:
        validation_core._add(
            early,
            "e01.easy_contract",
            "config",
            "E01 smoke is frozen to the exact synthetic_easy authority config",
        )
    if early:
        return _smoke_report(dataset, validation, tuple(early), pairs, persisted, started)

    true_delay = int(dataset.config.true_delay)
    horizon = true_delay + 2
    zero_banks = tuple(_bank(np.zeros(horizon)) for _ in range(256))
    trend_intervention = oracle.CounterfactualIntervention("trend", 1.0)
    trend = _mc(zero_banks, _analytic_dynamics(true_delay, 1.0), trend_intervention)
    wrong_delay_value = true_delay - 1 if true_delay > 0 else 1
    wrong_delay = _mc(
        zero_banks,
        _analytic_dynamics(wrong_delay_value, 1.0),
        trend_intervention,
    )
    random_seed = np.random.SeedSequence(dataset.config.root_seed).spawn(10)[9]
    random_source = 1.0 + abs(float(np.random.default_rng(random_seed).normal()))
    random_trend = _mc(
        zero_banks,
        _analytic_dynamics(true_delay, 1.0),
        oracle.CounterfactualIntervention("scale", random_source),
    )
    trend_mean = np.zeros((horizon, 1), dtype=np.float64)
    trend_mean[true_delay, 0] = 1.0
    trend_target = (
        trend_mean,
        np.zeros((horizon, 1)),
        np.repeat(trend_mean[None, ...], 3, axis=0),
    )
    delay = oracle.estimate_effect_delay(trend.mean_effect)
    s0, s1 = np.logaddexp(0, 0) + 0.01, np.logaddexp(0, 1) + 0.01
    delta, variance_target = float(s1 - s0), float(s1**2 - s0**2)
    points, final = _scale_convergence(dataset, pairs, delta, variance_target)
    scale_signature = tuple(
        np.mean(np.stack([_signature(result)[index] for result in final]), axis=0)
        for index in range(3)
    )
    epsilon = np.random.default_rng(
        np.random.SeedSequence(dataset.config.root_seed).spawn(10)[8].spawn(1)[0]
    ).normal(size=256)
    scale_banks = tuple(_bank(np.array([value])) for value in epsilon)
    wrong_scale = _mc(
        scale_banks,
        _analytic_dynamics(0, 0.0),
        oracle.CounterfactualIntervention("scale", 1.0),
    )
    random_scale = _mc(
        scale_banks,
        _analytic_dynamics(0, 1.0),
        oracle.CounterfactualIntervention("trend", random_source),
    )
    z = np.array([NormalDist().inv_cdf(float(q)) for q in _QUANTILES])
    scale_target = (np.zeros((1, 1)), np.array([[delta]]), (delta * z)[:, None, None])
    target = _combine(trend_target, scale_target)
    distances = (
        _signature_distance(_combine(_signature(trend), scale_signature), target),
        _signature_distance(_combine(_signature(wrong_delay), scale_signature), target),
        _signature_distance(_combine(_signature(trend), _signature(wrong_scale)), target),
        _signature_distance(_combine(_signature(random_trend), _signature(random_scale)), target),
    )
    errors = np.array([point.total_error for point in points])
    slope = float(np.polyfit(np.log(_MC_SIZES), np.log(errors + 1e-15), 1)[0])
    runtime, memory = time.perf_counter() - started, _memory_estimate(dataset, pairs)
    output = _output_size(persisted)
    trend_rmse = float(np.sqrt(np.mean((trend.mean_effect - trend_mean) ** 2)))
    checks = (
        (trend_rmse <= 1e-12, "e01.trend_mean"),
        (points[-1].std_relative_error <= 0.20, "e01.scale_std"),
        (points[-1].conditional_variance_relative_error <= 0.30, "e01.variance"),
        (points[-1].quantile_normalized_rmse <= 0.35, "e01.quantile"),
        (delay == true_delay, "e01.delay"),
        (errors[-1] < errors[0] and slope < 0, "e01.convergence"),
        (all(value > distances[0] for value in distances[1:]), "e01.controls"),
        (runtime <= _LIMITS["runtime"], "e01.runtime"),
        (memory <= _LIMITS["memory"], "e01.memory"),
        (output <= _LIMITS["output"], "e01.output"),
    )
    issues = [
        ValidationIssue(code, code, "fixed engineering-smoke gate failed")
        for passed, code in checks
        if not passed
    ]
    values = {
        "convergence": points,
        "trend_mean_rmse": trend_rmse,
        "scale_mean_rmse": points[-1].mean_rmse,
        "scale_std_relative_error": points[-1].std_relative_error,
        "conditional_variance_relative_error": points[-1].conditional_variance_relative_error,
        "quantile_normalized_rmse": points[-1].quantile_normalized_rmse,
        "true_delay": true_delay,
        "estimated_delay": delay,
        "delay_absolute_error": abs(delay - true_delay),
        "convergence_log_error_slope": slope,
        "correct_signature_distance": distances[0],
        "wrong_delay_signature_distance": distances[1],
        "wrong_scale_signature_distance": distances[2],
        "random_concept_signature_distance": distances[3],
        "estimator_variance": points[-1].estimator_variance,
    }
    return _smoke_report(dataset, validation, tuple(issues), pairs, persisted, started, values)
