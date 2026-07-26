"""Private truth, split, replay, and persistence validation core."""

from __future__ import annotations

import math
import stat
from dataclasses import field, fields, make_dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .dataset_builder import PersistedSyntheticDataset, SyntheticDataset
from .nonlinear_var import RegimeDynamics, rollout_nonlinear_var
from .regimes import sample_regime_sequence

ValidationStatus = Literal["VALIDATION_PASS", "VALIDATION_FAIL"]
EngineeringSmokeStatus = Literal["ENGINEERING_SMOKE_PASS", "ENGINEERING_SMOKE_FAIL"]
_MC_SIZES = (32, 64, 128, 256)
_E01_FLOATS = (
    "trend_mean_rmse scale_mean_rmse scale_std_relative_error "
    "conditional_variance_relative_error quantile_normalized_rmse "
    "convergence_log_error_slope correct_signature_distance "
    "wrong_delay_signature_distance wrong_scale_signature_distance "
    "random_concept_signature_distance estimator_variance runtime_seconds".split()
)
_E01_INTS = (
    "true_delay estimated_delay delay_absolute_error root_seed "
    "additional_memory_estimate_bytes output_size_bytes available_memory_bytes".split()
)
_HASH_LEN = len("sha256:") + 64


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HASH_LEN
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _split_boundaries(total_steps: int) -> tuple[int, int, int, int, int]:
    return (
        0,
        total_steps * 60 // 100,
        total_steps * 80 // 100,
        total_steps * 90 // 100,
        total_steps,
    )


def _number(value: object, name: str, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name}: expected finite real")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 and not signed):
        raise ValueError(f"{name}: expected finite non-negative real")
    return result


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}: expected integer")
    if value < minimum:
        raise ValueError(f"{name}: expected >= {minimum}")
    return value


def _record_post(self) -> None:
    kind = type(self).__name__
    if kind == "ValidationIssue":
        if any(
            not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
            for name in ("code", "location", "message")
        ):
            raise ValueError("issue fields: expected non-empty text")
        return
    if kind == "ValidationMetric":
        object.__setattr__(self, "value", _number(self.value, "value"))
        valid_text = all(
            isinstance(getattr(self, name), str) and getattr(self, name).strip()
            for name in ("name", "unit")
        )
        if not valid_text or not isinstance(self.passed, bool):
            raise TypeError("metric fields: invalid type")
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _number(self.threshold, "threshold"))
        return
    if kind == "ConvergencePoint":
        object.__setattr__(self, "mc_samples", _integer(self.mc_samples, "mc_samples", 1))
        for item in fields(self)[1:]:
            object.__setattr__(self, item.name, _number(getattr(self, item.name), item.name))
        return
    types = {
        "ValidationReport": (("issues", ValidationIssue), ("metrics", ValidationMetric)),
        "E01EngineeringSmokeReport": (
            ("convergence", ConvergencePoint),
            ("issues", ValidationIssue),
        ),
    }[kind]
    for name, item_type in types:
        value = tuple(getattr(self, name))
        if any(not isinstance(item, item_type) for item in value):
            raise TypeError(f"{name}: invalid item type")
        object.__setattr__(self, name, value)
    if kind == "ValidationReport":
        if self.status not in ("VALIDATION_PASS", "VALIDATION_FAIL"):
            raise ValueError("status: invalid validation vocabulary")
        if not isinstance(self.persisted_checked, bool):
            raise TypeError("persisted_checked: expected bool")
        if not _is_sha256_digest(self.config_hash) or not _is_sha256_digest(self.data_hash):
            raise ValueError("config_hash/data_hash: expected sha256:<64 lowercase hex>")
    else:
        sizes = tuple(self.mc_sample_sizes)
        if self.status not in ("ENGINEERING_SMOKE_PASS", "ENGINEERING_SMOKE_FAIL"):
            raise ValueError("status: invalid engineering-smoke vocabulary")
        if not isinstance(self.validation, ValidationReport) or sizes != _MC_SIZES:
            raise TypeError("validation/mc_sample_sizes: invalid")
        point_sizes = tuple(point.mc_samples for point in self.convergence)
        if point_sizes != _MC_SIZES:
            raise ValueError("convergence: expected one point for each fixed MC size")
        for name in _E01_FLOATS:
            signed = name == "convergence_log_error_slope"
            object.__setattr__(self, name, _number(getattr(self, name), name, signed=signed))
        for name in _E01_INTS:
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if (
            isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, int)
            or not 1 <= self.pair_count <= 16
        ):
            raise ValueError("pair_count: expected [1, 16]")
        if (
            isinstance(self.logical_cpu_count, bool)
            or not isinstance(self.logical_cpu_count, int)
            or self.logical_cpu_count < 1
        ):
            raise ValueError("logical_cpu_count: expected >= 1")
        if self.gpu_used is not False:
            raise ValueError("pair_count/GPU: engineering scope exceeded")
        if not isinstance(self.gpu_available, bool):
            raise TypeError("CPU/GPU facts: invalid")
        if not _is_sha256_digest(self.config_hash) or not _is_sha256_digest(self.data_hash):
            raise ValueError("config_hash/data_hash: expected sha256:<64 lowercase hex>")
        if (
            self.config_hash != self.validation.config_hash
            or self.data_hash != self.validation.data_hash
        ):
            raise ValueError("config_hash/data_hash: expected validation report identity")
        if (
            not isinstance(self.memory_estimation_method, str)
            or not self.memory_estimation_method.strip()
        ):
            raise ValueError("memory_estimation_method: expected non-empty text")
        if not (_is_git_commit(self.git_commit) or self.git_commit == "unavailable"):
            raise ValueError("git_commit: expected 40 lowercase hex or unavailable")
        object.__setattr__(self, "mc_sample_sizes", sizes)
    if (self.status.endswith("_PASS")) != (not self.issues):
        raise ValueError("status: must agree with issues")


def _record(name: str, names: str, optional: str = "") -> type:
    schema = [
        (item, object, field(default=None)) if item in optional.split() else (item, object)
        for item in names.split()
    ]
    return make_dataclass(
        name, schema, frozen=True, slots=True, namespace={"__post_init__": _record_post}
    )


ValidationIssue = _record("ValidationIssue", "code location message")
ValidationMetric = _record("ValidationMetric", "name value unit passed threshold", "threshold")
ValidationReport = _record(
    "ValidationReport", "status issues metrics persisted_checked config_hash data_hash"
)
ConvergencePoint = _record(
    "ConvergencePoint",
    "mc_samples mean_rmse std_relative_error conditional_variance_relative_error "
    "quantile_normalized_rmse total_error estimator_variance",
)
E01EngineeringSmokeReport = _record(
    "E01EngineeringSmokeReport",
    "status validation convergence trend_mean_rmse scale_mean_rmse "
    "scale_std_relative_error conditional_variance_relative_error "
    "quantile_normalized_rmse true_delay estimated_delay delay_absolute_error "
    "convergence_log_error_slope correct_signature_distance wrong_delay_signature_distance "
    "wrong_scale_signature_distance random_concept_signature_distance estimator_variance "
    "pair_count mc_sample_sizes runtime_seconds additional_memory_estimate_bytes "
    "memory_estimation_method output_size_bytes logical_cpu_count available_memory_bytes "
    "gpu_available gpu_used root_seed git_commit config_hash data_hash issues",
)


_add = lambda issues, code, location, message: issues.append(  # noqa: E731
    ValidationIssue(code, location, message)
)


def _truth_spec(dataset: SyntheticDataset) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    c, truth = dataset.config, dataset.truth
    n, d, r, u = c.total_steps, c.D, c.regimes, c.generation.exogenous_dimensions
    state_lag = int(np.max(truth["nonlinear_delays"])) + 1
    trend_lag = int(np.max(truth["resolved_true_delay"]))
    f, i, b = np.dtype("float64"), np.dtype("int64"), np.dtype("bool")
    groups = (
        (f, (n + 1,), "trend scale"),
        (
            i,
            (n,),
            "regime_sequence parameter_variant nonlinear_delay_schedule trend_delay_schedule",
        ),
        (
            f,
            (n,),
            "base_log_scale_schedule nonlinear_strength_schedule scale_loading_schedule "
            "raw_spectral_radius_schedule spectral_scale_factor_schedule "
            "final_spectral_radius_schedule trend_noise scale_noise",
        ),
        (f, (n, d), "x_complete observation_noise shock_sequence"),
        (f, (n, u), "exogenous"),
        (b, (n, d), "missing_mask"),
        (f, (n, d, d), "linear_matrix_schedule nonlinear_matrix_schedule"),
        (f, (n, d, u), "exogenous_matrix_schedule"),
        (b, (n, d, d), "true_graph_schedule"),
        (f, (r, r), "transition_matrix"),
        (
            f,
            (r,),
            "initial_probabilities trend_ar_coefficients scale_ar_coefficients "
            "seen_base_log_scales unseen_base_log_scales raw_spectral_radii "
            "spectral_scale_factors final_spectral_radii nonlinear_strengths scale_loadings",
        ),
        (i, (r,), "resolved_true_delay nonlinear_delays"),
        (f, (r, d, d), "linear_matrices nonlinear_matrices"),
        (f, (r, d, u), "exogenous_matrices"),
        (b, (r, d, d), "true_graph"),
        (f, (d,), "trend_loading"),
        (f, (c.burn_in + n,), "regime_uniforms"),
        (f, (state_lag, d), "replay_initial_history"),
        (f, (trend_lag,), "replay_trend_history"),
        (f, (), "observation_scale_floor stability_target"),
    )
    return {name: (dtype, shape) for dtype, shape, names in groups for name in names.split()}


def _validate_truth_arrays(dataset: SyntheticDataset, issues: list[ValidationIssue]) -> bool:
    try:
        spec = _truth_spec(dataset)
    except (KeyError, TypeError, ValueError) as error:
        _add(issues, "truth.shape", "truth", str(error))
        return False
    supplied = set(dataset.truth)
    if supplied != set(spec):
        missing, extra = sorted(set(spec) - supplied), sorted(supplied - set(spec))
        _add(issues, "truth.keys", "truth", f"missing={missing} extra={extra}")
    valid = supplied == set(spec)
    for name in sorted(supplied & set(spec)):
        value, (dtype, shape) = dataset.truth[name], spec[name]
        location = f"truth.{name}"
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            _add(issues, "truth.dtype", location, f"expected {dtype}")
            valid = False
            continue
        checks = (
            (value.shape == shape, "truth.shape", f"expected {shape}, got {value.shape}"),
            (bool(np.all(np.isfinite(value))), "truth.nonfinite", "values must be finite"),
            (not value.flags.writeable, "truth.mutable", "array must be read-only"),
        )
        for passed, code, message in checks:
            if not passed:
                _add(issues, code, location, message)
                valid = False
    return valid


def _validate_semantics(dataset: SyntheticDataset, issues: list[ValidationIssue]) -> None:
    c, t = dataset.config, dataset.truth
    regimes = t["regime_sequence"]
    replayed = sample_regime_sequence(
        t["transition_matrix"], t["initial_probabilities"], t["regime_uniforms"]
    )[c.burn_in :]
    if not np.array_equal(replayed, regimes):
        _add(issues, "regime.replay", "truth.regime_sequence", "full uniform replay differs")
    if np.any(regimes < 0) or np.any(regimes >= c.regimes):
        return
    recurrences = (
        t["trend_ar_coefficients"][regimes] * t["trend"][:-1] + t["trend_noise"],
        t["scale_ar_coefficients"][regimes] * t["scale"][:-1] + t["scale_noise"],
    )
    if not all(
        np.array_equal(expected, t[name][1:])
        for name, expected in zip(("trend", "scale"), recurrences, strict=True)
    ):
        _add(issues, "latent.recurrence", "truth.trend|scale", "recurrence differs")
    bounds = _split_boundaries(c.total_steps)
    variant = np.zeros(c.total_steps, dtype=np.int64)
    variant[bounds[3] :] = 1
    if not np.array_equal(variant, t["parameter_variant"]):
        _add(
            issues,
            "dynamics.unseen_boundary",
            "truth.parameter_variant",
            "unseen variant must start at final 10%",
        )
    supplied_variant = t["parameter_variant"]
    base = np.where(
        supplied_variant == 0,
        t["seen_base_log_scales"][regimes],
        t["unseen_base_log_scales"][regimes],
    )
    schedule_keys = {
        "linear_matrix_schedule": "linear_matrices",
        "nonlinear_matrix_schedule": "nonlinear_matrices",
        "exogenous_matrix_schedule": "exogenous_matrices",
        "nonlinear_strength_schedule": "nonlinear_strengths",
        "nonlinear_delay_schedule": "nonlinear_delays",
        "trend_delay_schedule": "resolved_true_delay",
        "scale_loading_schedule": "scale_loadings",
        "raw_spectral_radius_schedule": "raw_spectral_radii",
        "spectral_scale_factor_schedule": "spectral_scale_factors",
        "final_spectral_radius_schedule": "final_spectral_radii",
    }
    schedule_ok = np.array_equal(t["base_log_scale_schedule"], base)
    schedule_ok &= all(
        np.array_equal(t[schedule], t[parameter][regimes])
        for schedule, parameter in schedule_keys.items()
    )
    if not schedule_ok:
        _add(issues, "dynamics.schedule", "truth.*_schedule", "parameter selection differs")
    graph = (t["linear_matrices"] != 0.0) | (
        (t["nonlinear_strengths"][:, None, None] != 0.0) & (t["nonlinear_matrices"] != 0.0)
    )
    if not np.array_equal(t["true_graph"], graph) or not np.array_equal(
        t["true_graph_schedule"], graph[regimes]
    ):
        _add(issues, "dynamics.true_graph", "truth.true_graph", "active graph differs")
    target = float(t["stability_target"])
    actual = np.array([max(abs(np.linalg.eigvals(matrix))) for matrix in t["linear_matrices"]])
    factor = np.where(t["raw_spectral_radii"] > target, target / t["raw_spectral_radii"], 1.0)
    stable = np.allclose(actual, t["final_spectral_radii"], rtol=1e-12, atol=1e-12)
    stable &= np.allclose(factor, t["spectral_scale_factors"], rtol=1e-12, atol=1e-12)
    stable &= bool(np.all(t["final_spectral_radii"] <= target + 1e-12) and target <= 0.85)
    if not stable:
        _add(issues, "dynamics.stability", "truth.spectral_*", "spectral evidence differs")
    response = np.logaddexp(0.0, base + t["scale_loading_schedule"] * t["scale"][:-1]) + float(
        t["observation_scale_floor"]
    )
    if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
        _add(issues, "scale.response", "truth.scale", "response must be finite and positive")
    if c.missingness_kind == "none" and not np.all(t["missing_mask"]):
        _add(issues, "mask.semantics", "truth.missing_mask", "none must be all observed")


def _dynamics(dataset: SyntheticDataset, regime: int, variant: int) -> RegimeDynamics:
    t = dataset.truth
    scales = t["seen_base_log_scales"] if variant == 0 else t["unseen_base_log_scales"]
    return RegimeDynamics(
        regime_label=regime,
        linear_matrix=t["linear_matrices"][regime],
        nonlinear_matrix=t["nonlinear_matrices"][regime],
        exogenous_matrix=t["exogenous_matrices"][regime],
        nonlinear_strength=float(t["nonlinear_strengths"][regime]),
        base_log_scale=float(scales[regime]),
        scale_loading=float(t["scale_loadings"][regime]),
        nonlinear_delay=int(t["nonlinear_delays"][regime]),
        trend_delay=int(t["resolved_true_delay"][regime]),
        raw_spectral_radius=float(t["raw_spectral_radii"][regime]),
        spectral_scale_factor=float(t["spectral_scale_factors"][regime]),
        final_spectral_radius=float(t["final_spectral_radii"][regime]),
        stability_target=float(t["stability_target"]),
        true_graph=t["true_graph"][regime],
    )


def _validate_replay(dataset: SyntheticDataset, issues: list[ValidationIssue]) -> None:
    t, n, r = dataset.truth, dataset.config.total_steps, dataset.config.regimes
    if np.any(t["regime_sequence"] < 0) or np.any(t["regime_sequence"] >= r):
        return
    if np.any(t["parameter_variant"] < 0) or np.any(t["parameter_variant"] > 1):
        return
    try:
        base = {(i, v): _dynamics(dataset, i, v) for i in range(r) for v in (0, 1)}
        allocation = zip(
            t["regime_sequence"][: n - 1],
            t["parameter_variant"][: n - 1],
            strict=True,
        )
        schedule = tuple(base[(int(i), int(v))] for i, v in allocation)
        replay = rollout_nonlinear_var(
            initial_history=t["replay_initial_history"],
            trend_history=t["replay_trend_history"],
            trend_path=t["trend"][: n - 1],
            scale_path=t["scale"][: n - 1],
            dynamics_schedule=schedule,
            regime_labels=t["regime_sequence"][: n - 1],
            exogenous_inputs=t["exogenous"][: n - 1],
            observation_innovations=t["observation_noise"][: n - 1],
            shocks=t["shock_sequence"][: n - 1],
            trend_loading=t["trend_loading"],
            observation_scale_floor=float(t["observation_scale_floor"]),
            burn_in=0,
        )
        if replay.full_values.tobytes() != t["x_complete"][1:].tobytes():
            raise ValueError("bitwise X_burn replay differs")
    except (TypeError, ValueError) as error:
        _add(issues, "replay.factual", "truth.replay_initial_history", str(error))


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & marker)


def validate_synthetic_dataset(
    dataset: SyntheticDataset,
    *,
    persisted: PersistedSyntheticDataset | None = None,
) -> ValidationReport:
    if not isinstance(dataset, SyntheticDataset):
        raise TypeError("dataset: expected SyntheticDataset")
    if persisted is not None and not isinstance(persisted, PersistedSyntheticDataset):
        raise TypeError("persisted: expected PersistedSyntheticDataset or None")
    from . import _validation_integrity as integrity
    from . import _validation_persistence as persistence

    issues: list[ValidationIssue] = []
    stages: list[tuple[str, bool]] = []

    def stage(name: str, function, *args) -> None:
        before = len(issues)
        function(*args, issues)
        stages.append((name, len(issues) == before))

    structural = _validate_truth_arrays(dataset, issues)
    stages.append(("truth_contract", structural))
    if structural:
        stage("truth_semantics", _validate_semantics, dataset)
        stage("factual_replay", _validate_replay, dataset)
        stage("windows", integrity.validate_windows, dataset)
        stage("identity", integrity.validate_identity, dataset)
        stage("oracle_invariants", integrity.validate_oracle_invariants, dataset)
    if persisted is not None:
        stage("persistence", persistence.validate_persisted, dataset, persisted, _is_reparse)
    metrics = tuple(
        ValidationMetric(name, float(passed), "boolean", passed, 1.0) for name, passed in stages
    )
    config_hash, data_hash = integrity.computed_hashes(dataset)
    return ValidationReport(
        "VALIDATION_PASS" if not issues else "VALIDATION_FAIL",
        tuple(issues),
        metrics,
        persisted is not None,
        config_hash,
        data_hash,
    )
