"""Independent split, identity, persistence, and oracle integrity checks."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import torch

from tarca.contracts import (
    CONTRACT_SCHEMA_VERSION,
    DataManifest,
    DataSplitSummary,
    SplitPartition,
    WindowContractSummary,
)

from . import counterfactual_oracle as oracle
from ._validation_core import ValidationIssue, _add
from .dataset_builder import SyntheticDataset
from .nonlinear_var import RegimeDynamics
from .regimes import RANDOM_STREAM_NAMES

_SPLITS = ("train", "validation", "test_seen_regime", "test_unseen_regime")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _split_boundaries(total_steps: int) -> tuple[int, ...]:
    return (
        0,
        total_steps * 60 // 100,
        total_steps * 80 // 100,
        total_steps * 90 // 100,
        total_steps,
    )


def _arrow_metadata(
    name: str,
    history: int,
    dimension: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "physical_split": name,
        "tensor_dtype": "float64",
        "x_shape": [history, dimension],
        "y_shape": [horizon, dimension],
    }


def _dataset_hash(dataset: SyntheticDataset, config_hash: str) -> str:
    values = [config_hash.encode()]
    for name in sorted(dataset.truth):
        array = dataset.truth[name]
        values.extend(
            (
                name.encode(),
                array.dtype.str.encode(),
                _canonical(array.shape),
                array.tobytes(order="C"),
            )
        )
    values.append(_canonical(dataset.normalization.model_dump(mode="json")))
    values.extend(split.split_hash.encode() for split in dataset.physical_splits)
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"


def _expected_normalization(
    dataset: SyntheticDataset,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    config = dataset.config
    train_stop = _split_boundaries(config.total_steps)[1]
    fitted = dataset.truth["x_complete"][:train_stop]
    mean = np.mean(fitted, axis=0)
    raw_std = np.std(fitted, axis=0)
    scale = np.where(raw_std > config.normalization.epsilon, raw_std, 1.0)
    expected = {
        "epsilon": config.normalization.epsilon,
        "fill_value": config.normalization.fill_value,
        "mean": [float(value) for value in mean],
        "raw_std": [float(value) for value in raw_std],
        "scale": [float(value) for value in scale],
        "zero_variance_indices": [
            int(value) for value in np.flatnonzero(raw_std <= config.normalization.epsilon)
        ],
        "fit_start": 0,
        "fit_stop": train_stop,
    }
    return expected, mean, scale


def _tensor_matches(actual: object, expected: np.ndarray) -> bool:
    return (
        isinstance(actual, torch.Tensor)
        and actual.device.type == "cpu"
        and torch.equal(actual, torch.from_numpy(np.ascontiguousarray(expected)))
    )


def _batch_matches(
    dataset: SyntheticDataset,
    split_index: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> bool:
    config = dataset.config
    truth = dataset.truth
    split = dataset.physical_splits[split_index]
    bounds = _split_boundaries(config.total_steps)
    start, stop = bounds[split_index : split_index + 2]
    origins = np.arange(start + config.L - 1, stop - config.H, dtype=np.int64)
    if split.name != _SPLITS[split_index] or split.start != start or split.stop != stop:
        return False
    if not np.array_equal(split.prediction_origins, origins):
        return False
    history = origins[:, None] + np.arange(-config.L + 1, 1)
    future = origins[:, None] + np.arange(1, config.H + 1)
    future_covariates = origins[:, None] + np.arange(config.H)
    x_mask = truth["missing_mask"][history]
    y_mask = truth["missing_mask"][future]
    fill = config.normalization.fill_value
    expected_tensors = {
        "x": np.where(x_mask, (truth["x_complete"][history] - mean) / scale, fill),
        "y": np.where(y_mask, (truth["x_complete"][future] - mean) / scale, fill),
        "observed_covariates": truth["exogenous"][history],
        "known_future_covariates": truth["exogenous"][future_covariates],
        "x_observed_mask": x_mask,
        "y_observed_mask": y_mask,
        "observed_covariates_mask": np.ones(
            (*history.shape, config.generation.exogenous_dimensions),
            dtype=np.bool_,
        ),
        "known_future_covariates_mask": np.ones(
            (*future.shape, config.generation.exogenous_dimensions),
            dtype=np.bool_,
        ),
        "regime": truth["regime_sequence"][origins],
    }
    batch = split.batch
    if not all(
        _tensor_matches(getattr(batch, name), value) for name, value in expected_tensors.items()
    ):
        return False
    features = tuple(f"x{index}" for index in range(config.D))
    covariates = tuple(f"u{index}" for index in range(config.generation.exogenous_dimensions))
    origin_times = tuple(_EPOCH + timedelta(hours=int(origin)) for origin in origins)
    metadata = _arrow_metadata(split.name, config.L, config.D, config.H)
    metadata["x_shape"] = tuple(metadata["x_shape"])
    metadata["y_shape"] = tuple(metadata["y_shape"])
    expected_values = {
        "window_id": tuple(
            f"{config.name}:{dataset.config_hash[7:19]}:{split.name}:{int(origin):08d}"
            for origin in origins
        ),
        "input_feature_names": features,
        "target_names": features,
        "observed_covariate_names": covariates,
        "known_future_covariate_names": covariates,
        "feature_start": tuple(value - timedelta(hours=config.L - 1) for value in origin_times),
        "feature_end": origin_times,
        "prediction_start": tuple(value + timedelta(hours=1) for value in origin_times),
        "label_end": tuple(value + timedelta(hours=config.H) for value in origin_times),
        "forecast_time": tuple(
            tuple(value + timedelta(hours=horizon) for horizon in range(1, config.H + 1))
            for value in origin_times
        ),
        "metadata": metadata,
    }
    return all(getattr(batch, name) == value for name, value in expected_values.items())


def validate_windows(dataset: SyntheticDataset, issues: list[ValidationIssue]) -> None:
    """Independently derive train normalization, split geometry, and every batch field."""

    expected_normalization, mean, scale = _expected_normalization(dataset)
    if dataset.normalization.model_dump(mode="json") != expected_normalization:
        _add(
            issues,
            "normalization.train_only",
            "normalization",
            "differs from complete train interval",
        )
    if (
        len(dataset.physical_splits) != 4
        or tuple(split.name for split in dataset.physical_splits) != _SPLITS
    ):
        _add(issues, "split.contract", "splits", "expected four ordered physical splits")
        return
    for index, split in enumerate(dataset.physical_splits):
        if not _batch_matches(dataset, index, mean, scale):
            _add(
                issues,
                "split.standardization",
                f"split.{split.name}.batch",
                "independent truth/scaler/window reconstruction differs",
            )


def _expected_manifest(dataset: SyntheticDataset) -> DataManifest:
    config = dataset.config
    splits = dataset.physical_splits
    features = tuple(f"x{index}" for index in range(config.D))
    covariates = tuple(f"u{index}" for index in range(config.generation.exogenous_dimensions))
    test_hash = _digest(
        _canonical(
            {
                splits[2].name: splits[2].split_hash,
                splits[3].name: splits[3].split_hash,
            }
        )
    )
    return DataManifest(
        dataset_name=config.name,
        dataset_version="stage1b-v1",
        dataset_hash=dataset.dataset_hash,
        splits=(
            DataSplitSummary(
                partition=SplitPartition.TRAIN,
                split_hash=splits[0].split_hash,
                count=len(splits[0].batch.window_id),
            ),
            DataSplitSummary(
                partition=SplitPartition.VALIDATION,
                split_hash=splits[1].split_hash,
                count=len(splits[1].batch.window_id),
            ),
            DataSplitSummary(
                partition=SplitPartition.TEST,
                split_hash=test_hash,
                count=sum(len(split.batch.window_id) for split in splits[2:]),
            ),
        ),
        window_contract=WindowContractSummary(
            history_length=config.L,
            horizon=config.H,
            input_feature_names=features,
            target_names=features,
            observed_covariate_names=covariates,
            known_future_covariate_names=covariates,
            timezone="UTC",
            missingness_protocol=config.missingness_kind,
        ),
        source_description="TARCA Stage 1B deterministic synthetic SCM",
        created_at=dataset.data_manifest.created_at,
    )


def _provenance_matches(dataset: SyntheticDataset) -> bool:
    config, truth = dataset.config, dataset.truth
    provenance = dataset.synthetic_provenance
    bounds = _split_boundaries(config.total_steps)
    children = np.random.SeedSequence(config.root_seed).spawn(len(RANDOM_STREAM_NAMES))
    streams = tuple(
        (
            name,
            child.spawn_key,
            f"seed:{config.root_seed}:spawn:{'.'.join(map(str, child.spawn_key))}",
        )
        for name, child in zip(RANDOM_STREAM_NAMES, children, strict=True)
    )
    actual_streams = tuple(
        (record.name, record.spawn_key, record.reproducible_id)
        for record in provenance.random_streams
    )
    splits = tuple(
        (
            record.name,
            record.start,
            record.stop,
            record.count,
            record.split_hash,
        )
        for record in provenance.physical_splits
    )
    expected_splits = tuple(
        (
            split.name,
            split.start,
            split.stop,
            len(split.batch.window_id),
            split.split_hash,
        )
        for split in dataset.physical_splits
    )
    summary = provenance.parameter_summary
    summary_expected = {
        "dimension": config.D,
        "regimes": config.regimes,
        "exogenous_dimensions": config.generation.exogenous_dimensions,
        "resolved_true_delay": tuple(map(int, truth["resolved_true_delay"])),
        "stability_target": config.generation.spectral_radius_target,
        "transition_matrix": tuple(
            tuple(float(value) for value in row) for row in truth["transition_matrix"]
        ),
        "nonlinear_delays": tuple(map(int, truth["nonlinear_delays"])),
        "trend_support_size": int(np.count_nonzero(truth["trend_loading"])),
        "shock_rate": config.generation.shock_rate,
        "shock_scale": config.generation.shock_scale,
        "scale_floor": config.generation.scale_floor,
        "observation_noise_distribution": "standard_normal",
    }
    vector_fields = {
        "initial_probabilities": "initial_probabilities",
        "trend_ar_coefficients": "trend_ar_coefficients",
        "scale_ar_coefficients": "scale_ar_coefficients",
        "seen_base_log_scales": "seen_base_log_scales",
        "unseen_base_log_scales": "unseen_base_log_scales",
        "scale_loadings": "scale_loadings",
        "raw_spectral_radii": "raw_spectral_radii",
        "scale_factors": "spectral_scale_factors",
        "final_spectral_radii": "final_spectral_radii",
    }
    summary_expected.update(
        {field: tuple(map(float, truth[source])) for field, source in vector_fields.items()}
    )
    commit_valid = (
        provenance.git_commit == "unavailable"
        and isinstance(provenance.git_commit_status, str)
        and bool(provenance.git_commit_status.strip())
    ) or (
        bool(_COMMIT.fullmatch(provenance.git_commit))
        and provenance.git_commit_status == "resolved"
    )
    generated_at = provenance.generated_at
    utc_valid = (
        generated_at == dataset.data_manifest.created_at
        and generated_at.utcoffset() == timedelta(0)
    )
    software_valid = all(
        isinstance(getattr(provenance.software_versions, name), str)
        and bool(getattr(provenance.software_versions, name).strip())
        for name in ("python", "tarca", "numpy", "pydantic", "pyarrow", "torch")
    )
    difference = provenance.seen_unseen_parameter_difference
    interval = provenance.train_scaler_fit_interval
    return all(
        (
            provenance.config_hash == dataset.config_hash,
            provenance.root_seed == config.root_seed,
            provenance.root_entropy == config.root_seed,
            actual_streams == streams,
            splits == expected_splits,
            interval.start == 0 and interval.stop == bounds[1],
            difference.parameter_name == "base_log_scale",
            difference.base_log_scale_shift == config.generation.unseen_parameter_shift,
            difference.applies_from == bounds[3],
            all(getattr(summary, name) == value for name, value in summary_expected.items()),
            commit_valid,
            utc_valid,
            software_valid,
            provenance.research_status == "ENGINEERING_ARTIFACT",
        )
    )


def validate_identity(dataset: SyntheticDataset, issues: list[ValidationIssue]) -> None:
    """Recompute config/data hashes and deterministic manifest/provenance fields."""

    config_hash = _digest(_canonical(dataset.config.model_dump(mode="json")))
    if dataset.config_hash != config_hash or not _HASH.fullmatch(dataset.config_hash):
        _add(issues, "hash.config", "config", "config_hash differs or has invalid format")
    split_hashes = tuple(split.split_hash for split in dataset.physical_splits)
    try:
        from ._validation_persistence import canonical_split_hash

        expected_split_hashes = tuple(
            canonical_split_hash(split.batch, split.name) for split in dataset.physical_splits
        )
        split_hashes_valid = (
            all(isinstance(value, str) and bool(_HASH.fullmatch(value)) for value in split_hashes)
            and len(set(split_hashes)) == len(split_hashes)
            and split_hashes == expected_split_hashes
        )
    except (IndexError, TypeError, ValueError, pa.ArrowException):
        split_hashes_valid = False
    if not split_hashes_valid:
        _add(
            issues,
            "hash.split",
            "splits",
            "split hashes differ from independent canonical Arrow serialization",
        )
    data_hash = _dataset_hash(dataset, config_hash)
    if dataset.dataset_hash != data_hash or not _HASH.fullmatch(dataset.dataset_hash):
        _add(issues, "hash.dataset", "dataset", "dataset_hash differs or has invalid format")
    try:
        manifest_matches = dataset.data_manifest == _expected_manifest(dataset)
    except (IndexError, TypeError, ValueError):
        manifest_matches = False
    if not manifest_matches:
        _add(issues, "manifest.identity", "manifest", "canonical DataManifest differs")
    if not _provenance_matches(dataset):
        _add(
            issues,
            "provenance.identity",
            "provenance",
            "deterministic synthetic provenance differs",
        )


def _dynamics(dataset: SyntheticDataset, index: int) -> RegimeDynamics:
    truth = dataset.truth
    regime = int(truth["regime_sequence"][index])
    variant = int(truth["parameter_variant"][index])
    scales = truth["seen_base_log_scales"] if variant == 0 else truth["unseen_base_log_scales"]
    return RegimeDynamics(
        regime_label=regime,
        linear_matrix=truth["linear_matrices"][regime],
        nonlinear_matrix=truth["nonlinear_matrices"][regime],
        exogenous_matrix=truth["exogenous_matrices"][regime],
        nonlinear_strength=float(truth["nonlinear_strengths"][regime]),
        base_log_scale=float(scales[regime]),
        scale_loading=float(truth["scale_loadings"][regime]),
        nonlinear_delay=int(truth["nonlinear_delays"][regime]),
        trend_delay=int(truth["resolved_true_delay"][regime]),
        raw_spectral_radius=float(truth["raw_spectral_radii"][regime]),
        spectral_scale_factor=float(truth["spectral_scale_factors"][regime]),
        final_spectral_radius=float(truth["final_spectral_radii"][regime]),
        stability_target=float(truth["stability_target"]),
        true_graph=truth["true_graph"][regime],
    )


def _same_noise(left: oracle.FutureNoiseBank, right: oracle.FutureNoiseBank) -> bool:
    names = (
        "regime_path",
        "trend_innovations",
        "scale_innovations",
        "exogenous_inputs",
        "observation_innovations",
        "shocks",
    )
    return all(np.array_equal(getattr(left, name), getattr(right, name)) for name in names)


def validate_oracle_invariants(
    dataset: SyntheticDataset,
    issues: list[ValidationIssue],
) -> None:
    """Run small dataset-derived paired replays for oracle and concept invariants."""

    config, truth = dataset.config, dataset.truth
    origin = int(dataset.physical_splits[0].prediction_origins[0])
    horizon = config.H
    state_lag = int(np.max(truth["nonlinear_delays"])) + 1
    trend_lag = int(np.max(truth["resolved_true_delay"]))
    bank = oracle.FutureNoiseBank(
        regime_uniforms=None,
        regime_path=truth["regime_sequence"][origin : origin + horizon],
        trend_innovations=truth["trend_noise"][origin : origin + horizon],
        scale_innovations=truth["scale_noise"][origin : origin + horizon],
        exogenous_inputs=truth["exogenous"][origin : origin + horizon],
        observation_innovations=truth["observation_noise"][origin : origin + horizon],
        shocks=truth["shock_sequence"][origin : origin + horizon],
    )
    common = {
        "initial_history": truth["x_complete"][origin - state_lag + 1 : origin + 1],
        "trend_history": truth["trend"][origin - trend_lag : origin],
        "current_trend": float(truth["trend"][origin]),
        "current_scale": float(truth["scale"][origin]),
        "trend_ar_coefficients": truth["trend_ar_coefficients"],
        "scale_ar_coefficients": truth["scale_ar_coefficients"],
        "dynamics_schedule": tuple(
            _dynamics(dataset, index) for index in range(origin, origin + horizon)
        ),
        "noise_bank": bank,
        "trend_loading": truth["trend_loading"],
        "observation_scale_floor": float(truth["observation_scale_floor"]),
        "causal_delay": int(truth["trend_delay_schedule"][origin]),
    }
    try:
        none = oracle.replay_paired_counterfactual(intervention=None, **common)
        trend_base = oracle.replay_paired_counterfactual(
            intervention=oracle.CounterfactualIntervention(
                "trend",
                common["current_trend"],
            ),
            **common,
        )
        scale_base = oracle.replay_paired_counterfactual(
            intervention=oracle.CounterfactualIntervention(
                "scale",
                common["current_scale"],
            ),
            **common,
        )
        trend_shift = oracle.replay_paired_counterfactual(
            intervention=oracle.CounterfactualIntervention(
                "trend",
                float(common["current_trend"]) + 1.0,
            ),
            **common,
        )
        scale_shift = oracle.replay_paired_counterfactual(
            intervention=oracle.CounterfactualIntervention(
                "scale",
                float(common["current_scale"]) + 1.0,
            ),
            **common,
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        _add(issues, "oracle.execution", "oracle.dataset_replay", str(error))
        return
    expected_factual = truth["x_complete"][origin + 1 : origin + horizon + 1]
    if any(
        result.factual_path.full_values.tobytes() != expected_factual.tobytes()
        for result in (none, trend_base, scale_base, trend_shift, scale_shift)
    ):
        _add(
            issues,
            "oracle.factual",
            "oracle.factual_path",
            "dataset-derived factual replay differs",
        )
    zero = np.zeros_like(none.effect)
    if any(not np.array_equal(result.effect, zero) for result in (none, trend_base, scale_base)):
        _add(
            issues,
            "oracle.source_base",
            "oracle.effect",
            "no-intervention/source=base must be exact zero",
        )
    if np.array_equal(
        trend_shift.factual_concepts.trend,
        trend_shift.counterfactual_concepts.trend,
    ):
        _add(
            issues,
            "oracle.intervention_effect",
            "oracle.trend",
            "nontrivial trend intervention did not change the trend latent path",
        )
    if np.array_equal(
        scale_shift.factual_concepts.scale,
        scale_shift.counterfactual_concepts.scale,
    ):
        _add(
            issues,
            "oracle.intervention_effect",
            "oracle.scale",
            "nontrivial scale intervention did not change the scale latent path",
        )
    if not np.array_equal(
        trend_shift.factual_concepts.scale,
        trend_shift.counterfactual_concepts.scale,
    ):
        _add(
            issues,
            "oracle.concept_isolation",
            "oracle.scale",
            "trend intervention changed scale latent path",
        )
    if not np.array_equal(
        scale_shift.factual_concepts.trend,
        scale_shift.counterfactual_concepts.trend,
    ):
        _add(
            issues,
            "oracle.concept_isolation",
            "oracle.trend",
            "scale intervention changed trend latent path",
        )
    if any(
        not _same_noise(result.noise_bank, bank)
        for result in (none, trend_base, scale_base, trend_shift, scale_shift)
    ):
        _add(
            issues,
            "oracle.paired_noise",
            "oracle.noise_bank",
            "paired replay did not retain the dataset-derived future bank",
        )


def computed_hashes(dataset: SyntheticDataset) -> tuple[str, str]:
    """Return independently recomputed config and dataset hashes."""

    config_hash = _digest(_canonical(dataset.config.model_dump(mode="json")))
    return config_hash, _dataset_hash(dataset, config_hash)
