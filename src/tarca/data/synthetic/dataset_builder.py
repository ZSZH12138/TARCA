"""Deterministic synthetic dataset composition and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import make_dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

import numpy as np
import pyarrow as pa
import pydantic
import torch
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tarca import __version__
from tarca.contracts import (
    CONTRACT_SCHEMA_VERSION,
    DataManifest,
    DataSplitSummary,
    SplitPartition,
    WindowBatch,
    WindowContractSummary,
)

from ._path_safety import (
    StagingDirectory,
    capture_directory,
    cleanup_staging_directory,
    create_staging_directory,
    publish_staging_directory,
    verify_staging_directory,
)
from .latent_concepts import generate_latent_concepts
from .missingness import generate_missing_mask
from .nonlinear_var import RegimeDynamics, generate_regime_dynamics, rollout_nonlinear_var
from .regimes import (
    RANDOM_STREAM_NAMES,
    build_regime_parameter_schedule,
    make_unseen_parameter_shift,
    sample_regime_sequence,
    spawn_random_streams,
)

Array = NDArray[np.generic]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_SPLITS = ("train", "validation", "test_seen_regime", "test_unseen_regime")
_SNR_SCALE = {"high": -2.0, "medium": -1.25, "low": -0.55}
_OVERLAP_SUPPORT = {"low": 0.25, "medium": 0.50, "high": 1.0}
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)
# fmt: off
_TENSOR_FIELDS = (
    ("x", "float"), ("y", "float"), ("observed_covariates", "float"),
    ("known_future_covariates", "float"), ("x_observed_mask", "bool"),
    ("y_observed_mask", "bool"), ("observed_covariates_mask", "bool"),
    ("known_future_covariates_mask", "bool"))
_NAME_FIELDS = ("input_feature_names", "target_names", "observed_covariate_names",
    "known_future_covariate_names")
_TIME_FIELDS = ("feature_start", "feature_end", "prediction_start", "label_end")

class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

class GenerationConfig(_StrictModel):
    exogenous_dimensions: int = Field(ge=1)
    transition_persistence: FiniteFloat = Field(gt=0.0, lt=1.0)
    trend_ar: FiniteFloat = Field(gt=-1.0, lt=1.0)
    scale_ar: FiniteFloat = Field(gt=-1.0, lt=1.0)
    nonlinear_strength: FiniteFloat = Field(ge=0.0)
    nonlinear_lag: int = Field(ge=0)
    spectral_radius_target: FiniteFloat = Field(gt=0.0, le=0.85)
    shock_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    shock_scale: FiniteFloat = Field(ge=0.0)
    scale_floor: FiniteFloat = Field(gt=0.0)
    unseen_parameter_shift: FiniteFloat

class NormalizationConfig(_StrictModel):
    epsilon: FiniteFloat = Field(gt=0.0)
    fill_value: FiniteFloat = Field(ge=0.0, le=0.0)

class SyntheticConfig(_StrictModel):
    """Strict resolved generator configuration."""
    name: str = Field(min_length=1)
    D: int = Field(ge=1)
    L: int = Field(ge=1)
    H: int = Field(ge=1)
    regimes: int = Field(ge=2)
    true_delay: int | tuple[int, int]
    root_seed: int = Field(ge=0)
    burn_in: int = Field(ge=1)
    total_steps: int = Field(ge=1)
    mc_samples_smoke: int = Field(ge=1, le=256)
    oracle_pairs_smoke: int = Field(ge=1, le=16)
    snr: Literal["high", "medium", "low"]
    concept_overlap: Literal["low", "medium", "high"]
    missing_rate: FiniteFloat = Field(ge=0.0, lt=1.0)
    missingness_kind: Literal["none", "mcar", "block"]
    unseen_shift_kind: Literal["parameter_shift"]
    generation: GenerationConfig
    normalization: NormalizationConfig
    @field_validator("true_delay", mode="before")
    @classmethod
    def _delay(cls, value: object) -> int | tuple[int, int]:
        if type(value) is int:
            return value
        if type(value) in (list, tuple) and len(value) == 2:
            if all(type(item) is int for item in value):
                return (value[0], value[1])
        raise ValueError("true_delay: expected an int or closed two-int YAML interval")
    @model_validator(mode="after")
    def _geometry(self) -> Self:
        delays = (self.true_delay,) if isinstance(self.true_delay, int) else self.true_delay
        if min(delays) < 0 or delays[0] > delays[-1]:
            raise ValueError("true_delay: expected a non-negative sorted closed interval")
        if max(delays) >= min(self.L, self.H):
            raise ValueError("true_delay: maximum must be smaller than both L and H")
        if self.L < self.generation.nonlinear_lag + 1:
            raise ValueError("L: must be at least nonlinear_lag + 1")
        if self.burn_in < max(self.generation.nonlinear_lag + 1, max(delays)):
            raise ValueError("burn_in: too short for replay histories")
        for name, base in (
            ("trend_ar", self.generation.trend_ar),
            ("scale_ar", self.generation.scale_ar),
        ):
            if max(abs(base - 0.02), abs(base + 0.02)) >= 1.0:
                raise ValueError(f"{name}: regime offsets must remain strictly stable")
        bounds = _split_boundaries(self.total_steps)
        if any(b - a < self.L + self.H for a, b in pairwise(bounds)):
            raise ValueError("physical split: every block must contain at least L + H steps")
        if self.missingness_kind == "none" and self.missing_rate != 0.0:
            raise ValueError("missing_rate: must be 0 for missingness_kind 'none'")
        if self.missingness_kind != "none" and not 0.0 < self.missing_rate < 1.0:
            raise ValueError("missing_rate: must lie in (0, 1) when missingness is enabled")
        return self

_fields = lambda names, kind: dict.fromkeys(names.split(), kind)  # noqa: E731
_strict_record = lambda name, fields: type(  # noqa: E731
    name, (_StrictModel,), {"__annotations__": dict(fields)})

NormalizationRecord = _strict_record(
    "NormalizationRecord", _fields("epsilon fill_value", float)
    | _fields("mean raw_std scale", tuple[float, ...]) | {"zero_variance_indices": tuple[int, ...]}
    | _fields("fit_start fit_stop", int))
RandomStreamProvenance = _strict_record(
    "RandomStreamProvenance", _fields("name reproducible_id", str) | {"spawn_key": tuple[int, ...]})
PhysicalSplitProvenance = _strict_record(
    "PhysicalSplitProvenance", _fields("name split_hash", str) | _fields("start stop count", int))
IntervalProvenance = _strict_record("IntervalProvenance", _fields("start stop", int))
ParameterSummary = _strict_record(
    "ParameterSummary", _fields("dimension regimes exogenous_dimensions trend_support_size", int)
    | _fields("stability_target shock_rate shock_scale scale_floor", float)
    | _fields(
        "initial_probabilities trend_ar_coefficients scale_ar_coefficients "
        "seen_base_log_scales unseen_base_log_scales scale_loadings "
        "raw_spectral_radii scale_factors final_spectral_radii",
        tuple[float, ...])
    | _fields("resolved_true_delay nonlinear_delays", tuple[int, ...])
    | {"transition_matrix": tuple[tuple[float, ...], ...]}
    | {"observation_noise_distribution": Literal["standard_normal"]})
SeenUnseenParameterDifference = _strict_record(
    "SeenUnseenParameterDifference", {"parameter_name": Literal["base_log_scale"],
    "base_log_scale_shift": float, "applies_from": int})
SoftwareVersions = _strict_record(
    "SoftwareVersions", _fields("python tarca numpy pydantic pyarrow torch", str))
SyntheticProvenance = _strict_record(
    "SyntheticProvenance", _fields("config_hash git_commit git_commit_status", str)
    | _fields("root_seed root_entropy", int)
    | {"random_streams": tuple[RandomStreamProvenance, ...],
       "parameter_summary": ParameterSummary,
       "physical_splits": tuple[PhysicalSplitProvenance, ...],
       "train_scaler_fit_interval": IntervalProvenance,
       "seen_unseen_parameter_difference": SeenUnseenParameterDifference,
       "software_versions": SoftwareVersions, "generated_at": datetime,
       "research_status": Literal["ENGINEERING_ARTIFACT"]})
_CompositeManifest = _strict_record(
    "_CompositeManifest", {"data_manifest": DataManifest,
        "synthetic_provenance": SyntheticProvenance})
PhysicalSplit = make_dataclass("PhysicalSplit", [("name", str), ("start", int), ("stop", int),
    ("prediction_origins", NDArray[np.int64]), ("batch", WindowBatch), ("split_hash", str)],
    namespace={"__post_init__": lambda self: object.__setattr__(
        self, "prediction_origins", _read_only(self.prediction_origins))},
    frozen=True, slots=True)
SyntheticDataset = make_dataclass("SyntheticDataset", [("config", SyntheticConfig),
    ("config_hash", str), ("dataset_hash", str), ("truth", Mapping[str, Array]),
    ("normalization", NormalizationRecord), ("physical_splits", tuple[PhysicalSplit, ...]),
    ("data_manifest", DataManifest), ("synthetic_provenance", SyntheticProvenance)],
    namespace={"__post_init__": lambda self: object.__setattr__(
        self, "truth", _freeze_truth(self.truth))}, frozen=True, slots=True)
PersistedSyntheticDataset = make_dataclass("PersistedSyntheticDataset", [("output_root", Path),
    ("dataset_hash", str), ("files", Mapping[str, Path]), ("checksums", Mapping[str, str])],
    namespace={"__post_init__": lambda self: (
        object.__setattr__(self, "files", MappingProxyType(dict(self.files))),
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums))))},
    frozen=True, slots=True)
def load_synthetic_config(path: str | os.PathLike[str]) -> SyntheticConfig:
    """Load one strict synthetic YAML configuration."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config: expected a YAML mapping")
    return SyntheticConfig.model_validate(payload, strict=True)

def build_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Build one deterministic in-memory synthetic dataset."""
    if not isinstance(config, SyntheticConfig):
        raise TypeError("config: expected SyntheticConfig")
    config_hash = _digest(_canonical(config.model_dump(mode="json")))
    truth = _generate_truth(config)
    bounds = _split_boundaries(config.total_steps)
    normalization = _fit_normalization(truth["x_complete"],
        epsilon=config.normalization.epsilon, fit_start=0, fit_stop=bounds[1],
        fill_value=config.normalization.fill_value)
    splits = tuple(
        _build_split(config, config_hash, truth, normalization, name, start, stop)
        for name, start, stop in zip(_SPLITS, bounds[:-1], bounds[1:], strict=True))
    dataset_hash = _dataset_hash(config_hash, truth, normalization, splits)
    generated_at = datetime.now(UTC)
    manifest = _make_manifest(config, dataset_hash, splits, generated_at)
    provenance = _provenance(config, config_hash, truth, splits, bounds, generated_at)
    return SyntheticDataset(config, config_hash, dataset_hash, truth, normalization,
        splits, manifest, provenance)

def _make_manifest(config: SyntheticConfig, dataset_hash: str,
    splits: tuple[PhysicalSplit, ...], created_at: datetime) -> DataManifest:
    features = tuple(f"x{i}" for i in range(config.D))
    covariates = tuple(f"u{i}" for i in range(config.generation.exogenous_dimensions))
    test_hash = _digest(_canonical(
        {splits[2].name: splits[2].split_hash, splits[3].name: splits[3].split_hash}))
    return DataManifest(dataset_name=config.name, dataset_version="stage1b-v1",
        dataset_hash=dataset_hash, splits=(
            DataSplitSummary(partition=SplitPartition.TRAIN, split_hash=splits[0].split_hash,
                count=len(splits[0].batch.window_id)),
            DataSplitSummary(partition=SplitPartition.VALIDATION,
                split_hash=splits[1].split_hash, count=len(splits[1].batch.window_id)),
            DataSplitSummary(partition=SplitPartition.TEST, split_hash=test_hash,
                count=sum(len(split.batch.window_id) for split in splits[2:]))),
        window_contract=WindowContractSummary(history_length=config.L, horizon=config.H,
            input_feature_names=features, target_names=features,
            observed_covariate_names=covariates, known_future_covariate_names=covariates,
            timezone="UTC", missingness_protocol=config.missingness_kind),
        source_description="TARCA Stage 1B deterministic synthetic SCM", created_at=created_at)

def _provenance(config: SyntheticConfig, config_hash: str, truth: Mapping[str, Array],
    splits: tuple[PhysicalSplit, ...], bounds: tuple[int, ...],
    generated_at: datetime) -> SyntheticProvenance:
    commit, commit_status = _git_commit()
    spawned = spawn_random_streams(config.root_seed)
    streams = tuple(RandomStreamProvenance(name=name, spawn_key=spawned[name].spawn_key,
        reproducible_id=f"seed:{config.root_seed}:spawn:"
        f"{'.'.join(map(str, spawned[name].spawn_key))}") for name in RANDOM_STREAM_NAMES)
    same = ("initial_probabilities trend_ar_coefficients scale_ar_coefficients "
        "seen_base_log_scales unseen_base_log_scales scale_loadings raw_spectral_radii "
        "final_spectral_radii").split()
    vectors = {key: tuple(map(float, truth[key])) for key in same}
    vectors["scale_factors"] = tuple(map(float, truth["spectral_scale_factors"]))
    summary = ParameterSummary(
        dimension=config.D, regimes=config.regimes,
        exogenous_dimensions=config.generation.exogenous_dimensions,
        resolved_true_delay=tuple(map(int, truth["resolved_true_delay"])),
        stability_target=config.generation.spectral_radius_target,
        transition_matrix=tuple(tuple(float(v) for v in row) for row in truth["transition_matrix"]),
        nonlinear_delays=tuple(map(int, truth["nonlinear_delays"])), **vectors,
        trend_support_size=int(np.count_nonzero(truth["trend_loading"])),
        shock_rate=config.generation.shock_rate, shock_scale=config.generation.shock_scale,
        scale_floor=config.generation.scale_floor,
        observation_noise_distribution="standard_normal")
    split_records = tuple(
        PhysicalSplitProvenance(name=item.name, start=item.start, stop=item.stop,
            count=len(item.batch.window_id), split_hash=item.split_hash)
        for item in splits)
    return SyntheticProvenance(
        config_hash=config_hash, root_seed=config.root_seed, root_entropy=config.root_seed,
        random_streams=streams, parameter_summary=summary, physical_splits=split_records,
        train_scaler_fit_interval=IntervalProvenance(start=0, stop=bounds[1]),
        seen_unseen_parameter_difference=SeenUnseenParameterDifference(
            parameter_name="base_log_scale",
            base_log_scale_shift=config.generation.unseen_parameter_shift,
            applies_from=bounds[3]
        ),
        software_versions=SoftwareVersions(
            python=platform.python_version(), tarca=__version__, numpy=np.__version__,
            pydantic=pydantic.__version__, pyarrow=pa.__version__, torch=torch.__version__
        ),
        git_commit=commit, git_commit_status=commit_status, generated_at=generated_at,
        research_status="ENGINEERING_ARTIFACT")
def _generate_truth(config: SyntheticConfig) -> Mapping[str, Array]:
    streams = spawn_random_streams(config.root_seed)
    dynamics, unseen_dynamics, parameters = _generate_parameters(
        config, streams["parameter_generation"].generator)
    burn, steps, total = config.burn_in, config.total_steps, config.burn_in + config.total_steps
    transition = np.full((config.regimes, config.regimes),
        (1.0 - config.generation.transition_persistence) / (config.regimes - 1),
        dtype=np.float64)
    np.fill_diagonal(transition, config.generation.transition_persistence)
    initial = np.full(config.regimes, 1.0 / config.regimes, dtype=np.float64)
    regime_uniforms = streams["regime_transitions"].generator.random(total)
    trend_noise = streams["trend_innovations"].generator.normal(0.0, 0.08, total)
    scale_noise = streams["scale_innovations"].generator.normal(0.0, 0.08, total)
    exogenous = streams["exogenous_variables"].generator.normal(
        0.0, 1.0, (total, config.generation.exogenous_dimensions))
    observation_noise = streams["observation_innovations"].generator.normal(
        0.0, 1.0, (total, config.D))
    shock_rng = streams["sparse_shocks"].generator
    shocks = np.asarray((shock_rng.random((total, config.D)) < config.generation.shock_rate)
        * shock_rng.normal(0.0, config.generation.shock_scale, (total, config.D)),
        dtype=np.float64)
    missing_rng = streams["missingness"].generator
    uniforms = starts = lengths = None
    if config.missingness_kind == "mcar":
        uniforms = missing_rng.random((steps, config.D))
    elif config.missingness_kind == "block":
        length = max(1, round(config.missing_rate * steps))
        starts = np.array([missing_rng.integers(0, steps - length + 1)], dtype=np.int64)
        lengths = np.array([length], dtype=np.int64)
    regimes = sample_regime_sequence(transition, initial, regime_uniforms)
    trend_ar = np.asarray(config.generation.trend_ar
        + np.linspace(-0.02, 0.02, config.regimes), dtype=np.float64)
    scale_ar = np.asarray(config.generation.scale_ar
        + np.linspace(0.02, -0.02, config.regimes), dtype=np.float64)
    concepts = generate_latent_concepts(regimes, trend_ar, scale_ar,
        trend_noise, scale_noise, 0.0, 0.0)
    unseen_start = burn + _split_boundaries(steps)[3]
    variant = np.zeros(total, dtype=np.int64)
    variant[unseen_start:] = 1
    seen_parameters = {
        r: {"base_log_scale": float(parameters["seen_base_log_scales"][r])}
        for r in range(config.regimes)
    }
    unseen_parameters = make_unseen_parameter_shift(seen_parameters,
        {"base_log_scale": config.generation.unseen_parameter_shift})
    seen_schedule = build_regime_parameter_schedule(regimes, seen_parameters)
    shifted_schedule = build_regime_parameter_schedule(regimes, unseen_parameters)
    schedule = tuple(
        (dynamics if variant[i] == 0 else unseen_dynamics)[int(regime)]
        for i, regime in enumerate(regimes))
    base_schedule = np.asarray([
        (seen_schedule if variant[i] == 0 else shifted_schedule)[i]["base_log_scale"]
        for i in range(total)], dtype=np.float64)
    state_lag = max(item.nonlinear_delay for item in dynamics) + 1
    trend_lag = max(item.trend_delay for item in dynamics)
    trajectory = rollout_nonlinear_var(
        initial_history=np.zeros((state_lag, config.D), dtype=np.float64),
        trend_history=np.zeros(trend_lag, dtype=np.float64), trend_path=concepts.trend[:-1],
        scale_path=concepts.scale[:-1], dynamics_schedule=schedule, regime_labels=regimes,
        exogenous_inputs=exogenous, observation_innovations=observation_noise, shocks=shocks,
        trend_loading=parameters["trend_loading"],
        observation_scale_floor=config.generation.scale_floor, burn_in=burn)
    mask = generate_missing_mask(config.missingness_kind, (steps, config.D),
        uniforms, starts, lengths, config.missing_rate)
    post, replay = slice(burn, burn + steps), schedule[burn : burn + steps]
    post_sources = {"regime_sequence": regimes, "exogenous": exogenous,
        "observation_noise": observation_noise, "trend_noise": trend_noise,
        "scale_noise": scale_noise, "shock_sequence": shocks, "parameter_variant": variant,
        "base_log_scale_schedule": base_schedule}
    truth = {name: value[post] for name, value in post_sources.items()}
    truth.update({
        "x_complete": trajectory.full_values[burn - 1 : burn - 1 + steps],
        "trend": concepts.trend[burn:], "scale": concepts.scale[burn:],
        "missing_mask": mask,
        "trend_delay_schedule": _schedule_array(replay, "trend_delay", np.int64),
        "scale_loading_schedule": _schedule_array(replay, "scale_loading"),
        "transition_matrix": transition, "initial_probabilities": initial,
        "regime_uniforms": regime_uniforms, "trend_ar_coefficients": trend_ar,
        "scale_ar_coefficients": scale_ar,
        "replay_initial_history": trajectory.full_values[burn - state_lag : burn],
        "replay_trend_history": concepts.trend[burn - trend_lag : burn],
        "observation_scale_floor": np.asarray(config.generation.scale_floor),
        "stability_target": np.asarray(config.generation.spectral_radius_target)
    })
    schedule_fields = ("linear_matrix nonlinear_matrix exogenous_matrix nonlinear_strength "
        "nonlinear_delay raw_spectral_radius spectral_scale_factor "
        "final_spectral_radius true_graph").split()
    truth.update({f"{field}_schedule": _schedule_array(replay, field,
        np.int64 if field == "nonlinear_delay" else np.float64) for field in schedule_fields})
    truth.update(parameters)
    return MappingProxyType(truth)

def _generate_parameters(config: SyntheticConfig, rng: np.random.Generator
    ) -> tuple[tuple[RegimeDynamics, ...], tuple[RegimeDynamics, ...], dict[str, Array]]:
    r, d, u = config.regimes, config.D, config.generation.exogenous_dimensions
    shape = (r, d, d)
    linear = rng.normal(0.0, 0.45 / np.sqrt(d), shape)
    linear *= rng.random(shape) < 0.45
    nonlinear = rng.normal(0.0, 0.30 / np.sqrt(d), shape)
    exogenous = rng.normal(0.0, 0.20, (r, d, u))
    strengths = np.full(r, config.generation.nonlinear_strength, dtype=np.float64)
    base_scales = np.asarray(_SNR_SCALE[config.snr] + rng.normal(0.0, 0.08, r))
    scale_loadings = np.full(r, 0.30, dtype=np.float64)
    nonlinear_delays = np.full(r, config.generation.nonlinear_lag, dtype=np.int64)
    if isinstance(config.true_delay, int):
        trend_delays = np.full(r, config.true_delay, dtype=np.int64)
    else:
        trend_delays = rng.integers(config.true_delay[0], config.true_delay[1] + 1,
            size=r, dtype=np.int64)
    dynamics = generate_regime_dynamics(
        linear_candidates=np.asarray(linear, dtype=np.float64),
        nonlinear_matrices=np.asarray(nonlinear, dtype=np.float64),
        exogenous_matrices=np.asarray(exogenous, dtype=np.float64),
        nonlinear_strengths=strengths, base_log_scales=base_scales,
        scale_loadings=scale_loadings, nonlinear_delays=nonlinear_delays,
        trend_delays=trend_delays, target=config.generation.spectral_radius_target)
    seen = {i: {"base_log_scale": item.base_log_scale} for i, item in enumerate(dynamics)}
    shifted = make_unseen_parameter_shift(seen,
        {"base_log_scale": config.generation.unseen_parameter_shift})
    unseen = tuple(
        replace(item, base_log_scale=float(shifted[i]["base_log_scale"]))
        for i, item in enumerate(dynamics))
    support = int(np.ceil(d * _OVERLAP_SUPPORT[config.concept_overlap]))
    trend_loading = np.zeros(d, dtype=np.float64)
    selected = rng.permutation(d)[:support]
    trend_loading[selected] = rng.choice(np.array([-0.35, 0.35]), support)
    arrays: dict[str, Array] = {
        "resolved_true_delay": trend_delays,
        "linear_matrices": np.stack([item.linear_matrix for item in dynamics]),
        "nonlinear_matrices": np.stack([item.nonlinear_matrix for item in dynamics]),
        "exogenous_matrices": np.stack([item.exogenous_matrix for item in dynamics]),
        "seen_base_log_scales": base_scales,
        "unseen_base_log_scales": np.asarray(
            [shifted[i]["base_log_scale"] for i in range(r)], dtype=np.float64),
        "true_graph": np.stack([item.true_graph for item in dynamics]),
        "trend_loading": trend_loading,
        "raw_spectral_radii": np.asarray([item.raw_spectral_radius for item in dynamics]),
        "spectral_scale_factors": np.asarray([item.spectral_scale_factor for item in dynamics]),
        "final_spectral_radii": np.asarray([item.final_spectral_radius for item in dynamics]),
        "nonlinear_strengths": strengths, "scale_loadings": scale_loadings,
        "nonlinear_delays": nonlinear_delays
    }
    return dynamics, unseen, arrays
def _schedule_array(
    schedule: tuple[RegimeDynamics, ...], field: str, dtype: object = np.float64
) -> Array:
    values = [getattr(item, field) for item in schedule]
    return (np.stack(values) if isinstance(values[0], np.ndarray)
        else np.asarray(values, dtype=dtype))

def _fit_normalization(values: NDArray[np.float64], *, epsilon: float,
    fit_start: int, fit_stop: int, fill_value: float = 0.0) -> NormalizationRecord:
    fitted = values[fit_start:fit_stop]
    mean, raw_std = np.mean(fitted, axis=0), np.std(fitted, axis=0)
    scale = np.where(raw_std > epsilon, raw_std, 1.0)
    return NormalizationRecord(
        epsilon=epsilon, fill_value=fill_value, mean=tuple(float(v) for v in mean),
        raw_std=tuple(float(v) for v in raw_std), scale=tuple(float(v) for v in scale),
        zero_variance_indices=tuple(int(v) for v in np.flatnonzero(raw_std <= epsilon)),
        fit_start=fit_start, fit_stop=fit_stop)
def _build_split(config: SyntheticConfig, config_hash: str, truth: Mapping[str, Array],
    normalization: NormalizationRecord, name: str, start: int, stop: int) -> PhysicalSplit:
    origins = np.arange(start + config.L - 1, stop - config.H, dtype=np.int64)
    history = origins[:, None] + np.arange(-config.L + 1, 1)
    future = origins[:, None] + np.arange(1, config.H + 1)
    future_u = origins[:, None] + np.arange(config.H)
    values = np.asarray(truth["x_complete"])
    observed = np.asarray(truth["missing_mask"])
    exogenous = np.asarray(truth["exogenous"])
    mean, scale = np.asarray(normalization.mean), np.asarray(normalization.scale)
    x_mask, y_mask = observed[history], observed[future]
    x = np.where(x_mask, (values[history] - mean) / scale, normalization.fill_value)
    y = np.where(y_mask, (values[future] - mean) / scale, normalization.fill_value)
    u_history, u_future = exogenous[history], exogenous[future_u]
    features = tuple(f"x{i}" for i in range(config.D))
    covariates = tuple(f"u{i}" for i in range(config.generation.exogenous_dimensions))
    origin_times = tuple(_EPOCH + timedelta(hours=int(t)) for t in origins)
    metadata = _arrow_metadata(name, config.L, config.D, config.H)
    batch = WindowBatch(
        x=torch.from_numpy(np.ascontiguousarray(x)), y=torch.from_numpy(np.ascontiguousarray(y)),
        observed_covariates=torch.from_numpy(np.ascontiguousarray(u_history)),
        known_future_covariates=torch.from_numpy(np.ascontiguousarray(u_future)),
        x_observed_mask=torch.from_numpy(np.ascontiguousarray(x_mask)),
        y_observed_mask=torch.from_numpy(np.ascontiguousarray(y_mask)),
        observed_covariates_mask=torch.ones(u_history.shape, dtype=torch.bool),
        known_future_covariates_mask=torch.ones(u_future.shape, dtype=torch.bool),
        regime=torch.from_numpy(np.ascontiguousarray(truth["regime_sequence"][origins])),
        window_id=tuple(f"{config.name}:{config_hash[7:19]}:{name}:{int(t):08d}"
            for t in origins),
        input_feature_names=features, target_names=features,
        observed_covariate_names=covariates, known_future_covariate_names=covariates,
        feature_start=tuple(t - timedelta(hours=config.L - 1) for t in origin_times),
        feature_end=origin_times, prediction_start=tuple(
            t + timedelta(hours=1) for t in origin_times),
        label_end=tuple(t + timedelta(hours=config.H) for t in origin_times),
        forecast_time=tuple(tuple(t + timedelta(hours=h) for h in range(1, config.H + 1))
            for t in origin_times),
        metadata=metadata)
    return PhysicalSplit(name, start, stop, origins, batch,
        _digest(_window_batch_to_arrow_bytes(batch, name)))

_arrow_metadata = lambda name, history, dimension, horizon: {  # noqa: E731
    "contract_schema_version": CONTRACT_SCHEMA_VERSION, "physical_split": name,
    "tensor_dtype": "float64", "x_shape": [history, dimension], "y_shape": [horizon, dimension]}

def _arrow_schema(metadata: Mapping[str, object]) -> pa.Schema:
    scalar = {"float": pa.float64(), "bool": pa.bool_()}
    fields = [pa.field("window_id", pa.string(), nullable=False)]
    fields += [pa.field(name, pa.large_list(pa.large_list(scalar[kind])),
        nullable=name != "x") for name, kind in _TENSOR_FIELDS]
    fields.append(pa.field("regime", pa.int64()))
    fields += [pa.field(name, pa.large_list(pa.string()), nullable=False) for name in _NAME_FIELDS]
    timestamp = pa.timestamp("us", tz="UTC")
    fields += [pa.field(name, timestamp, nullable=False) for name in _TIME_FIELDS]
    fields += [pa.field("forecast_time", pa.large_list(timestamp), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False)]
    encoded = {key.encode(): (_canonical(value) if isinstance(value, list)
        else str(value).encode()) for key, value in metadata.items()}
    return pa.schema(fields, metadata=encoded)

def _nested_array(tensor: torch.Tensor, scalar_type: pa.DataType) -> pa.Array:
    values = np.ascontiguousarray(tensor.detach().cpu().numpy())
    batch, rows, columns = values.shape
    flat = pa.array(values.reshape(-1), type=scalar_type)
    inner = pa.LargeListArray.from_arrays(
        pa.array(np.arange(0, batch * rows * columns + 1, columns), type=pa.int64()), flat)
    return pa.LargeListArray.from_arrays(
        pa.array(np.arange(0, batch * rows + 1, rows), type=pa.int64()), inner)

def _arrow_tensor(tensor, scalar_type, count):
    return (pa.nulls(count, type=pa.large_list(pa.large_list(scalar_type)))
        if tensor is None else _nested_array(tensor, scalar_type))

def _window_batch_to_arrow_bytes(batch: WindowBatch, physical_split: str) -> bytes:
    metadata = _arrow_metadata(physical_split, int(batch.x.shape[1]),
        int(batch.x.shape[2]), len(batch.forecast_time[0]))
    schema, count = _arrow_schema(metadata), len(batch.window_id)
    scalar = {"float": pa.float64(), "bool": pa.bool_()}
    arrays = [pa.array(batch.window_id, type=pa.string())]
    arrays += [_arrow_tensor(getattr(batch, name), scalar[kind], count)
        for name, kind in _TENSOR_FIELDS]
    arrays.append(
        pa.nulls(count, type=pa.int64())
        if batch.regime is None
        else pa.array(batch.regime.cpu().numpy(), type=pa.int64()))
    arrays += [pa.array([list(getattr(batch, name))] * count,
        type=pa.large_list(pa.string())) for name in _NAME_FIELDS]
    timestamp = pa.timestamp("us", tz="UTC")
    arrays += [pa.array(getattr(batch, name), type=timestamp) for name in _TIME_FIELDS]
    horizon = len(batch.forecast_time[0])
    forecast = pa.array((v for row in batch.forecast_time for v in row), type=timestamp)
    offsets = pa.array(np.arange(0, count * horizon + 1, horizon), type=pa.int64())
    arrays.append(pa.LargeListArray.from_arrays(offsets, forecast))
    arrays.append(pa.array([_canonical(metadata).decode()] * count, type=pa.string()))
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V5,
        compression=None, use_threads=False)
    with pa.ipc.new_file(sink, schema, options=options) as writer:
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
    return sink.getvalue().to_pybytes()

def _window_batch_from_arrow_table(table: pa.Table, *, physical_split: str) -> WindowBatch:
    if len(table) == 0:
        raise ValueError("Arrow table: expected at least one row")
    metadata = json.loads(table["metadata_json"][0].as_py())
    expected = _arrow_metadata(physical_split, metadata["x_shape"][0],
        metadata["x_shape"][1], metadata["y_shape"][0])
    if metadata != expected:
        raise ValueError("metadata_json: physical split or contract metadata mismatch")
    _same(table["metadata_json"].to_pylist(), [_canonical(expected).decode()] * len(table),
        "metadata_json rows")
    if table.schema != _arrow_schema(expected):
        raise ValueError("Arrow schema: exact private schema or metadata mismatch")
    x_shape, y_shape = tuple(metadata["x_shape"]), tuple(metadata["y_shape"])
    names = {name: tuple(table[name][0].as_py()) for name in _NAME_FIELDS}
    for name in _NAME_FIELDS:
        _same(table[name].to_pylist(), [list(names[name])] * len(table), f"{name} rows")
    observed_width = len(names["observed_covariate_names"])
    forecast = table["forecast_time"].combine_chunks()
    if forecast.null_count or forecast.values.null_count:
        raise ValueError("forecast_time: null values are forbidden")
    _same(forecast.offsets.to_numpy(zero_copy_only=False),
        np.arange(0, (len(forecast) + 1) * y_shape[0], y_shape[0]),
        "forecast_time offsets")
    starts = table["prediction_start"].combine_chunks()
    if starts.null_count or any(left >= right for left, right in pairwise(starts.to_pylist())):
        raise ValueError("prediction_start: expected strictly increasing non-null values")
    def nested(name: str, shape: tuple[int, int], dtype: np.dtype) -> torch.Tensor | None:
        column = table[name].combine_chunks()
        if column.null_count == len(column):
            return None
        if column.null_count:
            raise ValueError(f"{name}: partially-null tensor column")
        inner = column.values
        _same(column.offsets.to_numpy(zero_copy_only=False),
            np.arange(0, (len(column) + 1) * shape[0], shape[0]), f"{name} outer offsets")
        _same(inner.offsets.to_numpy(zero_copy_only=False),
            np.arange(0, (len(inner) + 1) * shape[1], shape[1]), f"{name} inner offsets")
        values = np.asarray(inner.values.to_numpy(zero_copy_only=False), dtype=dtype)
        return torch.from_numpy(np.ascontiguousarray(values.reshape(len(column), *shape)).copy())
    def times(name: str) -> tuple[datetime, ...]:
        return tuple(table[name].to_pylist())
    shapes = {"x": x_shape, "y": y_shape, "x_observed_mask": x_shape,
        "y_observed_mask": y_shape,
        "observed_covariates": (x_shape[0], observed_width),
        "known_future_covariates": (y_shape[0], observed_width),
        "observed_covariates_mask": (x_shape[0], observed_width),
        "known_future_covariates_mask": (y_shape[0], observed_width)}
    tensors = {name: nested(name, shapes[name],
        np.dtype(np.float64 if kind == "float" else np.bool_))
        for name, kind in _TENSOR_FIELDS}
    regime_column = table["regime"].combine_chunks()
    tensors["regime"] = (None if regime_column.null_count == len(regime_column)
        else torch.from_numpy(np.asarray(regime_column.to_numpy(), dtype=np.int64).copy()))
    payload = {
        "window_id": tuple(table["window_id"].to_pylist()),
        **names,
        **{name: times(name) for name in _TIME_FIELDS},
        "forecast_time": tuple(tuple(row) for row in table["forecast_time"].to_pylist()),
        "metadata": metadata,
    }
    return WindowBatch(**tensors, **payload)
def persist_synthetic_dataset(
    dataset: SyntheticDataset, output_root: str | os.PathLike[str]
) -> PersistedSyntheticDataset:
    """Persist a synthetic dataset into a new directory with atomic publication."""
    if not isinstance(dataset, SyntheticDataset):
        raise TypeError("dataset: expected SyntheticDataset")
    arrow_payloads = _validate_dataset(dataset)
    target = _validate_output_path(output_root)
    parent = target.parent
    parent_guard = capture_directory(parent, label="output_root parent")
    staging_guard = create_staging_directory(target, parent_guard)
    staging = staging_guard.path
    try:
        verify_staging_directory(staging_guard)
        composite = _CompositeManifest(data_manifest=dataset.data_manifest,
            synthetic_provenance=dataset.synthetic_provenance)
        config_yaml = yaml.safe_dump(dataset.config.model_dump(mode="json"),
            sort_keys=True, allow_unicode=True).encode()
        _write_bytes(staging / "config_resolved.yaml", config_yaml, staging_guard)
        _write_bytes(
            staging / "manifest.json",
            composite.model_dump_json(indent=2).encode(),
            staging_guard,
        )
        verify_staging_directory(staging_guard)
        np.savez(staging / "truth.npz",
            **{name: dataset.truth[name] for name in sorted(dataset.truth)})
        verify_staging_directory(staging_guard)
        _fsync_existing(staging / "truth.npz")
        for split in dataset.physical_splits:
            payload = arrow_payloads[split.name]
            _write_bytes(staging / f"windows_{split.name}.arrow", payload, staging_guard)
        normalization = json.dumps(dataset.normalization.model_dump(mode="json"),
            ensure_ascii=False, indent=2).encode()
        _write_bytes(staging / "normalization.json", normalization, staging_guard)
        verify_staging_directory(staging_guard)
        checksums = {
            path.name: _digest(path.read_bytes())
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
        }
        checksum_bytes = json.dumps(checksums, ensure_ascii=False,
            indent=2, sort_keys=True).encode()
        _write_bytes(staging / "checksums.json", checksum_bytes, staging_guard)
        target = publish_staging_directory(staging_guard, target)
    except BaseException:
        cleanup_staging_directory(staging_guard)
        raise
    return PersistedSyntheticDataset(target, dataset.dataset_hash,
        {p.name: p for p in target.iterdir()}, checksums)

def _validate_output_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TypeError("output_root: expected a text filesystem path")
    if not raw or "\x00" in raw:
        raise ValueError("output_root: expected a non-empty path without NUL")
    if raw.startswith(("\\\\?\\", "\\\\.\\")):
        raise ValueError("output_root: extended or device paths are forbidden")
    lexical = Path(raw)
    if ".." in lexical.parts:
        raise ValueError("output_root: dot-dot path components are forbidden")
    if lexical.anchor.startswith("\\\\") or any(":" in part for part in lexical.parts[1:]):
        raise ValueError("output_root: UNC and alternate data stream paths are forbidden")
    absolute = lexical if lexical.is_absolute() else Path.cwd() / lexical
    if absolute == Path(absolute.anchor):
        raise ValueError("output_root: filesystem root is forbidden")
    _reject_reparse_components(absolute)
    target = absolute.resolve(strict=False)
    if os.path.lexists(target):
        raise ValueError(f"output_root: target already exists: {target}")
    if not target.parent.exists() or not target.parent.is_dir():
        raise ValueError("output_root: parent must exist and be a directory")
    _reject_reparse_components(target.parent)
    return target

def _reject_reparse_components(path: Path) -> None:
    current = Path(path.anchor)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current):
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & marker:
                raise ValueError(
                    f"output_root: symlink, junction, or reparse component: {current}")

def _write_bytes(path: Path, payload: bytes, staging: StagingDirectory) -> None:
    verify_staging_directory(staging)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    verify_staging_directory(staging)

def _fsync_existing(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())

_split_boundaries = lambda total: (  # noqa: E731
    0, total * 60 // 100, total * 80 // 100, total * 90 // 100, total)

def _dataset_hash(config_hash: str, truth: Mapping[str, Array],
    normalization: NormalizationRecord, splits: tuple[PhysicalSplit, ...]) -> str:
    digest = hashlib.sha256()
    values = [config_hash.encode()]
    for name in sorted(truth):
        array = truth[name]
        values.extend((name.encode(), array.dtype.str.encode(),
            _canonical(array.shape), array.tobytes(order="C")))
    values.extend([_canonical(normalization.model_dump(mode="json"))]
        + [split.split_hash.encode() for split in splits])
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"

def _freeze_truth(truth: Mapping[str, Array]) -> Mapping[str, Array]:
    _validate_truth(truth)
    return MappingProxyType({name: _read_only(value) for name, value in truth.items()})

def _validate_truth(truth: Mapping[str, Array]) -> None:
    for name, value in truth.items():
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            raise TypeError(f"truth.{name}: expected a numeric or bool NumPy array")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"truth.{name}: expected finite values")

def _same(actual, expected, field) -> None:
    equal = (np.array_equal(actual, expected)
        if isinstance(actual, np.ndarray) else actual == expected)
    if not equal:
        raise ValueError(f"{field}: persisted identity mismatch")

def _validate_dataset(dataset: SyntheticDataset) -> dict[str, bytes]:
    _validate_truth(dataset.truth)
    _same(dataset.config_hash,
        _digest(_canonical(dataset.config.model_dump(mode="json"))), "config_hash")
    splits, manifest = dataset.physical_splits, dataset.data_manifest
    bounds = _split_boundaries(dataset.config.total_steps)
    _same(tuple((s.start, s.stop) for s in splits), tuple(pairwise(bounds)), "split boundaries")
    _same(tuple(split.name for split in splits), _SPLITS, "physical split names and order")
    for split in splits:
        expected = np.arange(split.start + dataset.config.L - 1, split.stop - dataset.config.H)
        _same(split.prediction_origins, expected, f"{split.name} prediction_origins")
    payloads = {split.name: _window_batch_to_arrow_bytes(split.batch, split.name)
        for split in splits}
    hashes = tuple(_digest(payloads[split.name]) for split in splits)
    _same(hashes, tuple(split.split_hash for split in splits), "split hashes")
    _same(dataset.dataset_hash,
        _dataset_hash(dataset.config_hash, dataset.truth, dataset.normalization, splits),
        "dataset_hash")
    _same(manifest, _make_manifest(dataset.config, dataset.dataset_hash, splits,
        manifest.created_at), "data_manifest")
    _same(dataset.synthetic_provenance, _provenance(dataset.config, dataset.config_hash,
        dataset.truth, splits, bounds, manifest.created_at), "synthetic_provenance")
    return payloads

def _git_commit() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[4], check=True,
            capture_output=True, text=True, timeout=2)
        return result.stdout.strip(), "resolved"
    except (OSError, subprocess.SubprocessError) as error:
        return "unavailable", f"{type(error).__name__}: git rev-parse failed"

_canonical = lambda value: json.dumps(  # noqa: E731
    value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
_digest = lambda payload: f"sha256:{hashlib.sha256(payload).hexdigest()}"  # noqa: E731

def _read_only(value: Array) -> Array:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result
