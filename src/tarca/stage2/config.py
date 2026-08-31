from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import yaml
from pydantic import Field, field_serializer, field_validator, model_validator

from tarca.contracts import Sha256Hash, StrictContractModel, canonical_json_hash
from tarca.stage1b.config import SourceConfig
from tarca.stage2.seeds import derive_namespaced_seed, validate_seed_isolation

ModelId = Literal[
    "LAST_VALUE",
    "SEASONAL_NAIVE",
    "VAR",
    "DLINEAR",
    "PATCHTST",
    "ITRANSFORMER",
]

_DEVELOPMENT_NAMESPACES = tuple(
    f"tarca/stage2_probabilistic_forecasting_v1/dev-data/{index}" for index in range(3)
)
_INITIALIZATION_NAMESPACES = tuple(
    f"tarca/stage2_probabilistic_forecasting_v1/model-init/{index}" for index in range(3)
)
_DLINEAR_FOLD_NAMESPACES = tuple(
    f"tarca/stage2_probabilistic_forecasting_v1/dlinear-fold/{index}" for index in range(5)
)


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model parameters must contain finite JSON-compatible values")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("model parameters must use JSON-compatible string keys")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    raise ValueError("model parameters must contain only JSON-compatible values")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class Stage2UpstreamConfig(StrictContractModel):
    world_id: Literal["lorenz96_twoscale_v2"]
    stage1b_manifest_sha256: Sha256Hash
    e01_receipt_sha256: Sha256Hash

    @model_validator(mode="after")
    def _identities_are_exact(self) -> Self:
        expected_stage1b = "d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25"
        expected_e01 = "16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14"
        if (
            self.stage1b_manifest_sha256 != expected_stage1b
            or self.e01_receipt_sha256 != expected_e01
        ):
            raise ValueError("Stage 2 upstream identities must match the frozen handoff")
        return self


class Stage2DataConfig(StrictContractModel):
    development_namespaces: tuple[str, ...]
    development_seeds: tuple[int, ...]
    excluded_upstream_seeds: tuple[int, ...]
    history: Literal[64]
    horizon: Literal[24]
    trajectory_length: Literal[512]
    warmup_steps: Literal[0]
    train_trajectories_per_seed: Literal[24]
    validation_trajectories_per_seed: Literal[8]
    primary_horizons: tuple[int, int]
    secondary_horizons: tuple[tuple[int, int], tuple[int, int]]

    @field_validator(
        "development_namespaces",
        "development_seeds",
        "excluded_upstream_seeds",
        "primary_horizons",
        "secondary_horizons",
        mode="before",
    )
    @classmethod
    def _lists_become_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) if isinstance(item, list) else item for item in value)
        return value

    @model_validator(mode="after")
    def _data_design_is_exact(self) -> Self:
        if self.development_namespaces != _DEVELOPMENT_NAMESPACES:
            raise ValueError("Stage 2 development namespaces must match the frozen design")
        expected = tuple(derive_namespaced_seed(item) for item in self.development_namespaces)
        if self.development_seeds != expected:
            raise ValueError(
                "Stage 2 seed isolation failed: development seeds must match their namespaces"
            )
        if self.primary_horizons != (1, 6) or self.secondary_horizons != ((7, 12), (13, 24)):
            raise ValueError("Stage 2 horizon groups must match the frozen design")
        return self


class Stage2ModelConfig(StrictContractModel):
    model_id: ModelId
    adapter: str
    parameters: Mapping[str, Any]

    @field_validator("adapter")
    @classmethod
    def _adapter_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model adapter must not be blank")
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters_are_json_compatible(cls, value: object) -> object:
        _freeze_json(value)
        return value

    @model_validator(mode="after")
    def _parameters_are_immutable(self) -> Self:
        frozen = _freeze_json(self.parameters)
        if not isinstance(frozen, Mapping):
            raise ValueError("model parameters must be a JSON-compatible mapping")
        object.__setattr__(self, "parameters", frozen)
        return self

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, Any]) -> dict[str, Any]:
        thawed = _thaw_json(value)
        if not isinstance(thawed, dict):
            raise TypeError("model parameter serialization must produce an object")
        return thawed

    def parameter(self, name: str) -> Any:
        try:
            return self.parameters[name]
        except KeyError as error:
            raise KeyError(f"model {self.model_id} has no parameter {name}") from error


class Stage2TrainingConfig(StrictContractModel):
    initialization_namespaces: tuple[str, ...]
    initialization_seeds: tuple[int, ...]
    dlinear_fold_namespaces: tuple[str, ...]
    dlinear_fold_seeds: tuple[int, ...]
    optimizer: Literal["ADAMW"]
    betas: tuple[float, float]
    epsilon: float
    neural_weight_decay: float
    dlinear_weight_decay: float
    gradient_clip_norm: float
    scheduler: Literal["NONE"]
    deterministic_algorithms: Literal[True]
    cudnn_deterministic: Literal[True]
    cudnn_benchmark: Literal[False]
    precision_candidates: tuple[Literal["FP32", "AMP_FP16"], ...]
    amp_dtype: Literal["FLOAT16"]
    scale_floor: float
    scale_ceiling_multiplier: float
    scale_absolute_ceiling: float

    @field_validator(
        "initialization_namespaces",
        "initialization_seeds",
        "dlinear_fold_namespaces",
        "dlinear_fold_seeds",
        "betas",
        "precision_candidates",
        mode="before",
    )
    @classmethod
    def _lists_become_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _training_design_is_exact(self) -> Self:
        if self.initialization_namespaces != _INITIALIZATION_NAMESPACES:
            raise ValueError("Stage 2 initialization namespaces must match the frozen design")
        if self.dlinear_fold_namespaces != _DLINEAR_FOLD_NAMESPACES:
            raise ValueError("DLinear fold namespaces must match the frozen design")
        if self.initialization_seeds != tuple(
            derive_namespaced_seed(item) for item in self.initialization_namespaces
        ):
            raise ValueError("Stage 2 initialization seeds must match their namespaces")
        if self.dlinear_fold_seeds != tuple(
            derive_namespaced_seed(item) for item in self.dlinear_fold_namespaces
        ):
            raise ValueError("DLinear fold seeds must match their namespaces")
        observed = (
            self.betas,
            self.epsilon,
            self.neural_weight_decay,
            self.dlinear_weight_decay,
            self.gradient_clip_norm,
            self.precision_candidates,
            self.scale_floor,
            self.scale_ceiling_multiplier,
            self.scale_absolute_ceiling,
        )
        expected = (
            (0.9, 0.999),
            1e-8,
            0.01,
            0.0,
            1.0,
            ("FP32", "AMP_FP16"),
            1e-4,
            10.0,
            10.0,
        )
        if observed != expected:
            raise ValueError("Stage 2 training constants must match the frozen design")
        return self


class Stage2RuntimeProfile(StrictContractModel):
    profile_id: Literal["stage2-v1-two-rtx4090"]
    base_image: Literal["pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04"]
    expected_physical_cpu_cores: Literal[28]
    scheduler_monitor_cores: Literal[1]
    system_io_reserved_cores: Literal[3]
    maximum_work_cores: Literal[24]
    expected_ram_gib: Literal[224]
    host_memory_ceiling_gib: Literal[200]
    expected_gpu_count: Literal[2]
    expected_gpu_name_substring: Literal["RTX 4090"]
    expected_gpu_vram_gib: Literal[24]
    gpu_task_cpu_threads: Literal[4]
    gpu_task_host_memory_gib: Literal[32]
    gpu_task_vram_ceiling_gib: Literal[20]
    dataloader_workers_per_gpu_job: int = Field(ge=1, le=4)
    minimum_free_storage_gib: Literal[200]
    recommended_free_storage_gib: Literal[300]
    reset_limit_hours: Literal[24]
    reset_margin_hours: Literal[1]
    monitor_bind_host: Literal["127.0.0.1"]
    monitor_port: int = Field(ge=1024, le=65535)
    training_acknowledgement: Literal["I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN"]


class Stage2Config(StrictContractModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["TARCA-E2E-STAGE-PROTOCOL-2.0"]
    experiment_id: Literal["stage2_probabilistic_forecasting_v1"]
    upstream: Stage2UpstreamConfig
    data: Stage2DataConfig
    sources: tuple[SourceConfig, ...]
    models: tuple[Stage2ModelConfig, ...]
    training: Stage2TrainingConfig
    runtime_profile: Stage2RuntimeProfile

    @field_validator("sources", "models", mode="before")
    @classmethod
    def _lists_become_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _suite_is_complete_and_isolated(self) -> Self:
        source_ids = tuple(source.source_id for source in self.sources)
        if source_ids != ("dlinear", "itransformer", "patchtst", "scoring_rules_l96"):
            raise ValueError("Stage 2 sources must contain the frozen four-source set in order")
        model_ids = tuple(model.model_id for model in self.models)
        expected_models = (
            "LAST_VALUE",
            "SEASONAL_NAIVE",
            "VAR",
            "DLINEAR",
            "PATCHTST",
            "ITRANSFORMER",
        )
        if model_ids != expected_models:
            raise ValueError("Stage 2 models must contain the frozen six-model set in order")
        try:
            validate_seed_isolation(
                development_seeds=self.data.development_seeds,
                initialization_seeds=self.training.initialization_seeds,
                excluded_seeds=self.data.excluded_upstream_seeds,
            )
        except ValueError as error:
            raise ValueError("Stage 2 seed isolation failed") from error
        return self

    def source(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"unknown Stage 2 source: {source_id}")

    def model(self, model_id: ModelId) -> Stage2ModelConfig:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(f"unknown Stage 2 model: {model_id}")

    def scientific_payload(self) -> Mapping[str, Any]:
        return self.model_dump(mode="json", exclude={"runtime_profile"})

    def scientific_hash(self) -> str:
        return canonical_json_hash(self.scientific_payload())

    def runtime_hash(self) -> str:
        return canonical_json_hash(self.runtime_profile)


def load_stage2_config(path: Path) -> Stage2Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return Stage2Config.model_validate(payload)
