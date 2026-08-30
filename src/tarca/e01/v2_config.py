from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from tarca.contracts import Sha256Hash, StrictContractModel, canonical_json_hash
from tarca.e01.config import E01Condition, E01RuntimeProfile
from tarca.e01.v2_seeds import derive_v2_seeds

_V1_FORMAL_SEEDS = (1729, 2718, 3141, 5772, 8111)
_REQUIRED_CONDITIONS: tuple[E01Condition, ...] = (
    "CORRECT_SCM",
    "WRONG_SCM",
    "WRONG_LAG",
    "RANDOM_CONCEPT",
    "IDENTITY",
)


class E01V2SeedConfig(StrictContractModel):
    algorithm: Literal["sha256-first-31-bits-v1"]
    namespace: Literal["tarca/e01-v2/formal-test"]
    count: Literal[50]
    excluded_seeds: tuple[int, ...]

    @field_validator("excluded_seeds", mode="before")
    @classmethod
    def _list_becomes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _v1_seeds_are_excluded(self) -> Self:
        if self.excluded_seeds != _V1_FORMAL_SEEDS:
            raise ValueError("E01-v1 formal seeds must be excluded in their frozen order")
        derive_v2_seeds(self.namespace, self.count, self.excluded_seeds)
        return self


class E01V2WorldConfig(StrictContractModel):
    world_id: Literal["analytic_delayed_control_v1"]
    study: Literal["E01_A"]
    true_lag: int = Field(gt=0)
    wrong_lag: int = Field(gt=0)
    intervention_delta: float
    decay: float = Field(gt=0.0, lt=1.0)
    estimator_device: Literal["GPU"]

    @model_validator(mode="after")
    def _world_is_coherent(self) -> Self:
        if self.true_lag == self.wrong_lag:
            raise ValueError("wrong lag must differ from true lag")
        if not math.isfinite(self.intervention_delta) or self.intervention_delta == 0.0:
            raise ValueError("intervention delta must be finite and nonzero")
        return self


class E01V2GateConfig(StrictContractModel):
    gate_freeze_id: Literal["e01-gates-v2"]
    threshold_source_partition: Literal["VALIDATION"]
    formal_evaluation_partition: Literal["TEST"]
    frozen_before_formal: Literal[True]
    confidence: float
    required_seed_count: Literal[45]
    mcse_ratio_max: float
    interval_half_width_max: float
    aggregate_multiplier_bias_max: float
    control_win_fraction_min: float
    analytic_lag_tolerance_steps: Literal[1]
    endpoint_error_ratio_is_gate: Literal[False]

    @model_validator(mode="after")
    def _floating_thresholds_are_exact(self) -> Self:
        observed = (
            self.confidence,
            self.mcse_ratio_max,
            self.interval_half_width_max,
            self.aggregate_multiplier_bias_max,
            self.control_win_fraction_min,
        )
        if observed != (0.95, 0.25, 0.01, 0.005, 0.65):
            raise ValueError("E01-v2 floating gate thresholds must match the frozen design")
        return self


class E01V2CarryForwardConfig(StrictContractModel):
    report_path: str
    report_sha256: Sha256Hash
    recovery_validation_path: str
    recovery_validation_sha256: Sha256Hash
    expected_archive_sha256: Sha256Hash
    expected_stage1b_manifest_sha256: Sha256Hash
    expected_scientific_config_sha256: Sha256Hash
    required_convergence_seed_count: Literal[5]
    required_directional_seed_count: Literal[5]

    @field_validator("report_path", "recovery_validation_path")
    @classmethod
    def _path_is_safe_relative_text(cls, value: str) -> str:
        if not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("carry-forward paths must be safe repository-relative paths")
        return Path(value).as_posix()

    def scientific_payload(self) -> Mapping[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"report_path", "recovery_validation_path"},
        )


class E01V2Config(StrictContractModel):
    schema_version: Literal["2.0.0"]
    protocol_id: Literal["TARCA-E2E-STAGE-PROTOCOL-2.0"]
    experiment_id: Literal["e01_scm_truth_v2"]
    formal_partition: Literal["TEST"]
    formal_seed_config: E01V2SeedConfig
    sample_sizes: tuple[int, ...]
    horizons: tuple[int, ...]
    conditions: tuple[E01Condition, ...]
    worlds: tuple[E01V2WorldConfig, ...]
    gates: E01V2GateConfig
    carry_forward: E01V2CarryForwardConfig
    runtime_profile: E01RuntimeProfile

    @field_validator("sample_sizes", "horizons", "conditions", "worlds", mode="before")
    @classmethod
    def _lists_become_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("sample_sizes")
    @classmethod
    def _sample_sizes_are_frozen_nested_prefixes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        expected = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
        if value != expected or any(right != left * 2 for left, right in pairwise(value)):
            raise ValueError("sample sizes must be the frozen nested prefixes through 8192")
        return value

    @field_validator("horizons")
    @classmethod
    def _horizons_are_frozen(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(range(1, 13)):
            raise ValueError("E01-v2 horizons must be h1-h12")
        return value

    @field_validator("conditions")
    @classmethod
    def _conditions_are_complete(cls, value: tuple[E01Condition, ...]) -> tuple[E01Condition, ...]:
        if value != _REQUIRED_CONDITIONS:
            raise ValueError("conditions must contain every E01-v2 control in frozen order")
        return value

    @model_validator(mode="after")
    def _design_is_exact(self) -> Self:
        if len(self.formal_seeds) != 50:
            raise ValueError("E01-v2 requires exactly 50 formal TEST seeds")
        if len(self.worlds) != 1 or self.worlds[0].study != "E01_A":
            raise ValueError("E01-v2 reruns only the E01-A analytic world")
        return self

    @property
    def formal_seeds(self) -> tuple[int, ...]:
        seed_config = self.formal_seed_config
        return derive_v2_seeds(
            seed_config.namespace,
            seed_config.count,
            seed_config.excluded_seeds,
        )

    def scientific_payload(self) -> Mapping[str, Any]:
        payload = self.model_dump(mode="json", exclude={"runtime_profile", "carry_forward"})
        return {
            **payload,
            "carry_forward": self.carry_forward.scientific_payload(),
            "derived_formal_seeds": self.formal_seeds,
        }

    def scientific_hash(self) -> str:
        return canonical_json_hash(self.scientific_payload())

    def runtime_hash(self) -> str:
        return canonical_json_hash(self.runtime_profile)


def load_e01_v2_config(path: Path) -> E01V2Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return E01V2Config.model_validate(payload)
