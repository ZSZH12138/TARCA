from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from tarca.contracts import StrictContractModel, canonical_json_hash
from tarca.stage2.seeds import derive_namespaced_seed


class E02BootstrapConfig(StrictContractModel):
    method: Literal["PAIRED_STRATIFIED_TRAJECTORY_PERCENTILE"]
    replicates: Literal[5000]
    confidence: float
    seed_namespace: Literal["tarca/stage2_probabilistic_forecasting_v1/bootstrap/0"]
    seed: int
    strata: tuple[Literal["FORMAL_SEED", "REGIME"], Literal["FORMAL_SEED", "REGIME"]]
    unit: Literal["COMPLETE_TRAJECTORY"]

    @field_validator("strata", mode="before")
    @classmethod
    def _list_becomes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _bootstrap_is_exact(self) -> Self:
        expected_seed = derive_namespaced_seed(self.seed_namespace)
        if self.confidence != 0.90 or self.seed != expected_seed:
            raise ValueError("E02 bootstrap constants must match the frozen design")
        return self


class E02GateConfig(StrictContractModel):
    primary_model_id: Literal["ITRANSFORMER"]
    baseline_selection: Literal["VALIDATION_STRONGEST_OF_VAR_DLINEAR"]
    primary_horizons: tuple[int, int]
    minimum_crps_skill: float
    ci_lower_strictly_above: float
    minimum_positive_data_seeds: int
    data_seed_count: int
    minimum_positive_initializations: int
    initialization_count: int
    require_better_than_last_value: Literal[True]
    require_better_than_seasonal_naive: Literal[True]
    seen_skill_strictly_above: float
    unseen_skill_floor: float
    relative_nll_tolerance: float
    relative_mae_tolerance: float
    coverage_levels: tuple[float, ...]
    overall_coverage_error_max: float
    regime_coverage_error_max: float
    secondary_horizon_groups: tuple[tuple[int, int], tuple[int, int]]
    secondary_horizon_skill_floor: float
    require_finite: Literal[True]
    require_positive_scale: Literal[True]
    require_non_crossing_quantiles: Literal[True]
    required_completed_trajectories: int
    allowed_failed_trajectories: Literal[0]

    @field_validator(
        "primary_horizons",
        "coverage_levels",
        "secondary_horizon_groups",
        mode="before",
    )
    @classmethod
    def _lists_become_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) if isinstance(item, list) else item for item in value)
        return value

    @model_validator(mode="after")
    def _gate_is_exact(self) -> Self:
        observed = (
            self.primary_horizons,
            self.minimum_crps_skill,
            self.ci_lower_strictly_above,
            self.minimum_positive_data_seeds,
            self.data_seed_count,
            self.minimum_positive_initializations,
            self.initialization_count,
            self.seen_skill_strictly_above,
            self.unseen_skill_floor,
            self.relative_nll_tolerance,
            self.relative_mae_tolerance,
            self.coverage_levels,
            self.overall_coverage_error_max,
            self.regime_coverage_error_max,
            self.secondary_horizon_groups,
            self.secondary_horizon_skill_floor,
            self.required_completed_trajectories,
        )
        expected = (
            (1, 6),
            0.02,
            0.0,
            3,
            5,
            2,
            3,
            0.0,
            -0.05,
            0.05,
            0.05,
            (0.50, 0.80, 0.90, 0.95),
            0.05,
            0.10,
            ((7, 12), (13, 24)),
            -0.10,
            120,
        )
        if observed != expected:
            raise ValueError("E02 gate thresholds must match the frozen design")
        return self


class E02RuntimeProfile(StrictContractModel):
    profile_id: Literal["e02-v1-two-rtx4090"]
    monitor_bind_host: Literal["127.0.0.1"]
    monitor_port: int = Field(ge=1024, le=65535)
    formal_acknowledgement: Literal["I_ACKNOWLEDGE_E02_V1_FORMAL_RUN"]
    reset_margin_hours: Literal[1]


class E02Config(StrictContractModel):
    schema_version: Literal["1.0.0"]
    protocol_id: Literal["TARCA-E2E-STAGE-PROTOCOL-2.0"]
    experiment_id: Literal["e02_predictor_validity_v1"]
    expected_stage2_experiment_id: Literal["stage2_probabilistic_forecasting_v1"]
    formal_partition: Literal["TEST"]
    formal_seeds: tuple[int, ...]
    trajectories_seen_per_seed: Literal[12]
    trajectories_unseen_per_seed: Literal[12]
    bootstrap: E02BootstrapConfig
    gate: E02GateConfig
    runtime_profile: E02RuntimeProfile

    @field_validator("formal_seeds", mode="before")
    @classmethod
    def _list_becomes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _formal_design_is_exact(self) -> Self:
        if self.formal_seeds != (1729, 2718, 3141, 5772, 8111):
            raise ValueError("E02 formal seeds must match the frozen design")
        return self

    def scientific_payload(self) -> Mapping[str, Any]:
        return self.model_dump(mode="json", exclude={"runtime_profile"})

    def scientific_hash(self) -> str:
        return canonical_json_hash(self.scientific_payload())

    def runtime_hash(self) -> str:
        return canonical_json_hash(self.runtime_profile)


def load_e02_config(path: Path) -> E02Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return E02Config.model_validate(payload)
