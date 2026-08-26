from __future__ import annotations

import re
from collections.abc import Mapping
from math import isfinite
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from tarca.contracts import ArtifactRef, Sha256Hash, StrictContractModel


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


class SyntheticConfig(StrictContractModel):
    name: str
    D: int = Field(gt=0)
    L: int = Field(gt=0)
    H: int = Field(gt=0)
    regimes: int = Field(gt=0)
    true_delay: int | tuple[int, ...]
    root_seed: int = Field(ge=0)
    burn_in: int = Field(ge=0)
    total_steps: int = Field(gt=0)
    generation_settings: Mapping[str, object]
    normalization_settings: Mapping[str, object]

    @field_validator("name")
    @classmethod
    def _name_is_logical(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("synthetic config name must be a lowercase logical identifier")
        return value

    @field_validator("true_delay")
    @classmethod
    def _delays_are_positive(cls, value: int | tuple[int, ...]) -> int | tuple[int, ...]:
        delays = (value,) if isinstance(value, int) else value
        if not delays or any(delay <= 0 for delay in delays) or len(delays) != len(set(delays)):
            raise ValueError("true_delay values must be positive and unique")
        return value

    @field_validator("generation_settings", "normalization_settings")
    @classmethod
    def _settings_are_nonempty(
        cls, value: Mapping[str, object]
    ) -> Mapping[str, object]:
        if (
            not value
            or any(not key.strip() for key in value)
            or not all(_is_json_value(item) for item in value.values())
        ):
            raise ValueError("generator settings must be finite JSON values with nonblank keys")
        return dict(value)

    @model_validator(mode="after")
    def _time_and_normalization_are_coherent(self) -> Self:
        if self.total_steps <= self.burn_in:
            raise ValueError("total_steps must exceed burn_in")
        if self.normalization_settings.get("fit_partition") != "TRAIN":
            raise ValueError("normalization must be fit on TRAIN only")
        return self


class SCMTruthManifest(StrictContractModel):
    schema_version: Literal["2.0.0"]
    dataset_hash: Sha256Hash
    generator_config_hash: Sha256Hash
    concept_names: tuple[str, ...]
    regime_ids: tuple[str, ...]
    true_lags: Mapping[str, tuple[int, ...]]
    true_graph_ref: ArtifactRef
    latent_concepts_ref: ArtifactRef
    regime_sequence_ref: ArtifactRef
    exogenous_noise_ref: ArtifactRef
    shock_sequence_ref: ArtifactRef | None
    oracle_protocol_hash: Sha256Hash
    sealed: Literal[True]

    @field_validator("concept_names", "regime_ids")
    @classmethod
    def _names_are_nonempty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("truth names must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("truth names must be unique")
        return value

    @field_validator("true_lags")
    @classmethod
    def _lags_are_positive_and_unique(
        cls, value: Mapping[str, tuple[int, ...]]
    ) -> Mapping[str, tuple[int, ...]]:
        for name, lags in value.items():
            if (
                not name.strip()
                or not lags
                or any(lag <= 0 for lag in lags)
                or len(lags) != len(set(lags))
            ):
                raise ValueError("true lags must have nonblank names and positive unique lags")
        return dict(value)

    @model_validator(mode="after")
    def _truth_references_have_registered_roles(self) -> Self:
        required = (
            (self.true_graph_ref, "TRUE_GRAPH"),
            (self.latent_concepts_ref, "LATENT_CONCEPTS"),
            (self.regime_sequence_ref, "REGIME_SEQUENCE"),
            (self.exogenous_noise_ref, "EXOGENOUS_NOISE"),
        )
        for reference, expected_type in required:
            if reference.artifact_type != expected_type:
                raise ValueError(f"truth reference must have artifact type {expected_type}")
        if (
            self.shock_sequence_ref is not None
            and self.shock_sequence_ref.artifact_type != "SHOCK_SEQUENCE"
        ):
            raise ValueError("shock reference must have artifact type SHOCK_SEQUENCE")
        if set(self.true_lags) != set(self.concept_names):
            raise ValueError("true lag names must exactly match concept names")
        return self
