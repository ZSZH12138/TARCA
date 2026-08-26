from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validated_https_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{label} must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    return value


class WorldRole(StrEnum):
    CONTROL_LINEAR = "CONTROL_LINEAR"
    PRIMARY_MECHANISTIC = "PRIMARY_MECHANISTIC"
    ORACLE_AUXILIARY = "ORACLE_AUXILIARY"
    FORECAST_STRESS = "FORECAST_STRESS"
    EXTERNAL_REALISM = "EXTERNAL_REALISM"
    REFERENCE_ONLY = "REFERENCE_ONLY"


class WorldAdapter(StrEnum):
    TARCA_VAR = "TARCA_VAR"
    LORENZ96 = "LORENZ96"
    LORENZ96_TWO_SCALE = "LORENZ96_TWO_SCALE"
    GVAR_PREDATOR_PREY = "GVAR_PREDATOR_PREY"
    CORRECTED_CML = "CORRECTED_CML"


class NeuralAdapter(StrEnum):
    PATCHTST_REFERENCE = "PATCHTST_REFERENCE"
    ITRANSFORMER_REFERENCE = "ITRANSFORMER_REFERENCE"


class RegimeSplitRole(StrEnum):
    SEEN = "SEEN"
    UNSEEN = "UNSEEN"


class QualificationPartition(StrEnum):
    QUAL_TRAIN = "QUAL_TRAIN"
    QUAL_TUNE = "QUAL_TUNE"
    QUAL_SEEN = "QUAL_SEEN"
    QUAL_UNSEEN = "QUAL_UNSEEN"


class EvidenceFileConfig(FrozenModel):
    url: str
    sha256: str

    @field_validator("url")
    @classmethod
    def _url_is_https(cls, value: str) -> str:
        return _validated_https_url(value, "evidence URL")

    @field_validator("sha256")
    @classmethod
    def _sha_is_valid(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("evidence hash must be a lowercase SHA-256")
        return value


class SourceCodeUsage(StrEnum):
    DIRECT_OFFICIAL_CODE = "DIRECT_OFFICIAL_CODE"
    DIRECT_OFFICIAL_DATA = "DIRECT_OFFICIAL_DATA"
    DIRECT_OFFICIAL_CODE_AND_DATA = "DIRECT_OFFICIAL_CODE_AND_DATA"
    REIMPLEMENTED_EQUATIONS = "REIMPLEMENTED_EQUATIONS"


class SourceAuthorizationPolicy(StrEnum):
    LICENSED = "LICENSED"
    USER_AUTHORIZED_NO_LICENSE_BLOCK = "USER_AUTHORIZED_NO_LICENSE_BLOCK"


class SourceAssetConfig(FrozenModel):
    asset_id: str
    relative_path: str
    sha256: str
    required_for: tuple[Literal["REPRODUCTION", "ORACLE", "MODEL"], ...]

    @field_validator("asset_id")
    @classmethod
    def _asset_id_is_safe(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("asset_id must be a lowercase safe identifier")
        return value

    @field_validator("relative_path")
    @classmethod
    def _asset_path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.strip()
            or "\\" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() in {"", "."}
        ):
            raise ValueError("asset relative path must stay below the official source root")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def _asset_sha_is_valid(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("asset hash must be a lowercase SHA-256")
        return value

    @field_validator("required_for")
    @classmethod
    def _asset_uses_are_unique(
        cls,
        value: tuple[Literal["REPRODUCTION", "ORACLE", "MODEL"], ...],
    ) -> tuple[Literal["REPRODUCTION", "ORACLE", "MODEL"], ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("asset required_for roles must be nonempty and unique")
        return value


class SourceConfig(FrozenModel):
    source_id: str
    title: str
    repository_url: str
    paper_url: str
    commit: str
    license_id: str
    code_usage: SourceCodeUsage
    authorization_policy: SourceAuthorizationPolicy
    authorization_id: str
    assets: tuple[SourceAssetConfig, ...]
    evidence_files: tuple[EvidenceFileConfig, ...]

    @field_validator(
        "title",
        "repository_url",
        "paper_url",
        "license_id",
        "authorization_id",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source fields must not be blank")
        return value

    @field_validator("source_id")
    @classmethod
    def _source_id_is_safe(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("source_id must be a lowercase safe identifier")
        return value

    @field_validator("repository_url", "paper_url")
    @classmethod
    def _source_urls_are_https(cls, value: str) -> str:
        return _validated_https_url(value, "source URL")

    @field_validator("commit")
    @classmethod
    def _commit_is_sha1(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("source commit must be a lowercase Git SHA-1")
        return value

    @field_validator("assets")
    @classmethod
    def _assets_are_unique(
        cls, value: tuple[SourceAssetConfig, ...]
    ) -> tuple[SourceAssetConfig, ...]:
        if not value:
            raise ValueError("each direct source requires pinned assets")
        asset_ids = tuple(asset.asset_id for asset in value)
        paths = tuple(asset.relative_path for asset in value)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset IDs must be unique within a source")
        if len(paths) != len(set(paths)):
            raise ValueError("asset relative paths must be unique within a source")
        return value

    @field_validator("evidence_files")
    @classmethod
    def _evidence_is_present(
        cls, value: tuple[EvidenceFileConfig, ...]
    ) -> tuple[EvidenceFileConfig, ...]:
        if not value:
            raise ValueError("each source requires pinned evidence files")
        urls = tuple(item.url for item in value)
        if len(urls) != len(set(urls)):
            raise ValueError("evidence URLs must be unique within a source")
        return value

    @model_validator(mode="after")
    def _authorization_is_coherent(self) -> Self:
        if (
            self.authorization_policy is SourceAuthorizationPolicy.LICENSED
            and self.license_id.upper() in {"UNDECLARED", "REFERENCE_ONLY"}
        ):
            raise ValueError("licensed source authorization requires a declared license")
        return self


class TruthCapabilities(FrozenModel):
    shared_future_noise: bool
    graph: bool
    signed_graph: bool
    causal_lag: bool
    regime: bool
    source_pairs: bool
    negative_controls: bool


class GraphConfig(FrozenModel):
    kind: Literal["RING", "LORENZ96", "PREDATOR_PREY_BLOCK"]
    directed: bool


class RegimeConfig(FrozenModel):
    regime_id: str
    split_role: RegimeSplitRole
    changed_parameter: str
    parameters: tuple[tuple[str, float], ...]

    @field_validator("regime_id", "changed_parameter")
    @classmethod
    def _regime_labels_are_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("regime labels must not be blank")
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def _freeze_parameters(cls, value: object) -> tuple[tuple[str, float], ...]:
        return _freeze_numeric_mapping(value, "regime parameters")

    @model_validator(mode="after")
    def _changed_parameter_is_explicit(self) -> Self:
        if self.changed_parameter not in self.parameter_map():
            raise ValueError("changed_parameter must be present in regime parameters")
        return self

    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


class ConceptPairConfig(FrozenModel):
    pair_id: str
    concept: Literal["trend", "scale"]
    parameter_family: str
    factual_parameter_ref: str
    counterfactual_parameter_ref: str
    factual_value: float
    counterfactual_value: float
    shared_initial_state: Literal[True]
    shared_future_noise: Literal[True]
    evidence_asset_ids: tuple[str, ...]

    @field_validator(
        "pair_id",
        "parameter_family",
        "factual_parameter_ref",
        "counterfactual_parameter_ref",
    )
    @classmethod
    def _logical_identifiers_are_safe(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("concept pair identifiers must be lowercase and safe")
        return value

    @field_validator("factual_value", "counterfactual_value")
    @classmethod
    def _parameter_value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("concept pair parameter values must be finite")
        return value

    @field_validator("evidence_asset_ids")
    @classmethod
    def _evidence_assets_are_safe_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or any(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", item) is None for item in value)
            or len(value) != len(set(value))
        ):
            raise ValueError("concept pair evidence assets must be nonempty, safe, and unique")
        return value

    @model_validator(mode="after")
    def _counterfactual_changes_one_parameter(self) -> Self:
        if self.factual_parameter_ref == self.counterfactual_parameter_ref:
            raise ValueError("concept pair parameter references must differ")
        if self.factual_value == self.counterfactual_value:
            raise ValueError("concept pair parameter values must differ")
        if (
            self.concept == "scale"
            and min(
                self.factual_value,
                self.counterfactual_value,
            )
            < 0
        ):
            raise ValueError("scale concept pair values must be nonnegative")
        return self


def _freeze_numeric_mapping(value: object, label: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping")
    frozen: list[tuple[str, float]] = []
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{label} names must be nonblank strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{label} values must be numeric")
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} values must be finite")
        frozen.append((raw_name, numeric))
    return tuple(sorted(frozen))


class WorldConfig(FrozenModel):
    world_id: str
    family_id: str
    role: WorldRole
    source_id: str
    supporting_source_ids: tuple[str, ...] = ()
    adapter: WorldAdapter
    dimension: int = Field(ge=2)
    latent_dimension: int = Field(default=0, ge=0)
    boundary_policy: Literal["NONE", "DECLARED_ZERO_CLIP"] = "NONE"
    concepts: tuple[str, ...]
    concept_pairs: tuple[ConceptPairConfig, ...] = ()
    downstream_mappings: tuple[str, ...]
    truth_capabilities: TruthCapabilities
    graph: GraphConfig
    generator: tuple[tuple[str, float], ...]
    regimes: tuple[RegimeConfig, ...]

    @field_validator("world_id", "family_id", "source_id")
    @classmethod
    def _identifiers_are_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("world identifiers must not be blank")
        return value

    @field_validator("supporting_source_ids")
    @classmethod
    def _supporting_sources_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not source_id.strip() for source_id in value) or len(value) != len(set(value)):
            raise ValueError("supporting source IDs must be nonblank and unique")
        return value

    @field_validator("concepts", "downstream_mappings")
    @classmethod
    def _labels_are_unique_and_nonblank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("world labels must be nonempty and nonblank")
        if len(set(value)) != len(value):
            raise ValueError("world labels must be unique")
        return value

    @field_validator("concept_pairs")
    @classmethod
    def _concept_pairs_are_unique(
        cls, value: tuple[ConceptPairConfig, ...]
    ) -> tuple[ConceptPairConfig, ...]:
        pair_ids = tuple(pair.pair_id for pair in value)
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("concept pair IDs must be unique within a world")
        return value

    @field_validator("generator", mode="before")
    @classmethod
    def _generator_is_immutable(cls, value: object) -> tuple[tuple[str, float], ...]:
        return _freeze_numeric_mapping(value, "generator parameters")

    @field_validator("regimes")
    @classmethod
    def _regimes_are_unique(cls, value: tuple[RegimeConfig, ...]) -> tuple[RegimeConfig, ...]:
        ids = tuple(regime.regime_id for regime in value)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("regime IDs must be nonempty and unique within a world")
        return value

    @model_validator(mode="after")
    def _world_contract_is_coherent(self) -> Self:
        if self.adapter is WorldAdapter.LORENZ96_TWO_SCALE and self.latent_dimension <= 0:
            raise ValueError("two-scale Lorenz-96 requires latent variables")
        if (
            self.boundary_policy == "DECLARED_ZERO_CLIP"
            and self.adapter is not WorldAdapter.GVAR_PREDATOR_PREY
        ):
            raise ValueError("zero clipping is declared only by the published predator-prey world")
        if self.role is not WorldRole.PRIMARY_MECHANISTIC:
            return self
        required_parameter_families = {
            WorldAdapter.LORENZ96: {
                "trend": "forcing",
                "scale": "measurement_noise",
            },
            WorldAdapter.LORENZ96_TWO_SCALE: {
                "trend": "forcing",
                "scale": "coupling_h",
            },
            WorldAdapter.GVAR_PREDATOR_PREY: {
                "trend": "alpha",
                "scale": "dynamic_noise_scale",
            },
            WorldAdapter.CORRECTED_CML: {
                "trend": "alpha",
                "scale": "epsilon",
            },
            WorldAdapter.TARCA_VAR: {
                "trend": "coefficient_scale",
                "scale": "innovation_scale",
            },
        }.get(self.adapter)
        if required_parameter_families is None:
            raise ValueError("primary world adapter has no registered concept parameter families")
        by_concept = {pair.concept: pair for pair in self.concept_pairs}
        if set(by_concept) != {"trend", "scale"} or len(self.concept_pairs) != 2:
            raise ValueError("primary worlds require exactly one trend and scale concept pair")
        for pair in self.concept_pairs:
            if pair.parameter_family != required_parameter_families[pair.concept]:
                raise ValueError(
                    f"{pair.concept} concept pair uses an unsupported parameter family"
                )
        capabilities = self.truth_capabilities
        if not capabilities.shared_future_noise:
            raise ValueError("primary worlds require shared future noise")
        if not all(
            (
                capabilities.graph,
                capabilities.signed_graph,
                capabilities.causal_lag,
                capabilities.regime,
                capabilities.source_pairs,
                capabilities.negative_controls,
            )
        ):
            raise ValueError(
                "primary worlds require graph, signed graph, lag, regime, pairs, and controls"
            )
        if len(self.concepts) < 2:
            raise ValueError("primary worlds require at least two structural concepts")
        if {regime.split_role for regime in self.regimes} != {
            RegimeSplitRole.SEEN,
            RegimeSplitRole.UNSEEN,
        }:
            raise ValueError("primary worlds require seen and unseen regimes")
        return self

    def generator_map(self) -> dict[str, float]:
        return dict(self.generator)


class WorldSuiteConfig(FrozenModel):
    schema_version: Literal["2.0.0"]
    suite_id: str
    sources: tuple[SourceConfig, ...]
    worlds: tuple[WorldConfig, ...]

    @model_validator(mode="after")
    def _suite_is_self_contained(self) -> Self:
        if not self.suite_id.strip():
            raise ValueError("suite_id must not be blank")
        source_ids = tuple(source.source_id for source in self.sources)
        world_ids = tuple(world.world_id for world in self.worlds)
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be nonempty and unique")
        if not world_ids or len(world_ids) != len(set(world_ids)):
            raise ValueError("world IDs must be nonempty and unique")
        referenced_sources = {
            source_id
            for world in self.worlds
            for source_id in (world.source_id, *world.supporting_source_ids)
        }
        unknown = sorted(referenced_sources - set(source_ids))
        if unknown:
            raise ValueError(f"worlds reference unknown sources: {', '.join(unknown)}")
        sources_by_id = {source.source_id: source for source in self.sources}
        for world in self.worlds:
            authorized_sources = (
                sources_by_id[source_id]
                for source_id in (world.source_id, *world.supporting_source_ids)
            )
            oracle_assets = {
                asset.asset_id
                for source in authorized_sources
                for asset in source.assets
                if "ORACLE" in asset.required_for
            }
            referenced_assets = {
                asset_id for pair in world.concept_pairs for asset_id in pair.evidence_asset_ids
            }
            missing_assets = sorted(referenced_assets - oracle_assets)
            if missing_assets:
                raise ValueError(
                    f"world {world.world_id} concept pair evidence assets are not registered "
                    f"for ORACLE use: {', '.join(missing_assets)}"
                )
        primary_families = {
            world.family_id for world in self.worlds if world.role is WorldRole.PRIMARY_MECHANISTIC
        }
        if len(primary_families) < 2:
            raise ValueError("suite requires two independent primary families")
        return self

    def world(self, world_id: str) -> WorldConfig:
        for world in self.worlds:
            if world.world_id == world_id:
                return world
        raise KeyError(world_id)

    def source(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    def source_manifest_sha256(self) -> str:
        payload = self.model_dump_json(include={"sources"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class TrajectoryPartitionCounts(FrozenModel):
    qual_train: int = Field(alias="QUAL_TRAIN", gt=0)
    qual_tune: int = Field(alias="QUAL_TUNE", gt=0)
    qual_seen: int = Field(alias="QUAL_SEEN", gt=0)
    qual_unseen: int = Field(alias="QUAL_UNSEEN", gt=0)


class NeuralModelConfig(FrozenModel):
    model_id: str
    adapter: NeuralAdapter
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    revin: bool
    patch_length: int | None = Field(default=None, gt=1)
    patch_stride: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _model_is_coherent(self) -> Self:
        if not self.model_id.strip():
            raise ValueError("model_id must not be blank")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.patience >= self.max_epochs:
            raise ValueError("patience must be smaller than max_epochs")
        if not self.revin:
            raise ValueError("v2 neural adapters require window normalization")
        if self.adapter is NeuralAdapter.PATCHTST_REFERENCE:
            if self.patch_length is None or self.patch_stride is None:
                raise ValueError("PatchTST requires patch length and stride")
        elif self.patch_length is not None or self.patch_stride is not None:
            raise ValueError("patch settings are exclusive to PatchTST")
        return self


class VarSearchConfig(FrozenModel):
    lag_orders: tuple[int, ...]
    ridge: tuple[float, ...]

    @field_validator("lag_orders")
    @classmethod
    def _lags_are_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(type(item) is not int or item <= 0 for item in value):
            raise ValueError("VAR lag orders must be positive integers")
        if len(value) != len(set(value)):
            raise ValueError("VAR lag orders must be unique")
        return value

    @field_validator("ridge")
    @classmethod
    def _ridge_is_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not math.isfinite(item) or item < 0 for item in value):
            raise ValueError("VAR ridge values must be finite and nonnegative")
        return value


class QualificationGateConfig(FrozenModel):
    primary_metric: Literal["CRPS"]
    bootstrap_replicates: int = Field(ge=1000)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    guardrail_relative_tolerance: float = Field(ge=0.0, lt=1.0)
    minimum_primary_families: int = Field(ge=1)
    minimum_comparison_units: int = Field(ge=40)
    minimum_win_rate: float = Field(gt=0.5, le=1.0)
    minimum_skill_score: float = Field(ge=0.0, lt=1.0)
    require_seen_and_unseen_majority: bool


class QualificationConfig(FrozenModel):
    schema_version: Literal["2.0.0"]
    qualification_id: str
    partitions: tuple[QualificationPartition, ...]
    qualification_seeds: tuple[int, int, int]
    reserved_formal_seeds: tuple[int, ...]
    history_length: int = Field(gt=0)
    horizon: int = Field(gt=0)
    horizon_groups: tuple[tuple[int, int], ...]
    trajectory_length: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    trajectories_per_partition: TrajectoryPartitionCounts
    models: tuple[NeuralModelConfig, ...]
    var_search: VarSearchConfig
    gate: QualificationGateConfig

    @field_validator("partitions", mode="before")
    @classmethod
    def _reject_formal_partitions(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            allowed = {partition.value for partition in QualificationPartition}
            if any(item not in allowed for item in value):
                raise ValueError("configuration may use qualification-only partitions")
        return value

    @model_validator(mode="after")
    def _qualification_is_coherent(self) -> Self:
        if not self.qualification_id.strip():
            raise ValueError("qualification_id must not be blank")
        if len(self.partitions) != 4 or set(self.partitions) != set(QualificationPartition):
            raise ValueError("configuration must contain exactly the qualification-only partitions")
        if len(set(self.qualification_seeds)) != 3 or any(
            seed < 0 for seed in self.qualification_seeds
        ):
            raise ValueError("qualification seeds must be three unique nonnegative integers")
        if not self.reserved_formal_seeds or len(set(self.reserved_formal_seeds)) != len(
            self.reserved_formal_seeds
        ):
            raise ValueError("reserved formal seeds must be nonempty and unique")
        if set(self.qualification_seeds) & set(self.reserved_formal_seeds):
            raise ValueError("qualification seeds must not overlap reserved formal seeds")
        if self.trajectory_length <= self.history_length + self.horizon:
            raise ValueError("trajectory length must exceed history plus horizon")
        covered = [step for start, end in self.horizon_groups for step in range(start, end + 1)]
        if covered != list(range(1, self.horizon + 1)):
            raise ValueError("horizon groups must cover each horizon exactly once in order")
        model_ids = tuple(model.model_id for model in self.models)
        adapters = {model.adapter for model in self.models}
        if len(model_ids) != len(set(model_ids)) or adapters != set(NeuralAdapter):
            raise ValueError("qualification requires one unique PatchTST and iTransformer adapter")
        return self


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_world_suite(path: Path) -> WorldSuiteConfig:
    return WorldSuiteConfig.model_validate(_load_yaml_mapping(path))


def load_qualification_config(path: Path) -> QualificationConfig:
    return QualificationConfig.model_validate(_load_yaml_mapping(path))
