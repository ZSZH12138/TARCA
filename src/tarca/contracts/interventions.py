from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal, Self

import torch
from pydantic import Field, field_validator, model_validator
from torch import Tensor

from .base import Sha256Hash, StrictContractModel
from .data import SplitPartition, WindowBatch, validate_window_batch
from .forecasts import ForecastDistribution


class RegimeRelation(StrEnum):
    SAME = "SAME"
    CROSS = "CROSS"
    UNKNOWN = "UNKNOWN"


class InterventionKind(StrEnum):
    FULL_SWAP = "FULL_SWAP"
    SUBSPACE_SWAP = "SUBSPACE_SWAP"


@dataclass(frozen=True, slots=True)
class InterventionSite:
    site_name: str
    layer: int | None
    tensor_rank: int
    batch_axis: int
    variable_axis: int | None
    patch_axis: int | None
    feature_axis: int
    shape_template: tuple[int | None, ...]


def validate_intervention_site(site: InterventionSite) -> InterventionSite:
    if not site.site_name.strip():
        raise ValueError("site_name must not be blank")
    if site.layer is not None and site.layer < 0:
        raise ValueError("layer must be non-negative")
    if site.tensor_rank <= 0 or len(site.shape_template) != site.tensor_rank:
        raise ValueError("shape_template must match positive tensor rank")
    if any(size is not None and size <= 0 for size in site.shape_template):
        raise ValueError("known shape dimensions must be positive")
    axes = (site.batch_axis, site.variable_axis, site.patch_axis, site.feature_axis)
    declared_axes = tuple(axis for axis in axes if axis is not None)
    if any(axis < 0 or axis >= site.tensor_rank for axis in declared_axes):
        raise ValueError("intervention axes must be within tensor rank")
    if len(set(declared_axes)) != len(declared_axes):
        raise ValueError("intervention axes must be unique")
    return site


class PairingSpec(StrictContractModel):
    spec_id: str
    partition: SplitPartition
    concept_name: str
    regime_relation: RegimeRelation
    min_concept_delta: float
    distance_metric: Literal["STRATIFIED_RANDOM", "EUCLIDEAN", "MAHALANOBIS", "LOCAL_OT"]
    matching_feature_names: tuple[str, ...]
    max_source_reuse: int = Field(gt=0)
    max_time_overlap: int = Field(ge=0)
    seed: int
    spec_hash: Sha256Hash

    @field_validator("spec_id", "concept_name")
    @classmethod
    def _identifiers_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pairing identifiers must not be blank")
        return value

    @field_validator("min_concept_delta")
    @classmethod
    def _minimum_delta_is_finite(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("min_concept_delta must be finite and non-negative")
        return value

    @field_validator("matching_feature_names")
    @classmethod
    def _matching_features_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value):
            raise ValueError("matching feature names must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("matching feature names must be unique")
        return value


class InterventionPair(StrictContractModel):
    schema_version: Literal["1.0.0"]
    pair_id: Sha256Hash
    partition: SplitPartition
    base_window_id: str
    source_window_id: str
    concept_name: str
    regime_relation: RegimeRelation
    matching_distance: float
    concept_delta: float

    @field_validator("base_window_id", "source_window_id", "concept_name")
    @classmethod
    def _identifiers_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("intervention pair identifiers must not be blank")
        return value

    @field_validator("matching_distance")
    @classmethod
    def _distance_is_valid(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("matching_distance must be finite and non-negative")
        return value

    @field_validator("concept_delta")
    @classmethod
    def _delta_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("concept_delta must be finite")
        return value

    @model_validator(mode="after")
    def _windows_differ(self) -> Self:
        if self.base_window_id == self.source_window_id:
            raise ValueError("base and source windows must differ")
        return self


@dataclass(frozen=True, slots=True)
class InterventionPairSet:
    pair_ids: tuple[str, ...]
    source_label: str


def validate_intervention_pair_set(pair_set: InterventionPairSet) -> InterventionPairSet:
    if not pair_set.source_label.strip():
        raise ValueError("pair set source_label must not be blank")
    if not pair_set.pair_ids:
        raise ValueError("pair set must contain pair IDs")
    if any(re.fullmatch(r"[0-9a-f]{64}", pair_id) is None for pair_id in pair_set.pair_ids):
        raise ValueError("pair IDs must be stable SHA-256 identifiers")
    if len(set(pair_set.pair_ids)) != len(pair_set.pair_ids):
        raise ValueError("pair IDs must be unique")
    return pair_set


@dataclass(frozen=True, slots=True)
class ResolvedInterventionPairBatch:
    pairs: tuple[InterventionPair, ...]
    base: WindowBatch
    source: WindowBatch
    base_row_for_pair: tuple[int, ...]
    source_row_for_pair: tuple[int, ...]
    dataset_hash: Sha256Hash


def _window_split(batch: WindowBatch, label: str) -> SplitPartition:
    raw_partition = batch.metadata.get("partition", batch.metadata.get("physical_partition"))
    if not isinstance(raw_partition, str):
        raise ValueError(f"{label} WindowBatch must record its physical partition")
    mapping = {
        "TRAIN": SplitPartition.TRAIN,
        "VALIDATION": SplitPartition.VALIDATION,
        "TEST": SplitPartition.TEST,
        "TEST_SEEN_REGIME": SplitPartition.TEST,
        "TEST_UNSEEN_REGIME": SplitPartition.TEST,
    }
    try:
        return mapping[raw_partition]
    except KeyError as exc:
        raise ValueError(f"{label} WindowBatch has an unknown physical partition") from exc


def _validate_pair_rows(
    batch: ResolvedInterventionPairBatch,
) -> None:
    for index, pair in enumerate(batch.pairs):
        base_row = batch.base_row_for_pair[index]
        source_row = batch.source_row_for_pair[index]
        if type(base_row) is not int or not 0 <= base_row < len(batch.base.window_id):
            raise ValueError("base row index is outside the WindowBatch")
        if type(source_row) is not int or not 0 <= source_row < len(batch.source.window_id):
            raise ValueError("source row index is outside the WindowBatch")
        if batch.base.window_id[base_row] != pair.base_window_id:
            raise ValueError("pair base window direction does not match base_row_for_pair")
        if batch.source.window_id[source_row] != pair.source_window_id:
            raise ValueError("pair source window direction does not match source_row_for_pair")


def validate_resolved_intervention_pair_batch(
    batch: ResolvedInterventionPairBatch,
) -> ResolvedInterventionPairBatch:
    pair_count = len(batch.pairs)
    if pair_count == 0:
        raise ValueError("resolved pair batch must not be empty")
    if not (pair_count == len(batch.base_row_for_pair) == len(batch.source_row_for_pair)):
        raise ValueError("resolved pair rows must align with pairs")
    validate_window_batch(batch.base)
    validate_window_batch(batch.source)
    base_split = _window_split(batch.base, "base")
    source_split = _window_split(batch.source, "source")
    if base_split is not source_split:
        raise ValueError("base and source WindowBatch partitions must agree")
    if any(pair.partition is not base_split for pair in batch.pairs):
        raise ValueError("pair partition must match the physical WindowBatch partition")
    pair_ids = tuple(pair.pair_id for pair in batch.pairs)
    if len(set(pair_ids)) != pair_count:
        raise ValueError("resolved pair IDs must be unique")
    if re.fullmatch(r"[0-9a-f]{64}", batch.dataset_hash) is None:
        raise ValueError("resolved pair dataset_hash must be a lowercase SHA-256 hash")
    _validate_pair_rows(batch)
    return batch


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    site_name: str
    layer: int | None
    variable_index: int | None
    patch_index: int | None
    lag: int
    subspace_basis: Tensor | None
    intervention_kind: InterventionKind


def validate_intervention_spec(
    spec: InterventionSpec,
    orthogonality_tolerance: float,
) -> InterventionSpec:
    if not spec.site_name.strip():
        raise ValueError("site_name must not be blank")
    indices = (spec.layer, spec.variable_index, spec.patch_index, spec.lag)
    if any(value is not None and value < 0 for value in indices):
        raise ValueError("intervention indices and lag must be non-negative")
    if not isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0:
        raise ValueError("orthogonality_tolerance must be finite and positive")
    if spec.intervention_kind is InterventionKind.FULL_SWAP:
        if spec.subspace_basis is not None:
            raise ValueError("FULL_SWAP must not carry a subspace basis")
        return spec
    basis = spec.subspace_basis
    if basis is None or not isinstance(basis, Tensor) or basis.ndim != 2:
        raise ValueError("SUBSPACE_SWAP requires a rank-2 subspace basis")
    if not basis.is_floating_point() or not bool(torch.isfinite(basis).all()):
        raise ValueError("subspace basis must contain finite floating values")
    if min(basis.shape) <= 0 or basis.shape[1] > basis.shape[0]:
        raise ValueError("subspace basis must have shape [features, subspace]")
    gram = basis.transpose(0, 1) @ basis
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    if not torch.allclose(gram, identity, rtol=0.0, atol=orthogonality_tolerance):
        raise ValueError("subspace basis columns must be orthonormal")
    return spec


@dataclass(frozen=True, slots=True)
class InterventionResult:
    pair_id: str
    spec: InterventionSpec
    factual: ForecastDistribution
    intervened: ForecastDistribution
