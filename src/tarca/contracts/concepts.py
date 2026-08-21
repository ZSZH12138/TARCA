from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol, Self, runtime_checkable

import torch
from pydantic import Field, field_validator, model_validator
from torch import Tensor

from .base import Sha256Hash, StrictContractModel
from .data import LeakageAudit, WindowBatch


class ConceptSpec(StrictContractModel):
    name: str
    definition_version: str
    required_history: int = Field(ge=0)
    history_only: bool
    source_kind: Literal["ANALYTIC", "WEAK_SUPERVISION", "CONSTRAINED_LEARNED", "SYNTHETIC_TRUTH"]
    intervention_semantics: str
    valid_range: tuple[float | None, float | None]
    expected_effect_components: tuple[str, ...]
    definition_hash: Sha256Hash

    @field_validator("name", "definition_version", "intervention_semantics")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("concept text fields must not be blank")
        return value

    @field_validator("expected_effect_components")
    @classmethod
    def _components_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not component.strip() for component in value):
            raise ValueError("effect components must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("effect components must be unique")
        return value

    @model_validator(mode="after")
    def _range_is_ordered_and_finite(self) -> Self:
        lower, upper = self.valid_range
        if any(bound is not None and not isfinite(bound) for bound in self.valid_range):
            raise ValueError("valid_range bounds must be finite")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("valid_range lower bound must not exceed upper bound")
        return self


@dataclass(frozen=True, slots=True)
class ConceptBatch:
    values: Tensor
    valid_mask: Tensor
    names: tuple[str, ...]
    window_id: tuple[str, ...]
    computed_from_history_only: bool
    definition_version: str


def validate_concept_batch(batch: ConceptBatch) -> ConceptBatch:
    if not isinstance(batch.values, Tensor) or batch.values.ndim != 2:
        raise ValueError("concept values must be a rank-2 Tensor")
    batch_size, concept_count = batch.values.shape
    if min(batch_size, concept_count) <= 0:
        raise ValueError("concept values dimensions must be positive")
    if not batch.values.is_floating_point() or not bool(torch.isfinite(batch.values).all()):
        raise ValueError("concept values must be finite floating values")
    if not isinstance(batch.valid_mask, Tensor) or batch.valid_mask.shape != batch.values.shape:
        raise ValueError("concept valid_mask shape must match values")
    if batch.valid_mask.dtype is not torch.bool:
        raise ValueError("concept valid_mask must have bool dtype")
    if batch.valid_mask.device != batch.values.device:
        raise ValueError("concept valid_mask device must match values")
    if len(batch.names) != concept_count or len(batch.window_id) != batch_size:
        raise ValueError("concept names and window IDs must match tensor axes")
    if any(not value.strip() for value in (*batch.names, *batch.window_id)):
        raise ValueError("concept names and window IDs must not be blank")
    if len(set(batch.names)) != len(batch.names) or len(set(batch.window_id)) != len(
        batch.window_id
    ):
        raise ValueError("concept names and window IDs must be unique")
    if not batch.definition_version.strip():
        raise ValueError("concept definition_version must not be blank")
    return batch


@dataclass(frozen=True, slots=True)
class ConceptIntervention:
    concept_name: str
    delta: float

    def __post_init__(self) -> None:
        if not self.concept_name.strip():
            raise ValueError("concept_name must not be blank")
        if not isfinite(self.delta):
            raise ValueError("concept delta must be finite")


@runtime_checkable
class ConceptExtractor(Protocol):
    def compute(self, batch: WindowBatch) -> ConceptBatch: ...

    def leakage_audit(self, batch: WindowBatch) -> LeakageAudit: ...
