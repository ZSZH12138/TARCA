from __future__ import annotations

from math import isfinite

from pydantic import Field, field_validator

from .base import PROTOCOL_ID, Sha256Hash, StrictContractModel
from .data import SplitPartition


class MetricContext(StrictContractModel):
    experiment_id: str
    run_id: str
    split: SplitPartition
    data_hash: Sha256Hash
    model_id: str | None
    protocol_id: str
    gate_scope: str | None

    @field_validator("experiment_id", "run_id")
    @classmethod
    def _identifiers_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metric context identifiers must not be blank")
        return value

    @field_validator("model_id", "gate_scope")
    @classmethod
    def _optional_identifiers_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional metric context identifiers must not be blank")
        return value

    @field_validator("protocol_id")
    @classmethod
    def _protocol_is_frozen(cls, value: str) -> str:
        if value != PROTOCOL_ID:
            raise ValueError("protocol_id must match the frozen protocol")
        return value


class MetricRecord(StrictContractModel):
    experiment_id: str
    run_id: str
    split: SplitPartition
    metric_name: str
    value: float
    regime: str | int | None
    horizon: int | None = Field(default=None, ge=0)
    concept: str | None

    @field_validator("experiment_id", "run_id", "metric_name")
    @classmethod
    def _identifiers_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metric identifiers must not be blank")
        return value

    @field_validator("value")
    @classmethod
    def _value_is_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("metric value must be finite")
        return value

    @field_validator("regime", "concept")
    @classmethod
    def _optional_text_not_blank(cls, value: str | int | None) -> str | int | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("optional metric labels must not be blank")
        return value
