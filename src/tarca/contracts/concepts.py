"""Validated immutable concept batch contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ConceptBatch:
    """Concept values aligned to forecast windows."""

    values: Tensor
    valid_mask: Tensor
    names: tuple[str, ...]
    window_id: tuple[str, ...]
    computed_from_history_only: bool
    definition_version: str

    def __post_init__(self) -> None:
        batch_size, concept_count = _validate_values(self.values)
        _validate_mask(self.valid_mask, self.values)
        names = _validate_string_tuple(self.names, "names", concept_count)
        window_id = _validate_string_tuple(self.window_id, "window_id", batch_size)
        if type(self.computed_from_history_only) is not bool:
            raise ValueError("computed_from_history_only: expected an explicit bool")
        if not isinstance(self.definition_version, str) or not self.definition_version.strip():
            raise ValueError("definition_version: expected a non-empty string")

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "window_id", window_id)


def _validate_values(values: object) -> tuple[int, int]:
    if not isinstance(values, Tensor):
        raise ValueError("values: expected a torch.Tensor")
    if not torch.is_floating_point(values):
        raise ValueError("values: expected a floating tensor")
    if values.device.type == "meta":
        raise ValueError("values: values must be materialized and finite")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values: values must be finite")
    if values.ndim != 2:
        raise ValueError("values: expected rank 2 [B, K]")
    if any(dimension <= 0 for dimension in values.shape):
        raise ValueError("values: dimensions must both be positive")
    return tuple(values.shape)  # type: ignore[return-value]


def _validate_mask(valid_mask: object, values: Tensor) -> None:
    if not isinstance(valid_mask, Tensor):
        raise ValueError("valid_mask: expected a torch.Tensor")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask: shape must exactly match values")
    if valid_mask.device != values.device:
        raise ValueError("valid_mask: device must match values")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask: expected bool dtype")


def _validate_string_tuple(value: object, field_name: str, expected_size: int) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    normalized = tuple(value)
    if not all(isinstance(item, str) for item in normalized):
        raise ValueError(f"{field_name}: expected a sequence of strings")
    if len(normalized) != expected_size:
        raise ValueError(f"{field_name}: expected {expected_size} entries")
    if any(not item.strip() for item in normalized):
        raise ValueError(f"{field_name}: entries must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name}: entries must be unique")
    return normalized
