"""Reusable validation for contract metadata and values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch

from .types import JSONMetadata, JSONValue


def validate_json_metadata(metadata: object, *, field_path: str = "metadata") -> JSONMetadata:
    """Validate and freeze JSON-compatible metadata with path-aware errors."""
    validated = _validate_json_value(metadata, field_path)
    if not isinstance(validated, Mapping):
        raise ValueError(f"{field_path}: expected a string-key mapping")
    return validated


def _validate_json_value(value: object, field_path: str) -> JSONValue:
    if isinstance(value, torch.Tensor):
        raise ValueError(f"{field_path}: tensors are not JSON-compatible")
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_path}: floats must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_path}: mapping keys must be strings")
            frozen[key] = _validate_json_value(nested_value, f"{field_path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _validate_json_value(item, f"{field_path}[{index}]") for index, item in enumerate(value)
        )
    raise ValueError(f"{field_path}: value is not JSON-compatible")
