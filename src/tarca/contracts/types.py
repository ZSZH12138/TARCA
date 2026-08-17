"""Shared enums and JSON-compatible metadata types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 server compatibility
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042 - compatibility fallback for Python 3.10
        def __str__(self) -> str:
            return str(self.value)


class SplitPartition(StrEnum):
    """Dataset split partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RegimeRelation(StrEnum):
    """Relationship between source and target regimes."""

    SAME = "same"
    CROSS = "cross"
    UNKNOWN = "unknown"


class InterventionKind(StrEnum):
    """Supported intervention mechanisms."""

    FULL_SWAP = "full_swap"
    SUBSPACE_SWAP = "subspace_swap"


class RunStatus(StrEnum):
    """Minimal honest lifecycle states for a manifest run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]
JSONMetadata: TypeAlias = Mapping[str, JSONValue]
