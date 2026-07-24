"""Immutable value objects used by the Stage 0 diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

Status = Literal["PASS", "FAIL", "WARN", "SKIP"]
_VALID_STATUSES = frozenset({"PASS", "FAIL", "WARN", "SKIP"})


def _freeze(value: Any) -> Any:
    """Return an immutable, detached representation of a JSON-like value."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a fresh, JSON-serializable representation of a frozen value."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_thaw(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one focused diagnostic component."""

    name: str
    status: Status
    details: Mapping[str, Any]
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CheckResult.name must not be blank.")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported diagnostic status: {self.status!r}")
        object.__setattr__(self, "details", _freeze(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh plain object suitable for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "details": _thaw(self.details),
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DoctorReport:
    """Immutable ordered collection of diagnostic results."""

    results: tuple[CheckResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def has_failures(self) -> bool:
        return any(result.status == "FAIL" for result in self.results)

    @property
    def overall_status(self) -> Literal["PASS", "FAIL"]:
        return "FAIL" if self.has_failures else "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Return a new stable-schema JSON payload."""
        summary = {
            status: sum(result.status == status for result in self.results)
            for status in ("PASS", "FAIL", "WARN", "SKIP")
        }
        return {
            "schema_version": "1.0",
            "summary": summary,
            "results": [result.to_dict() for result in self.results],
        }
