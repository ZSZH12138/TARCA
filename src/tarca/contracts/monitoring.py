"""Terminal-safe monitoring payloads with no scientific result fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    """Read-only scheduler/resource view; never a source of scientific metrics."""

    phase: str
    terminal_status: str
    task_counts: Mapping[str, int]
    resource_summary: Mapping[str, float]
    heartbeat_age_seconds: float
    eta_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_counts, Mapping) or not isinstance(
            self.resource_summary, Mapping
        ):
            raise TypeError("monitoring summaries must be mappings")
        object.__setattr__(self, "task_counts", MappingProxyType(dict(self.task_counts)))
        object.__setattr__(self, "resource_summary", MappingProxyType(dict(self.resource_summary)))
