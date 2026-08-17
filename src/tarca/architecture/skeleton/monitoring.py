"""Read-only monitoring skeleton."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.monitoring import MonitoringSnapshot


class ReadOnlyMonitor(Protocol):
    def read_status(self) -> MonitoringSnapshot: ...

    def read_resources(self) -> MonitoringSnapshot: ...

    def read_telemetry(self) -> MonitoringSnapshot: ...


def validate_monitoring_snapshot(snapshot: MonitoringSnapshot) -> MonitoringSnapshot:
    if not isinstance(snapshot, MonitoringSnapshot):
        raise TypeError("snapshot must be MonitoringSnapshot")
    return snapshot
