from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tarca.monitoring.repository import MonitoringRepository, open_readonly


def test_repository_builds_only_explicit_runtime_views(monitoring_database: Path) -> None:
    snapshot = MonitoringRepository(monitoring_database).snapshot()

    assert snapshot.run.run_id == "run-a"
    assert snapshot.run.total_tasks == 1
    assert snapshot.jobs[0].world_id == "lorenz96_f10_v2"
    assert snapshot.jobs[0].expected_cpu_cores == 4
    assert snapshot.jobs[0].actual_effective_busy_cores == 18.5
    assert snapshot.jobs[0].actual_vram_bytes == 18 * 1024**3
    assert snapshot.jobs[0].epoch == 3
    assert {resource.label for resource in snapshot.resources} == {"主机", "GPU 0", "GPU 1"}
    assert snapshot.alerts[0].category == "GPU_PRESSURE"


def test_monitoring_connection_is_sqlite_read_only(monitoring_database: Path) -> None:
    with (
        open_readonly(monitoring_database) as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("DELETE FROM alerts")
