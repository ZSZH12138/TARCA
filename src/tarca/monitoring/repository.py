from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import psutil

from tarca.monitoring.schemas import (
    AlertView,
    JobStatusView,
    ResourceView,
    RunSummaryView,
    RuntimeSnapshotView,
)

_GIB = 1024**3

SAFE_JOB_COLUMNS = (
    "task_id",
    "phase",
    "world_id",
    "model_id",
    "seed",
    "state",
    "pid",
    "alive",
    "gpu_ids",
    "expected_cpu_cores",
    "actual_effective_busy_cores",
    "expected_ram_bytes",
    "actual_rss_bytes",
    "expected_vram_bytes",
    "actual_vram_bytes",
    "epoch",
    "batch",
    "heartbeat_at_utc",
    "retry_count",
    "eta_seconds",
    "error_category",
)


def open_readonly(database_path: Path) -> sqlite3.Connection:
    resolved = database_path.resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
        timeout=5.0,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _object(payload: str | None) -> dict[str, Any]:
    if payload is None:
        return {}
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("monitoring database JSON payload must contain an object")
    return cast(dict[str, Any], value)


def _integer(value: object, default: int = 0) -> int:
    return value if type(value) is int else default


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default


def _gpu_samples(resource: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = resource.get("gpu_samples")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, dict))


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    return datetime.fromisoformat(value)


class MonitoringRepository:
    """Read-only projection from execution state to approved runtime fields."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve(strict=True)

    def _run_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT run_id, graph_id, status, created_at_utc
            FROM runs ORDER BY created_at_utc DESC, run_id DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise ValueError("monitoring database contains no run")
        return cast(sqlite3.Row, row)

    def _job_rows(
        self,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> tuple[sqlite3.Row, ...]:
        rows = connection.execute(
            """
            SELECT attempts.attempt_id, attempts.attempt_number, attempts.state,
                   attempts.pid, attempts.heartbeat_at_utc, attempts.error_category,
                   attempts.allocation_json, job_nodes.task_id, job_nodes.phase,
                   job_nodes.scientific_identity_json, job_nodes.resource_request_json,
                   (SELECT payload_json FROM progress_events
                    WHERE progress_events.attempt_id = attempts.attempt_id
                    ORDER BY event_id DESC LIMIT 1) AS progress_json,
                   (SELECT payload_json FROM resource_samples
                    WHERE resource_samples.attempt_id = attempts.attempt_id
                    ORDER BY sample_id DESC LIMIT 1) AS resource_json
            FROM attempts JOIN job_nodes USING(task_id)
            WHERE job_nodes.run_id = ?
              AND attempts.attempt_number = (
                  SELECT MAX(latest.attempt_number) FROM attempts AS latest
                  WHERE latest.task_id = attempts.task_id
              )
            ORDER BY job_nodes.task_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(rows)

    @staticmethod
    def _job_view(row: sqlite3.Row) -> JobStatusView:
        identity = _object(row["scientific_identity_json"])
        request = _object(row["resource_request_json"])
        allocation = _object(row["allocation_json"])
        progress = _object(row["progress_json"])
        resource = _object(row["resource_json"])
        gpu_ids = tuple(
            int(item)
            for item in allocation.get("gpu_ids", [])
            if type(item) is int and item >= 0
        )
        actual_vram = sum(
            _integer(sample.get("memory_used_bytes"))
            for sample in _gpu_samples(resource)
            if _integer(sample.get("gpu_id"), -1) in gpu_ids
        )
        pid = row["pid"] if type(row["pid"]) is int else None
        return JobStatusView(
            task_id=str(row["task_id"]),
            phase=str(row["phase"]),
            world_id=(str(identity["data_id"]) if "data_id" in identity else None),
            model_id=(str(identity["model_id"]) if "model_id" in identity else None),
            seed=_integer(identity.get("seed"), -1) if "seed" in identity else None,
            state=str(row["state"]),
            pid=pid,
            alive=bool(pid is not None and psutil.pid_exists(pid)),
            gpu_ids=gpu_ids,
            expected_cpu_cores=_integer(request.get("cpu_threads")),
            actual_effective_busy_cores=_number(resource.get("effective_busy_cores")),
            expected_ram_bytes=round(_number(request.get("host_memory_gib")) * _GIB),
            actual_rss_bytes=_integer(resource.get("process_rss_bytes")),
            expected_vram_bytes=round(_number(request.get("gpu_memory_gib")) * _GIB),
            actual_vram_bytes=actual_vram,
            epoch=_integer(progress.get("epoch"), -1) if "epoch" in progress else None,
            batch=_integer(progress.get("batch"), -1) if "batch" in progress else None,
            heartbeat_at_utc=_datetime(row["heartbeat_at_utc"]),
            retry_count=max(0, int(row["attempt_number"]) - 1),
            eta_seconds=None,
            error_category=(
                str(row["error_category"]) if row["error_category"] is not None else None
            ),
        )

    @staticmethod
    def _run_view(row: sqlite3.Row, jobs: tuple[JobStatusView, ...]) -> RunSummaryView:
        total = len(jobs)
        completed = sum(job.state == "COMPLETED" for job in jobs)
        running = sum(job.state == "RUNNING" for job in jobs)
        failed = sum(job.state in {"FAILED", "STALLED"} for job in jobs)
        pending = total - completed - running - failed
        status = "FAILED" if failed else "COMPLETED" if total and completed == total else "ACTIVE"
        active_phase = next((job.phase for job in jobs if job.state == "RUNNING"), None)
        eta_status: Literal["CALIBRATING", "AVAILABLE", "COMPLETE", "FAILED"] = (
            "FAILED" if failed else "COMPLETE" if status == "COMPLETED" else "CALIBRATING"
        )
        return RunSummaryView(
            run_id=str(row["run_id"]),
            graph_id=str(row["graph_id"]),
            status=status,
            phase=active_phase,
            total_tasks=total,
            completed_tasks=completed,
            running_tasks=running,
            failed_tasks=failed,
            pending_tasks=pending,
            progress_fraction=0.0 if total == 0 else completed / total,
            eta_seconds=None,
            eta_status=eta_status,
            created_at_utc=datetime.fromisoformat(str(row["created_at_utc"])),
        )

    @staticmethod
    def _resources(
        connection: sqlite3.Connection,
        run_id: str,
        jobs: tuple[JobStatusView, ...],
    ) -> tuple[ResourceView, ...]:
        row = connection.execute(
            """
            SELECT payload_json FROM resource_samples
            WHERE run_id = ? ORDER BY sample_id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        sample = _object(None if row is None else row["payload_json"])
        sampled_at = _datetime(sample.get("sampled_at_utc"))
        active = tuple(job for job in jobs if job.state == "RUNNING")
        views = [
            ResourceView(
                resource_id="host",
                label="主机",
                kind="HOST",
                expected_cpu_cores=sum(job.expected_cpu_cores for job in active),
                actual_effective_busy_cores=_number(sample.get("effective_busy_cores")),
                expected_memory_bytes=sum(job.expected_ram_bytes for job in active),
                actual_memory_bytes=_integer(sample.get("host_memory_used_bytes")),
                utilization_percent=_number(sample.get("host_cpu_percent")),
                temperature_celsius=None,
                power_watts=None,
                active_processes=sum(job.alive for job in active),
                sampled_at_utc=sampled_at,
            )
        ]
        for gpu in _gpu_samples(sample):
            gpu_id = _integer(gpu.get("gpu_id"), -1)
            gpu_jobs = tuple(job for job in active if gpu_id in job.gpu_ids)
            processes = gpu.get("compute_pids")
            views.append(
                ResourceView(
                    resource_id=f"gpu-{gpu_id}",
                    label=f"GPU {gpu_id}",
                    kind="GPU",
                    expected_cpu_cores=sum(job.expected_cpu_cores for job in gpu_jobs),
                    actual_effective_busy_cores=0.0,
                    expected_memory_bytes=sum(job.expected_vram_bytes for job in gpu_jobs),
                    actual_memory_bytes=_integer(gpu.get("memory_used_bytes")),
                    utilization_percent=_number(gpu.get("utilization_percent")),
                    temperature_celsius=_number(gpu.get("temperature_celsius")),
                    power_watts=_number(gpu.get("power_watts")),
                    active_processes=len(processes) if isinstance(processes, list) else 0,
                    sampled_at_utc=sampled_at,
                )
            )
        return tuple(views)

    @staticmethod
    def _alerts(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> tuple[AlertView, ...]:
        rows = connection.execute(
            """
            SELECT alerts.alert_id, attempts.task_id, alerts.category,
                   alerts.message, alerts.created_at_utc
            FROM alerts LEFT JOIN attempts USING(attempt_id)
            WHERE alerts.run_id = ? ORDER BY alerts.alert_id DESC LIMIT 200
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            AlertView(
                alert_id=int(row["alert_id"]),
                task_id=str(row["task_id"]) if row["task_id"] is not None else None,
                category=str(row["category"]),
                message=str(row["message"]),
                created_at_utc=datetime.fromisoformat(str(row["created_at_utc"])),
            )
            for row in rows
        )

    def snapshot(self) -> RuntimeSnapshotView:
        with closing(open_readonly(self.database_path)) as connection:
            run_row = self._run_row(connection)
            run_id = str(run_row["run_id"])
            jobs = tuple(self._job_view(row) for row in self._job_rows(connection, run_id))
            return RuntimeSnapshotView(
                run=self._run_view(run_row, jobs),
                jobs=jobs,
                resources=self._resources(connection, run_id, jobs),
                alerts=self._alerts(connection, run_id),
            )
