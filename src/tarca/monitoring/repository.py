from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from types import MappingProxyType
from typing import Any, Literal, cast

import psutil

from tarca.contracts import canonical_json_hash, sha256_file
from tarca.execution.contracts import ResourceRequest, RunPlanNode
from tarca.monitoring.schemas import (
    AlertView,
    JobStatusView,
    ResourceView,
    RunSummaryView,
    RuntimeSnapshotView,
)

_GIB = 1024**3
_LIVE_SAMPLE_SECONDS = 10.0

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


def _optional_integer(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _gpu_samples(resource: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = resource.get("gpu_samples")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, dict))


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _trusted_preflight_eta(runtime: Path) -> float | None:
    """Read a hash-bound runtime estimate without trusting arbitrary UI input."""
    receipt_path = runtime / "preflight_receipt.json"
    evidence_path = runtime / "bootstrap_evidence.json"
    if not receipt_path.is_file() or not evidence_path.is_file():
        return None
    try:
        receipt = _object(receipt_path.read_text(encoding="utf-8"))
        evidence = _object(evidence_path.read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if any(
            (
                receipt.get("schema_version") != "tarca-stage2-preflight-v1",
                receipt.get("status") != "PREFLIGHT_PASS",
                receipt.get("formal_tasks_executed") != 0,
                receipt.get("receipt_sha256") != canonical_json_hash(unsigned),
                receipt.get("evidence_sha256") != sha256_file(evidence_path),
                evidence.get("status") != "PREFLIGHT_PASS",
                evidence.get("eta_gate_passed") is not True,
                evidence.get("formal_tasks_executed") != 0,
            )
        ):
            return None
        eta = _optional_number(evidence.get("estimated_remaining_seconds"))
        return eta if eta is not None and eta > 0 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


@dataclass(frozen=True, slots=True)
class _JobRecord:
    node: RunPlanNode
    attempt_id: str | None
    attempt_number: int
    state: str
    pid: int | None
    heartbeat_at_utc: datetime | None
    error_category: str | None
    allocation: dict[str, Any]
    progress: dict[str, Any]
    resource: dict[str, Any]
    process_started_at_utc: datetime | None
    progress_recorded_at_utc: datetime | None
    updated_at_utc: datetime | None


_ResourceHistoryKey = tuple[int, int, float, float]


@dataclass(frozen=True, slots=True)
class _DurationHistory:
    exact: Mapping[tuple[str, str], float]
    phase: Mapping[str, float]
    resource: Mapping[_ResourceHistoryKey, float]


def _resource_history_key(request: ResourceRequest) -> _ResourceHistoryKey:
    return (
        request.cpu_threads,
        request.gpu_count,
        request.gpu_memory_gib,
        request.host_memory_gib,
    )


def _attempt_rows(connection: sqlite3.Connection, run_id: str) -> dict[str, sqlite3.Row]:
    rows = connection.execute(
        """
        SELECT attempts.attempt_id, attempts.attempt_number, attempts.state,
               attempts.pid, attempts.heartbeat_at_utc, attempts.error_category,
               attempts.allocation_json, attempts.process_started_at_utc,
               attempts.updated_at_utc, job_nodes.task_id, job_nodes.phase,
               job_nodes.scientific_identity_json, job_nodes.resource_request_json,
               (SELECT payload_json FROM progress_events
                WHERE progress_events.attempt_id = attempts.attempt_id
                ORDER BY event_id DESC LIMIT 1) AS progress_json,
               (SELECT recorded_at_utc FROM progress_events
                WHERE progress_events.attempt_id = attempts.attempt_id
                ORDER BY event_id DESC LIMIT 1) AS progress_recorded_at_utc,
               (SELECT payload_json FROM resource_samples
                WHERE resource_samples.attempt_id = attempts.attempt_id
                ORDER BY sample_id DESC LIMIT 1) AS resource_json
        FROM attempts JOIN job_nodes USING(task_id)
        WHERE job_nodes.run_id = ?
          AND attempts.attempt_number = (
              SELECT MAX(latest.attempt_number) FROM attempts AS latest
              WHERE latest.task_id = attempts.task_id
          )
        """,
        (run_id,),
    ).fetchall()
    return {str(row["task_id"]): row for row in rows}


def _fallback_node(row: sqlite3.Row) -> RunPlanNode:
    return RunPlanNode.model_validate(
        {
            "identity": _object(row["scientific_identity_json"]),
            "phase": str(row["phase"]),
            "resource_request": _object(row["resource_request_json"]),
            "dependency_task_ids": (),
        }
    )


def _job_record(node: RunPlanNode, row: sqlite3.Row | None) -> _JobRecord:
    if row is None:
        return _JobRecord(node, None, 0, "PENDING", None, None, None, {}, {}, {}, None, None, None)
    return _JobRecord(
        node=node,
        attempt_id=str(row["attempt_id"]),
        attempt_number=int(row["attempt_number"]),
        state=str(row["state"]),
        pid=int(row["pid"]) if type(row["pid"]) is int else None,
        heartbeat_at_utc=_datetime(row["heartbeat_at_utc"]),
        error_category=(str(row["error_category"]) if row["error_category"] else None),
        allocation=_object(row["allocation_json"]),
        progress=_object(row["progress_json"]),
        resource=_object(row["resource_json"]),
        process_started_at_utc=_datetime(row["process_started_at_utc"]),
        progress_recorded_at_utc=_datetime(row["progress_recorded_at_utc"]),
        updated_at_utc=_datetime(row["updated_at_utc"]),
    )


_PROGRESS_PAIRS: tuple[tuple[str, str], ...] = (
    ("completed_steps", "total_steps"),
    ("completed_conditions", "total_conditions"),
    ("completed_base_groups", "total_base_groups"),
    ("completed_seed_worlds", "total_seed_worlds"),
    ("completed_seed_reports", "total_seed_reports"),
)


def _progress_counts(progress: Mapping[str, Any]) -> tuple[float | None, float | None]:
    for completed_key, total_key in _PROGRESS_PAIRS:
        if completed_key in progress or total_key in progress:
            return (
                _optional_number(progress.get(completed_key)),
                _optional_number(progress.get(total_key)),
            )
    return None, None


def _remaining_seconds(record: _JobRecord) -> float | None:
    if record.state == "COMPLETED":
        return 0.0
    if record.state != "RUNNING":
        return None
    completed, total = _progress_counts(record.progress)
    started = record.process_started_at_utc
    progressed = record.progress_recorded_at_utc
    if completed is None or total is None or started is None or progressed is None:
        return None
    elapsed = (progressed - started).total_seconds()
    if completed <= 0 or total <= 0 or completed > total or elapsed < 0:
        return None
    remaining = elapsed * (total - completed) / completed
    return remaining if math.isfinite(remaining) and remaining >= 0 else None


def _actual_vram(record: _JobRecord, gpu_ids: tuple[int, ...]) -> int | None:
    if not record.resource:
        return None
    if not gpu_ids:
        return 0
    samples = {
        _integer(sample.get("gpu_id"), -1): sample for sample in _gpu_samples(record.resource)
    }
    if any(gpu_id not in samples for gpu_id in gpu_ids):
        return None
    return sum(_integer(samples[gpu_id].get("memory_used_bytes")) for gpu_id in gpu_ids)


class MonitoringRepository:
    """Read-only projection from execution state to approved runtime fields."""

    def __init__(
        self,
        database_path: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.database_path = database_path.resolve(strict=True)
        self._now = now or (lambda: datetime.now(UTC))

    @staticmethod
    def _run_row(connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT run_id, graph_id, status, created_at_utc "
            "FROM runs ORDER BY created_at_utc DESC, run_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("monitoring database contains no run")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _job_records(connection: sqlite3.Connection, run_id: str) -> tuple[_JobRecord, ...]:
        attempts = _attempt_rows(connection, run_id)
        plan_rows = connection.execute(
            "SELECT task_id, node_json FROM run_plan_nodes WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        if not plan_rows:
            return tuple(_job_record(_fallback_node(row), row) for row in attempts.values())
        planned_ids = {str(row["task_id"]) for row in plan_rows}
        if set(attempts) - planned_ids:
            raise ValueError("execution state contains a task outside the immutable run plan")
        return tuple(
            _job_record(
                RunPlanNode.model_validate_json(row["node_json"]),
                attempts.get(str(row["task_id"])),
            )
            for row in plan_rows
        )

    @staticmethod
    def _history(records: tuple[_JobRecord, ...]) -> _DurationHistory:
        exact_durations: dict[tuple[str, str], list[float]] = {}
        phase_durations: dict[str, list[float]] = {}
        resource_durations: dict[_ResourceHistoryKey, list[float]] = {}
        for record in records:
            if record.state != "COMPLETED":
                continue
            started = record.process_started_at_utc
            finished = record.updated_at_utc
            if started is None or finished is None:
                continue
            duration = (finished - started).total_seconds()
            if not math.isfinite(duration) or duration < 0:
                continue
            exact_key = (record.node.phase, record.node.identity.model_id)
            phase_key = record.node.phase
            resource_key = _resource_history_key(record.node.resource_request)
            exact_durations[exact_key] = [*exact_durations.get(exact_key, []), duration]
            phase_durations[phase_key] = [*phase_durations.get(phase_key, []), duration]
            resource_durations[resource_key] = [
                *resource_durations.get(resource_key, []),
                duration,
            ]
        return _DurationHistory(
            exact=MappingProxyType(
                {key: float(median(values)) for key, values in exact_durations.items()}
            ),
            phase=MappingProxyType(
                {key: float(median(values)) for key, values in phase_durations.items()}
            ),
            resource=MappingProxyType(
                {key: float(median(values)) for key, values in resource_durations.items()}
            ),
        )

    @staticmethod
    def _job_view(record: _JobRecord, history: _DurationHistory) -> JobStatusView:
        request = record.node.resource_request
        gpu_ids = tuple(
            int(item)
            for item in record.allocation.get("gpu_ids", [])
            if type(item) is int and item >= 0
        )
        eta = _remaining_seconds(record)
        if record.state in {"PENDING", "READY"}:
            candidates = (
                history.exact.get((record.node.phase, record.node.identity.model_id)),
                history.phase.get(record.node.phase),
                history.resource.get(_resource_history_key(request)),
            )
            eta = next((candidate for candidate in candidates if candidate is not None), None)
        return JobStatusView(
            task_id=record.node.task_id,
            phase=record.node.phase,
            world_id=record.node.identity.data_id,
            model_id=record.node.identity.model_id,
            seed=record.node.identity.seed,
            state=record.state,
            pid=record.pid,
            alive=bool(record.pid is not None and psutil.pid_exists(record.pid)),
            gpu_ids=gpu_ids,
            expected_cpu_cores=request.cpu_threads,
            actual_effective_busy_cores=_optional_number(
                record.resource.get("effective_busy_cores")
            ),
            cpu_affinity_ids=tuple(
                int(cpu)
                for cpu in record.resource.get("process_affinity_cpu_ids", [])
                if type(cpu) is int and cpu >= 0
            ),
            expected_ram_bytes=round(request.host_memory_gib * _GIB),
            actual_rss_bytes=_optional_integer(record.resource.get("process_rss_bytes")),
            expected_vram_bytes=round(request.gpu_memory_gib * _GIB),
            actual_vram_bytes=_actual_vram(record, gpu_ids),
            epoch=(
                _integer(record.progress.get("epoch"), -1) if "epoch" in record.progress else None
            ),
            batch=(
                _integer(record.progress.get("batch"), -1) if "batch" in record.progress else None
            ),
            heartbeat_at_utc=record.heartbeat_at_utc,
            retry_count=max(0, record.attempt_number - 1),
            eta_seconds=eta,
            error_category=record.error_category,
        )

    @staticmethod
    def _latest_run_sample(
        connection: sqlite3.Connection, run_id: str
    ) -> tuple[dict[str, Any], datetime | None]:
        row = connection.execute(
            """
            SELECT sampled_at_utc, payload_json FROM resource_samples
            WHERE run_id = ? AND attempt_id IS NULL ORDER BY sample_id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return {}, None
        return _object(row["payload_json"]), _datetime(row["sampled_at_utc"])

    @staticmethod
    def _latest_checkpoint(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> datetime | None:
        row = connection.execute(
            """
            SELECT MAX(attempts.updated_at_utc) AS checkpoint_at_utc
            FROM attempts JOIN job_nodes USING(task_id)
            WHERE job_nodes.run_id = ? AND attempts.state = 'COMPLETED'
            """,
            (run_id,),
        ).fetchone()
        return None if row is None else _datetime(row["checkpoint_at_utc"])

    def _telemetry_status(
        self, sampled_at: datetime | None
    ) -> Literal["LIVE", "STALE", "UNAVAILABLE"]:
        if sampled_at is None:
            return "UNAVAILABLE"
        age = (self._now() - sampled_at).total_seconds()
        return "LIVE" if 0 <= age <= _LIVE_SAMPLE_SECONDS else "STALE"

    @staticmethod
    def _run_view(
        row: sqlite3.Row,
        jobs: tuple[JobStatusView, ...],
        last_sampled_at: datetime | None,
        last_checkpoint_at: datetime | None,
        preflight_eta_seconds: float | None,
    ) -> RunSummaryView:
        total = len(jobs)
        completed = sum(job.state == "COMPLETED" for job in jobs)
        running = sum(job.state == "RUNNING" for job in jobs)
        failed = sum(job.state in {"FAILED", "STALLED"} for job in jobs)
        pending = total - completed - running - failed
        status = "FAILED" if failed else "COMPLETED" if total and completed == total else "ACTIVE"
        eta_status: Literal["CALIBRATING", "AVAILABLE", "COMPLETE", "FAILED"]
        eta_source: Literal["NONE", "PREFLIGHT_ESTIMATE", "RUNTIME_PROGRESS", "COMPLETE"]
        eta: float | None = None
        if failed:
            eta_status = "FAILED"
            eta_source = "NONE"
        elif status == "COMPLETED":
            eta_status, eta = "COMPLETE", 0.0
            eta_source = "COMPLETE"
        else:
            unknown = tuple(
                job
                for job in jobs
                if job.state in {"RUNNING", "PENDING", "READY"} and job.eta_seconds is None
            )
            running_etas = tuple(
                job.eta_seconds
                for job in jobs
                if job.state == "RUNNING" and job.eta_seconds is not None
            )
            pending_etas = tuple(
                job.eta_seconds
                for job in jobs
                if job.state in {"PENDING", "READY"} and job.eta_seconds is not None
            )
            if unknown or not running_etas:
                if preflight_eta_seconds is None:
                    eta_status = "CALIBRATING"
                    eta_source = "NONE"
                else:
                    eta_status, eta = "AVAILABLE", preflight_eta_seconds
                    eta_source = "PREFLIGHT_ESTIMATE"
            else:
                eta_status = "AVAILABLE"
                eta = max(running_etas) + sum(pending_etas)
                eta_source = "RUNTIME_PROGRESS"
        return RunSummaryView(
            run_id=str(row["run_id"]),
            graph_id=str(row["graph_id"]),
            status=status,
            phase=next((job.phase for job in jobs if job.state == "RUNNING"), None),
            total_tasks=total,
            completed_tasks=completed,
            running_tasks=running,
            failed_tasks=failed,
            pending_tasks=pending,
            progress_fraction=0.0 if total == 0 else completed / total,
            eta_seconds=eta,
            eta_status=eta_status,
            eta_source=eta_source,
            created_at_utc=datetime.fromisoformat(str(row["created_at_utc"])),
            last_sampled_at_utc=last_sampled_at,
            last_checkpoint_at_utc=last_checkpoint_at,
        )

    def _resources(
        self,
        jobs: tuple[JobStatusView, ...],
        sample: dict[str, Any],
        sampled_at: datetime | None,
    ) -> tuple[ResourceView, ...]:
        status = self._telemetry_status(sampled_at)
        active = tuple(job for job in jobs if job.state == "RUNNING")
        views = [
            ResourceView(
                resource_id="host",
                label="主机",
                kind="HOST",
                expected_cpu_cores=sum(job.expected_cpu_cores for job in active),
                actual_effective_busy_cores=_optional_number(sample.get("effective_busy_cores")),
                expected_memory_bytes=sum(job.expected_ram_bytes for job in active),
                actual_memory_bytes=_optional_integer(sample.get("host_memory_used_bytes")),
                utilization_percent=_optional_number(sample.get("host_cpu_percent")),
                temperature_celsius=None,
                power_watts=None,
                active_processes=sum(job.alive for job in active),
                disk_read_bytes_per_second=_optional_number(
                    sample.get("disk_read_bytes_per_second")
                ),
                disk_write_bytes_per_second=_optional_number(
                    sample.get("disk_write_bytes_per_second")
                ),
                sampled_at_utc=sampled_at,
                telemetry_status=status,
            )
        ]
        gpu_samples = {_integer(item.get("gpu_id"), -1): item for item in _gpu_samples(sample)}
        gpu_ids = sorted({*gpu_samples, *(gpu_id for job in active for gpu_id in job.gpu_ids)})
        for gpu_id in gpu_ids:
            gpu = gpu_samples.get(gpu_id, {})
            gpu_jobs = tuple(job for job in active if gpu_id in job.gpu_ids)
            processes = gpu.get("compute_pids")
            views.append(
                ResourceView(
                    resource_id=f"gpu-{gpu_id}",
                    label=f"GPU {gpu_id}",
                    kind="GPU",
                    expected_cpu_cores=sum(job.expected_cpu_cores for job in gpu_jobs),
                    actual_effective_busy_cores=None,
                    expected_memory_bytes=sum(job.expected_vram_bytes for job in gpu_jobs),
                    actual_memory_bytes=_optional_integer(gpu.get("memory_used_bytes")),
                    utilization_percent=_optional_number(gpu.get("utilization_percent")),
                    temperature_celsius=_optional_number(gpu.get("temperature_celsius")),
                    power_watts=_optional_number(gpu.get("power_watts")),
                    active_processes=(
                        len(processes)
                        if isinstance(processes, list)
                        else sum(job.alive for job in gpu_jobs)
                    ),
                    disk_read_bytes_per_second=None,
                    disk_write_bytes_per_second=None,
                    sampled_at_utc=sampled_at,
                    telemetry_status=status,
                )
            )
        return tuple(views)

    @staticmethod
    def _alerts(connection: sqlite3.Connection, run_id: str) -> tuple[AlertView, ...]:
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
            records = self._job_records(connection, run_id)
            history = self._history(records)
            jobs = tuple(self._job_view(record, history) for record in records)
            sample, sampled_at = self._latest_run_sample(connection, run_id)
            checkpoint_at = self._latest_checkpoint(connection, run_id)
            return RuntimeSnapshotView(
                run=self._run_view(
                    run_row,
                    jobs,
                    sampled_at,
                    checkpoint_at,
                    _trusted_preflight_eta(self.database_path.parent),
                ),
                jobs=jobs,
                resources=self._resources(jobs, sample, sampled_at),
                alerts=self._alerts(connection, run_id),
            )
