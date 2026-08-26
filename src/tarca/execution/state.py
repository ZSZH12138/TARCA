from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from pydantic import BaseModel

from tarca.contracts import ArtifactRef, canonical_json_bytes
from tarca.execution.contracts import ResourceAllocation, TaskSpec

ArtifactVerifier = Callable[[ArtifactRef], bool]


class AttemptState(StrEnum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALLED = "STALLED"


class RetryDisposition(StrEnum):
    RETRY_ONCE = "RETRY_ONCE"
    RETRY_ONCE_WITH_LOWER_PACKING = "RETRY_ONCE_WITH_LOWER_PACKING"
    TERMINAL = "TERMINAL"


RETRY_POLICY: Mapping[str, RetryDisposition] = MappingProxyType(
    {
        "TRANSIENT_IO": RetryDisposition.RETRY_ONCE,
        "WORKER_DIED": RetryDisposition.RETRY_ONCE,
        "CUDA_OOM": RetryDisposition.RETRY_ONCE_WITH_LOWER_PACKING,
        "HASH_DRIFT": RetryDisposition.TERMINAL,
        "TRUTH_LEAKAGE": RetryDisposition.TERMINAL,
        "NONFINITE": RetryDisposition.TERMINAL,
        "IDENTITY_DRIFT": RetryDisposition.TERMINAL,
    }
)


class StateTransitionConflict(RuntimeError):
    def __init__(self, attempt_id: str) -> None:
        super().__init__(f"attempt state changed concurrently: {attempt_id}")
        self.attempt_id = attempt_id


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    run_id: str
    attempt_id: str
    task: TaskSpec
    executor_key: str
    worker_id: str


@dataclass(frozen=True, slots=True)
class QueuedTask:
    run_id: str
    attempt_id: str
    task: TaskSpec
    executor_key: str
    packing_level: int


@dataclass(frozen=True, slots=True)
class FailedAttempt:
    attempt_id: str
    error_category: str
    packing_level: int


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    process_started_at_utc: datetime
    run_id: str
    task_id: str

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise ValueError("process PID must be positive")
        _require_utc(self.process_started_at_utc)
        if not self.run_id.strip() or not self.task_id.strip():
            raise ValueError("process run and task identities must not be blank")


class ProcessProbe(Protocol):
    def inspect(self, pid: int) -> ProcessIdentity | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    live_task_ids: tuple[str, ...]
    stalled_task_ids: tuple[str, ...]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("execution timestamps must be timezone-aware UTC")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return _require_utc(value).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _require_utc(parsed)


def _safe_identifier(value: str, label: str) -> str:
    if (
        not value.strip()
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError(f"{label} must be a safe logical identifier")
    return value


def _json_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        serializable: object = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        serializable = asdict(value)
    elif isinstance(value, Mapping):
        serializable = dict(value)
    else:
        raise TypeError("state event payload must be a model, dataclass, or mapping")
    return canonical_json_bytes(serializable).decode("utf-8")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_nodes (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    phase TEXT NOT NULL,
    executor_key TEXT NOT NULL,
    output_artifact_type TEXT NOT NULL,
    scientific_identity_json TEXT NOT NULL,
    resource_request_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_specs (
    task_id TEXT PRIMARY KEY REFERENCES job_nodes(task_id),
    spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dependencies (
    task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    dependency_task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    input_ordinal INTEGER NOT NULL CHECK (input_ordinal >= 0),
    PRIMARY KEY (task_id, dependency_task_id),
    UNIQUE (task_id, input_ordinal)
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    state TEXT NOT NULL CHECK (state IN ('READY','RUNNING','COMPLETED','FAILED','STALLED')),
    worker_id TEXT,
    allocation_json TEXT,
    pid INTEGER,
    process_started_at_utc TEXT,
    heartbeat_at_utc TEXT,
    error_category TEXT,
    artifact_json TEXT,
    packing_level INTEGER NOT NULL DEFAULT 1 CHECK (packing_level >= 1),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    UNIQUE (task_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS progress_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    recorded_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    sampled_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    created_at_utc TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_state ON attempts(state, created_at_utc);
CREATE INDEX IF NOT EXISTS idx_progress_attempt ON progress_events(attempt_id, event_id);
"""


class ExecutionStateStore:
    def __init__(
        self,
        database_path: Path,
        *,
        artifact_verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_verifier = artifact_verifier
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def pragma(self, name: str) -> str | int:
        if name not in {"journal_mode", "foreign_keys", "busy_timeout"}:
            raise ValueError("PRAGMA name is not allowlisted")
        with self._connect() as connection:
            row = connection.execute(f"PRAGMA {name}").fetchone()
        if row is None or not isinstance(row[0], (str, int)):
            raise RuntimeError("SQLite PRAGMA returned no scalar value")
        return row[0]

    def table_names(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ? ORDER BY name",
                ("table",),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def create_run(self, run_id: str, graph_id: str) -> None:
        _safe_identifier(run_id, "run_id")
        _safe_identifier(graph_id, "graph_id")
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT graph_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if existing["graph_id"] != graph_id:
                    raise ValueError("run ID is already bound to a different graph")
                return
            connection.execute(
                "INSERT INTO runs(run_id, graph_id, status, created_at_utc) VALUES (?, ?, ?, ?)",
                (run_id, graph_id, "ACTIVE", now),
            )

    def enqueue_task(
        self,
        run_id: str,
        task: TaskSpec,
        executor_key: str,
        dependency_task_ids: tuple[str, ...] = (),
        *,
        packing_level: int = 1,
    ) -> str:
        _safe_identifier(run_id, "run_id")
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", executor_key) is None:
            raise ValueError("executor key must be a safe registry identifier")
        if len(dependency_task_ids) != len(set(dependency_task_ids)):
            raise ValueError("dependency task IDs must be unique")
        if dependency_task_ids and len(dependency_task_ids) != len(task.inputs):
            raise ValueError("dependency task IDs must align with frozen task inputs")
        if task.task_id in dependency_task_ids:
            raise ValueError("task cannot depend on itself")
        if packing_level <= 0:
            raise ValueError("packing level must be positive")
        spec_json = task.model_dump_json()
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("task run is not registered")
            existing = connection.execute(
                """
                SELECT task_specs.spec_json, job_nodes.run_id, job_nodes.executor_key
                FROM task_specs JOIN job_nodes USING(task_id)
                WHERE task_id = ?
                """,
                (task.task_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["spec_json"] != spec_json
                    or existing["run_id"] != run_id
                    or existing["executor_key"] != executor_key
                ):
                    raise ValueError("task ID is already bound to a different immutable task")
                latest = connection.execute(
                    """
                    SELECT attempt_id FROM attempts
                    WHERE task_id = ? ORDER BY attempt_number DESC LIMIT 1
                    """,
                    (task.task_id,),
                ).fetchone()
                if latest is None:
                    raise RuntimeError("registered task has no attempt")
                return str(latest["attempt_id"])
            connection.execute(
                """
                INSERT INTO job_nodes(
                    task_id, run_id, phase, executor_key, output_artifact_type,
                    scientific_identity_json, resource_request_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    run_id,
                    task.phase,
                    executor_key,
                    task.output_artifact_type,
                    task.identity.model_dump_json(),
                    task.resource_request.model_dump_json(),
                ),
            )
            connection.execute(
                "INSERT INTO task_specs(task_id, spec_json) VALUES (?, ?)",
                (task.task_id, spec_json),
            )
            for ordinal, dependency_id in enumerate(dependency_task_ids):
                completed = connection.execute(
                    """
                    SELECT artifact_json FROM attempts
                    WHERE task_id = ? AND state = ?
                    ORDER BY attempt_number DESC LIMIT 1
                    """,
                    (dependency_id, AttemptState.COMPLETED.value),
                ).fetchone()
                if completed is None or completed["artifact_json"] is None:
                    raise ValueError("task dependency is not completed with a verified artifact")
                dependency_artifact = ArtifactRef.model_validate_json(
                    completed["artifact_json"]
                )
                if dependency_artifact != task.inputs[ordinal]:
                    raise ValueError("task input does not match its completed dependency artifact")
                connection.execute(
                    """
                    INSERT INTO dependencies(task_id, dependency_task_id, input_ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (task.task_id, dependency_id, ordinal),
                )
            attempt_id = f"{task.task_id}-attempt-1"
            connection.execute(
                """
                INSERT INTO attempts(
                    attempt_id, task_id, attempt_number, state, packing_level,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (attempt_id, task.task_id, 1, AttemptState.READY.value, packing_level, now, now),
            )
        return attempt_id

    def transition(
        self,
        attempt_id: str,
        expected_state: AttemptState,
        next_state: AttemptState,
    ) -> None:
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE attempts SET state = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ?
                """,
                (next_state.value, now, attempt_id, expected_state.value),
            )
            if cursor.rowcount != 1:
                raise StateTransitionConflict(attempt_id)

    def attempt_state(self, attempt_id: str) -> AttemptState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise ValueError("attempt is not registered")
        return AttemptState(row["state"])

    def claim_ready(self, worker_id: str, *, limit: int) -> tuple[ClaimedTask, ...]:
        _safe_identifier(worker_id, "worker_id")
        if limit <= 0:
            raise ValueError("claim limit must be positive")
        now = _timestamp(_utc_now())
        claimed: list[ClaimedTask] = []
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT attempts.attempt_id, attempts.task_id, job_nodes.run_id,
                       job_nodes.executor_key, task_specs.spec_json
                FROM attempts
                JOIN task_specs USING(task_id)
                JOIN job_nodes USING(task_id)
                WHERE attempts.state = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM dependencies
                      WHERE dependencies.task_id = attempts.task_id
                        AND NOT EXISTS (
                            SELECT 1 FROM attempts AS dependency_attempt
                            WHERE dependency_attempt.task_id = dependencies.dependency_task_id
                              AND dependency_attempt.state = 'COMPLETED'
                        )
                  )
                ORDER BY attempts.created_at_utc, attempts.attempt_id
                LIMIT ?
                """,
                (AttemptState.READY.value, limit),
            ).fetchall()
            for row in rows:
                cursor = connection.execute(
                    """
                    UPDATE attempts
                    SET state = ?, worker_id = ?, heartbeat_at_utc = ?, updated_at_utc = ?
                    WHERE attempt_id = ? AND state = ?
                    """,
                    (
                        AttemptState.RUNNING.value,
                        worker_id,
                        now,
                        now,
                        row["attempt_id"],
                        AttemptState.READY.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StateTransitionConflict(str(row["attempt_id"]))
                claimed.append(
                    ClaimedTask(
                        run_id=str(row["run_id"]),
                        attempt_id=str(row["attempt_id"]),
                        task=TaskSpec.model_validate_json(row["spec_json"]),
                        executor_key=str(row["executor_key"]),
                        worker_id=worker_id,
                    )
                )
        return tuple(claimed)

    def ready_tasks(self, run_id: str) -> tuple[QueuedTask, ...]:
        """Return dependency-ready attempts without changing their state."""
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.attempt_id, attempts.packing_level,
                       job_nodes.run_id, job_nodes.executor_key, task_specs.spec_json
                FROM attempts
                JOIN task_specs USING(task_id)
                JOIN job_nodes USING(task_id)
                WHERE job_nodes.run_id = ? AND attempts.state = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM dependencies
                      WHERE dependencies.task_id = attempts.task_id
                        AND NOT EXISTS (
                            SELECT 1 FROM attempts AS dependency_attempt
                            WHERE dependency_attempt.task_id = dependencies.dependency_task_id
                              AND dependency_attempt.state = 'COMPLETED'
                        )
                  )
                ORDER BY attempts.created_at_utc, attempts.attempt_id
                """,
                (run_id, AttemptState.READY.value),
            ).fetchall()
        return tuple(
            QueuedTask(
                run_id=str(row["run_id"]),
                attempt_id=str(row["attempt_id"]),
                task=TaskSpec.model_validate_json(row["spec_json"]),
                executor_key=str(row["executor_key"]),
                packing_level=int(row["packing_level"]),
            )
            for row in rows
        )

    def claim_attempt(
        self,
        attempt_id: str,
        worker_id: str,
        allocation: ResourceAllocation | None = None,
    ) -> ClaimedTask | None:
        """Atomically claim one previously inspected attempt by its exact ID."""
        _safe_identifier(attempt_id, "attempt_id")
        _safe_identifier(worker_id, "worker_id")
        if allocation is not None and allocation.worker_id != worker_id:
            raise ValueError("allocation worker identity does not match claim")
        now = _timestamp(_utc_now())
        allocation_json = None if allocation is None else allocation.model_dump_json()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT attempts.attempt_id, job_nodes.run_id, job_nodes.executor_key,
                       task_specs.spec_json
                FROM attempts
                JOIN task_specs USING(task_id)
                JOIN job_nodes USING(task_id)
                WHERE attempts.attempt_id = ? AND attempts.state = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM dependencies
                      WHERE dependencies.task_id = attempts.task_id
                        AND NOT EXISTS (
                            SELECT 1 FROM attempts AS dependency_attempt
                            WHERE dependency_attempt.task_id = dependencies.dependency_task_id
                              AND dependency_attempt.state = 'COMPLETED'
                        )
                  )
                """,
                (attempt_id, AttemptState.READY.value),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE attempts
                SET state = ?, worker_id = ?, allocation_json = ?,
                    heartbeat_at_utc = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ?
                """,
                (
                    AttemptState.RUNNING.value,
                    worker_id,
                    allocation_json,
                    now,
                    now,
                    attempt_id,
                    AttemptState.READY.value,
                ),
            )
            if cursor.rowcount != 1:
                return None
        return ClaimedTask(
            run_id=str(row["run_id"]),
            attempt_id=str(row["attempt_id"]),
            task=TaskSpec.model_validate_json(row["spec_json"]),
            executor_key=str(row["executor_key"]),
            worker_id=worker_id,
        )

    def run_attempt_counts(self, run_id: str) -> dict[str, int]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.state, COUNT(*) AS count
                FROM attempts JOIN job_nodes USING(task_id)
                WHERE job_nodes.run_id = ?
                  AND attempts.attempt_number = (
                      SELECT MAX(latest.attempt_number) FROM attempts AS latest
                      WHERE latest.task_id = attempts.task_id
                  )
                GROUP BY attempts.state
                """,
                (run_id,),
            ).fetchall()
        return {str(row["state"]): int(row["count"]) for row in rows}

    def completed_artifacts(self, run_id: str) -> dict[str, ArtifactRef]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.task_id, attempts.artifact_json
                FROM attempts JOIN job_nodes USING(task_id)
                WHERE job_nodes.run_id = ? AND attempts.state = ?
                  AND attempts.artifact_json IS NOT NULL
                ORDER BY attempts.task_id, attempts.attempt_number DESC
                """,
                (run_id, AttemptState.COMPLETED.value),
            ).fetchall()
        completed: dict[str, ArtifactRef] = {}
        for row in rows:
            task_id = str(row["task_id"])
            if task_id not in completed:
                completed[task_id] = ArtifactRef.model_validate_json(row["artifact_json"])
        return completed

    def latest_failed_attempts(self, run_id: str) -> tuple[FailedAttempt, ...]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.attempt_id, attempts.error_category, attempts.packing_level
                FROM attempts JOIN job_nodes USING(task_id)
                WHERE job_nodes.run_id = ?
                  AND attempts.state IN (?, ?)
                  AND attempts.attempt_number = (
                      SELECT MAX(latest.attempt_number) FROM attempts AS latest
                      WHERE latest.task_id = attempts.task_id
                  )
                ORDER BY attempts.attempt_id
                """,
                (run_id, AttemptState.FAILED.value, AttemptState.STALLED.value),
            ).fetchall()
        return tuple(
            FailedAttempt(
                attempt_id=str(row["attempt_id"]),
                error_category=str(row["error_category"] or "WORKER_DIED"),
                packing_level=int(row["packing_level"]),
            )
            for row in rows
        )

    def bind_running_process(
        self,
        attempt_id: str,
        worker_id: str,
        pid: int,
        process_started_at_utc: datetime,
        *,
        now: datetime | None = None,
    ) -> None:
        _safe_identifier(worker_id, "worker_id")
        if pid <= 0:
            raise ValueError("worker PID must be positive")
        heartbeat = _timestamp(now or _utc_now())
        started = _timestamp(process_started_at_utc)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise ValueError("attempt is not registered")
            state = AttemptState(row["state"])
            if state not in {AttemptState.READY, AttemptState.RUNNING}:
                raise StateTransitionConflict(attempt_id)
            cursor = connection.execute(
                """
                UPDATE attempts
                SET state = ?, worker_id = ?, pid = ?, process_started_at_utc = ?,
                    heartbeat_at_utc = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ?
                """,
                (
                    AttemptState.RUNNING.value,
                    worker_id,
                    pid,
                    started,
                    heartbeat,
                    heartbeat,
                    attempt_id,
                    state.value,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionConflict(attempt_id)

    def heartbeat(
        self,
        attempt_id: str,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        heartbeat = _timestamp(now or _utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE attempts SET heartbeat_at_utc = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ? AND worker_id = ?
                """,
                (heartbeat, heartbeat, attempt_id, AttemptState.RUNNING.value, worker_id),
            )
            if cursor.rowcount != 1:
                raise StateTransitionConflict(attempt_id)

    def claimed_task(self, attempt_id: str) -> ClaimedTask:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempts.state, attempts.worker_id, attempts.attempt_id,
                       job_nodes.run_id, job_nodes.executor_key, task_specs.spec_json
                FROM attempts
                JOIN task_specs USING(task_id)
                JOIN job_nodes USING(task_id)
                WHERE attempts.attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise ValueError("attempt is not registered")
        if AttemptState(row["state"]) is not AttemptState.RUNNING or row["worker_id"] is None:
            raise StateTransitionConflict(attempt_id)
        return ClaimedTask(
            run_id=str(row["run_id"]),
            attempt_id=str(row["attempt_id"]),
            task=TaskSpec.model_validate_json(row["spec_json"]),
            executor_key=str(row["executor_key"]),
            worker_id=str(row["worker_id"]),
        )

    def complete_attempt(self, attempt_id: str, artifact: ArtifactRef) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempts.state, job_nodes.output_artifact_type
                FROM attempts JOIN job_nodes USING(task_id)
                WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        if row is None or AttemptState(row["state"]) is not AttemptState.RUNNING:
            raise StateTransitionConflict(attempt_id)
        if artifact.artifact_type != row["output_artifact_type"]:
            raise ValueError("artifact identity does not match the task output type")
        if self._artifact_verifier is None or not self._artifact_verifier(artifact):
            raise ValueError("artifact verification failed due to hash drift")
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE attempts
                SET state = ?, artifact_json = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ?
                """,
                (
                    AttemptState.COMPLETED.value,
                    artifact.model_dump_json(),
                    now,
                    attempt_id,
                    AttemptState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionConflict(attempt_id)

    def fail_attempt(self, attempt_id: str, error_category: str) -> None:
        if not error_category.strip():
            raise ValueError("error category must not be blank")
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE attempts
                SET state = ?, error_category = ?, updated_at_utc = ?
                WHERE attempt_id = ? AND state = ?
                """,
                (
                    AttemptState.FAILED.value,
                    error_category,
                    now,
                    attempt_id,
                    AttemptState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise StateTransitionConflict(attempt_id)

    def attempt_error(self, attempt_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT error_category FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise ValueError("attempt is not registered")
        return cast(str | None, row["error_category"])

    def retry_disposition(self, error_category: str) -> RetryDisposition:
        return RETRY_POLICY.get(error_category, RetryDisposition.TERMINAL)

    def retry_attempt(
        self,
        attempt_id: str,
        error_category: str,
        *,
        lower_packing_applied: bool = False,
    ) -> str | None:
        disposition = self.retry_disposition(error_category)
        if disposition is RetryDisposition.TERMINAL:
            return None
        if (
            disposition is RetryDisposition.RETRY_ONCE_WITH_LOWER_PACKING
            and not lower_packing_applied
        ):
            return None
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT task_id, attempt_number, state, error_category, packing_level
                FROM attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise ValueError("attempt is not registered")
            if AttemptState(row["state"]) not in {AttemptState.FAILED, AttemptState.STALLED}:
                raise StateTransitionConflict(attempt_id)
            if row["error_category"] not in {None, error_category}:
                raise ValueError("retry category does not match the failed attempt")
            attempt_number = int(row["attempt_number"])
            if attempt_number >= 2:
                return None
            task_id = str(row["task_id"])
            next_attempt = f"{task_id}-attempt-{attempt_number + 1}"
            packing_level = int(row["packing_level"])
            if disposition is RetryDisposition.RETRY_ONCE_WITH_LOWER_PACKING:
                packing_level = max(1, packing_level - 1)
            connection.execute(
                """
                INSERT INTO attempts(
                    attempt_id, task_id, attempt_number, state, packing_level,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_attempt,
                    task_id,
                    attempt_number + 1,
                    AttemptState.READY.value,
                    packing_level,
                    now,
                    now,
                ),
            )
        return next_attempt

    def reconcile_processes(
        self,
        probe: ProcessProbe,
        *,
        now: datetime | None = None,
        heartbeat_timeout_seconds: float = 10.0,
    ) -> ReconciliationResult:
        checked_at = _require_utc(now or _utc_now())
        if not math.isfinite(heartbeat_timeout_seconds) or heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat timeout must be finite and positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.attempt_id, attempts.task_id, attempts.pid,
                       attempts.process_started_at_utc, attempts.heartbeat_at_utc,
                       job_nodes.run_id
                FROM attempts JOIN job_nodes USING(task_id)
                WHERE attempts.state = ?
                ORDER BY attempts.task_id
                """,
                (AttemptState.RUNNING.value,),
            ).fetchall()
        live: list[str] = []
        stalled: list[str] = []
        for row in rows:
            pid = row["pid"]
            started_text = row["process_started_at_utc"]
            heartbeat_text = row["heartbeat_at_utc"]
            process = probe.inspect(int(pid)) if isinstance(pid, int) and pid > 0 else None
            heartbeat_fresh = False
            if isinstance(heartbeat_text, str):
                heartbeat_age = (checked_at - _parse_timestamp(heartbeat_text)).total_seconds()
                heartbeat_fresh = 0.0 <= heartbeat_age <= heartbeat_timeout_seconds
            expected_started = (
                _parse_timestamp(started_text) if isinstance(started_text, str) else None
            )
            identity_matches = (
                process is not None
                and expected_started is not None
                and process.pid == pid
                and process.process_started_at_utc == expected_started
                and process.run_id == row["run_id"]
                and process.task_id == row["task_id"]
            )
            task_id = str(row["task_id"])
            if heartbeat_fresh and identity_matches:
                live.append(task_id)
                continue
            try:
                self.transition(
                    str(row["attempt_id"]),
                    AttemptState.RUNNING,
                    AttemptState.STALLED,
                )
            except StateTransitionConflict:
                continue
            stalled.append(task_id)
        return ReconciliationResult(tuple(live), tuple(stalled))

    def record_progress(self, attempt_id: str, progress: object) -> None:
        payload = _json_payload(progress)
        now = _timestamp(_utc_now())
        with self._transaction() as connection:
            state = connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if state is None or AttemptState(state["state"]) is not AttemptState.RUNNING:
                raise StateTransitionConflict(attempt_id)
            connection.execute(
                """
                INSERT INTO progress_events(attempt_id, recorded_at_utc, payload_json)
                VALUES (?, ?, ?)
                """,
                (attempt_id, now, payload),
            )

    def progress_events(self, attempt_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM progress_events
                WHERE attempt_id = ? ORDER BY event_id
                """,
                (attempt_id,),
            ).fetchall()
        return tuple(cast(dict[str, Any], json.loads(row["payload_json"])) for row in rows)

    def record_resource_sample(
        self,
        run_id: str,
        sample: object,
        *,
        attempt_id: str | None = None,
    ) -> None:
        payload = _json_payload(sample)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO resource_samples(run_id, attempt_id, sampled_at_utc, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, attempt_id, _timestamp(_utc_now()), payload),
            )

    def add_alert(
        self,
        run_id: str,
        category: str,
        message: str,
        *,
        attempt_id: str | None = None,
    ) -> None:
        if not category.strip() or not message.strip():
            raise ValueError("alert category and message must not be blank")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO alerts(run_id, attempt_id, created_at_utc, category, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, attempt_id, _timestamp(_utc_now()), category, message),
            )
