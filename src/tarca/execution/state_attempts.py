from __future__ import annotations

import math
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime
from typing import cast

from tarca.contracts import ArtifactRef
from tarca.execution.contracts import TaskSpec
from tarca.execution.state_models import (
    RETRY_POLICY,
    ArtifactVerifier,
    AttemptState,
    ClaimedTask,
    ProcessProbe,
    ReconciliationResult,
    RetryDisposition,
    StateTransitionConflict,
)
from tarca.execution.state_models import (
    parse_timestamp as _parse_timestamp,
)
from tarca.execution.state_models import (
    require_utc as _require_utc,
)
from tarca.execution.state_models import (
    safe_identifier as _safe_identifier,
)
from tarca.execution.state_models import (
    timestamp as _timestamp,
)
from tarca.execution.state_models import (
    utc_now as _utc_now,
)


class StateAttemptsMixin:
    _artifact_verifier: ArtifactVerifier | None

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

    def transition(
        self,
        attempt_id: str,
        expected: AttemptState,
        target: AttemptState,
    ) -> None:
        raise NotImplementedError

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
