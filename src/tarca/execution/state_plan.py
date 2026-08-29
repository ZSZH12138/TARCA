from __future__ import annotations

import re
import sqlite3
from contextlib import AbstractContextManager

from tarca.contracts import ArtifactRef
from tarca.execution.contracts import ResourceAllocation, RunPlanNode, TaskSpec
from tarca.execution.state_models import (
    AttemptState,
    ClaimedTask,
    FailedAttempt,
    QueuedTask,
    RunningAttempt,
    StateTransitionConflict,
)
from tarca.execution.state_models import (
    parse_timestamp as _parse_timestamp,
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


class StatePlanMixin:
    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

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

    def register_run_plan(self, run_id: str, nodes: tuple[RunPlanNode, ...]) -> None:
        _safe_identifier(run_id, "run_id")
        if not nodes:
            raise ValueError("run plan must contain at least one node")
        task_ids = tuple(node.task_id for node in nodes)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("run plan contains duplicate task IDs")
        known_ids = set(task_ids)
        if any(
            dependency not in known_ids for node in nodes for dependency in node.dependency_task_ids
        ):
            raise ValueError("run plan contains an unknown dependency")
        dependencies = {node.task_id: node.dependency_task_ids for node in nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("run plan contains a dependency cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)

        serialized = tuple(node.model_dump_json() for node in nodes)
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("run plan run is not registered")
            existing_rows = connection.execute(
                "SELECT node_json FROM run_plan_nodes WHERE run_id = ? ORDER BY ordinal",
                (run_id,),
            ).fetchall()
            if existing_rows:
                existing = tuple(str(row["node_json"]) for row in existing_rows)
                if existing != serialized:
                    raise ValueError("run ID is already bound to a different immutable plan")
                return
            connection.executemany(
                """
                INSERT INTO run_plan_nodes(run_id, task_id, ordinal, node_json)
                VALUES (?, ?, ?, ?)
                """,
                tuple(
                    (run_id, node.task_id, ordinal, serialized[ordinal])
                    for ordinal, node in enumerate(nodes)
                ),
            )

    def run_plan_nodes(self, run_id: str) -> tuple[RunPlanNode, ...]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT node_json FROM run_plan_nodes WHERE run_id = ? ORDER BY ordinal",
                (run_id,),
            ).fetchall()
        return tuple(RunPlanNode.model_validate_json(row["node_json"]) for row in rows)

    def planned_task_count(self, run_id: str) -> int:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM run_plan_nodes WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("run plan count query returned no row")
        return int(row[0])

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
                dependency_artifact = ArtifactRef.model_validate_json(completed["artifact_json"])
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

    def running_attempts(self, run_id: str) -> tuple[RunningAttempt, ...]:
        """Return latest running attempts with their committed allocations."""
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT attempts.attempt_id, attempts.allocation_json, attempts.pid,
                       attempts.process_started_at_utc, attempts.heartbeat_at_utc,
                       job_nodes.run_id, task_specs.spec_json
                FROM attempts
                JOIN task_specs USING(task_id)
                JOIN job_nodes USING(task_id)
                WHERE job_nodes.run_id = ? AND attempts.state = ?
                  AND attempts.attempt_number = (
                      SELECT MAX(latest.attempt_number) FROM attempts AS latest
                      WHERE latest.task_id = attempts.task_id
                  )
                ORDER BY attempts.created_at_utc, attempts.attempt_id
                """,
                (run_id, AttemptState.RUNNING.value),
            ).fetchall()
        running: list[RunningAttempt] = []
        for row in rows:
            allocation_json = row["allocation_json"]
            if not isinstance(allocation_json, str):
                raise RuntimeError("running attempt has no committed resource allocation")
            started = row["process_started_at_utc"]
            heartbeat = row["heartbeat_at_utc"]
            pid = row["pid"]
            running.append(
                RunningAttempt(
                    run_id=str(row["run_id"]),
                    attempt_id=str(row["attempt_id"]),
                    task=TaskSpec.model_validate_json(row["spec_json"]),
                    allocation=ResourceAllocation.model_validate_json(allocation_json),
                    pid=int(pid) if isinstance(pid, int) and pid > 0 else None,
                    process_started_at_utc=(
                        _parse_timestamp(started) if isinstance(started, str) else None
                    ),
                    heartbeat_at_utc=(
                        _parse_timestamp(heartbeat) if isinstance(heartbeat, str) else None
                    ),
                )
            )
        return tuple(running)

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
