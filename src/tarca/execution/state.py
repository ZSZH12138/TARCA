from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tarca.execution.state_attempts import StateAttemptsMixin
from tarca.execution.state_models import (
    RETRY_POLICY,
    ArtifactVerifier,
    AttemptState,
    ClaimedTask,
    FailedAttempt,
    ProcessIdentity,
    ProcessProbe,
    QueuedTask,
    ReconciliationResult,
    RetryDisposition,
    RunningAttempt,
    StateTransitionConflict,
)
from tarca.execution.state_observability import StateObservabilityMixin
from tarca.execution.state_plan import StatePlanMixin
from tarca.execution.state_schema import SCHEMA

__all__ = [
    "RETRY_POLICY",
    "AttemptState",
    "ClaimedTask",
    "ExecutionStateStore",
    "FailedAttempt",
    "ProcessIdentity",
    "ProcessProbe",
    "QueuedTask",
    "ReconciliationResult",
    "RetryDisposition",
    "RunningAttempt",
    "StateTransitionConflict",
]


class ExecutionStateStore(StatePlanMixin, StateAttemptsMixin, StateObservabilityMixin):
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
            connection.executescript(SCHEMA)

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
