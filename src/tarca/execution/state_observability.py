from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, cast

from tarca.execution.state_models import (
    AttemptState,
    StateTransitionConflict,
)
from tarca.execution.state_models import (
    json_payload as _json_payload,
)
from tarca.execution.state_models import (
    parse_timestamp as _parse_timestamp,
)
from tarca.execution.state_models import (
    resource_sample as _resource_sample,
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
from tarca.execution.telemetry import ResourceSample


class StateObservabilityMixin:
    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

    def record_progress(
        self,
        attempt_id: str,
        progress: object,
        *,
        now: datetime | None = None,
    ) -> None:
        payload = _json_payload(progress)
        recorded_at = _timestamp(now or _utc_now())
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
                (attempt_id, recorded_at, payload),
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
        sample: ResourceSample,
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
                (run_id, attempt_id, _timestamp(sample.sampled_at_utc), payload),
            )

    def resource_samples(
        self,
        run_id: str,
        *,
        attempt_id: str | None,
    ) -> tuple[ResourceSample, ...]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            if attempt_id is None:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM resource_samples
                    WHERE run_id = ? AND attempt_id IS NULL ORDER BY sample_id
                    """,
                    (run_id,),
                ).fetchall()
            else:
                _safe_identifier(attempt_id, "attempt_id")
                rows = connection.execute(
                    """
                    SELECT payload_json FROM resource_samples
                    WHERE run_id = ? AND attempt_id = ? ORDER BY sample_id
                    """,
                    (run_id, attempt_id),
                ).fetchall()
        return tuple(_resource_sample(str(row["payload_json"])) for row in rows)

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

    def add_alert_once(
        self,
        run_id: str,
        category: str,
        message: str,
        *,
        attempt_id: str | None = None,
    ) -> bool:
        if not category.strip() or not message.strip():
            raise ValueError("alert category and message must not be blank")
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT alert_id FROM alerts
                WHERE run_id = ? AND attempt_id IS ? AND category = ? AND message = ?
                LIMIT 1
                """,
                (run_id, attempt_id, category, message),
            ).fetchone()
            if existing is not None:
                return False
            connection.execute(
                """
                INSERT INTO alerts(run_id, attempt_id, created_at_utc, category, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, attempt_id, _timestamp(_utc_now()), category, message),
            )
        return True

    def alerts(self, run_id: str) -> tuple[dict[str, Any], ...]:
        _safe_identifier(run_id, "run_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT alert_id, attempt_id, category, message, created_at_utc
                FROM alerts WHERE run_id = ? ORDER BY alert_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            {
                "alert_id": int(row["alert_id"]),
                "attempt_id": (str(row["attempt_id"]) if row["attempt_id"] is not None else None),
                "category": str(row["category"]),
                "message": str(row["message"]),
                "created_at_utc": _parse_timestamp(str(row["created_at_utc"])),
            }
            for row in rows
        )
