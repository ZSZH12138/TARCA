from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import MappingProxyType

from tarca.execution.state import ExecutionStateStore
from tarca.execution.telemetry import (
    TelemetryPolicy,
    TelemetryProbe,
    collect_resource_sample,
    monitor_overhead_alerts,
)


class RuntimeSupervisor:
    """Best-effort runtime sampler with no access to scientific evidence."""

    def __init__(
        self,
        store: ExecutionStateStore,
        probe: TelemetryProbe,
        policy: TelemetryPolicy | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._probe = probe
        self._policy = policy or TelemetryPolicy()
        self._clock = clock
        self._last_sampled_by_run: Mapping[str, float] = MappingProxyType({})

    def sample_if_due(self, run_id: str, supervisor_pid: int) -> bool:
        now = self._clock()
        previous = self._last_sampled_by_run.get(run_id)
        if previous is not None and now - previous < self._policy.sample_interval_seconds:
            return False
        self._last_sampled_by_run = MappingProxyType({**self._last_sampled_by_run, run_id: now})

        try:
            run_sample = collect_resource_sample(supervisor_pid, self._probe)
            self._store.record_resource_sample(run_id, run_sample)
        except Exception as error:
            self._record_unavailable(run_id, error)
            return False

        for category in monitor_overhead_alerts(run_sample, self._policy):
            self._store.add_alert_once(
                run_id,
                category,
                "Runtime monitor exceeded its reserved resource budget",
            )

        for attempt in self._store.running_attempts(run_id):
            if attempt.pid is None:
                continue
            try:
                sample = collect_resource_sample(attempt.pid, self._probe)
                self._store.record_resource_sample(
                    run_id,
                    sample,
                    attempt_id=attempt.attempt_id,
                )
            except Exception as error:
                self._record_unavailable(run_id, error, attempt_id=attempt.attempt_id)
        return True

    def _record_unavailable(
        self,
        run_id: str,
        error: Exception,
        *,
        attempt_id: str | None = None,
    ) -> None:
        self._store.add_alert_once(
            run_id,
            "TELEMETRY_UNAVAILABLE",
            f"Runtime telemetry probe failed ({type(error).__name__})",
            attempt_id=attempt_id,
        )
