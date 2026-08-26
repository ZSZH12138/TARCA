from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tarca.execution.eta import (
    InfeasibleRuntimeError,
    RemainingTask,
    RuntimeAuthorizationRequired,
    enforce_time_gate,
    estimate_run_eta,
)


def _task(
    task_id: str,
    lane: str,
    hours: float | None,
    dependencies: tuple[str, ...] = (),
) -> RemainingTask:
    return RemainingTask(
        task_id=task_id,
        lane_id=lane,
        remaining_work_units=1.0,
        seconds_per_work_unit=None if hours is None else hours * 3600.0,
        fixed_overhead_seconds=0.0,
        dependency_ids=dependencies,
    )


def test_eta_uses_two_gpu_critical_path() -> None:
    now = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    estimate = estimate_run_eta(
        (_task("gpu-0", "cuda:0", 5.0), _task("gpu-1", "cuda:1", 5.0)),
        now=now,
    )
    assert estimate.remaining_seconds == pytest.approx(5 * 3600.0)
    assert estimate.expected_completion_utc is not None
    assert estimate.expected_completion_utc.timestamp() == pytest.approx(now.timestamp() + 5 * 3600)


def test_eta_uses_longest_dependency_path_and_calibrates_unknown_rates() -> None:
    estimate = estimate_run_eta(
        (
            _task("first", "cpu", 2.0),
            _task("second", "cuda:0", 4.0, dependencies=("first",)),
        )
    )
    assert estimate.remaining_seconds == pytest.approx(6 * 3600.0)
    calibrating = estimate_run_eta((_task("unknown", "cuda:0", None),))
    assert calibrating.status == "CALIBRATING"
    assert calibrating.remaining_seconds is None


def test_time_gate_requires_authorization_and_never_hides_infeasibility() -> None:
    needs_authorization = estimate_run_eta((_task("long", "cuda:0", 25.0),))
    assert needs_authorization.exceeds_24_hours
    with pytest.raises(RuntimeAuthorizationRequired):
        enforce_time_gate(needs_authorization, authorized_over_24_hours=False)
    enforce_time_gate(needs_authorization, authorized_over_24_hours=True)

    infeasible = estimate_run_eta((_task("too-long", "cuda:0", 121.0),))
    with pytest.raises(InfeasibleRuntimeError):
        enforce_time_gate(infeasible, authorized_over_24_hours=True)
