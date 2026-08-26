from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal


class RuntimeAuthorizationRequired(RuntimeError):
    """Raised when a 24-hour local run lacks explicit user authorization."""


class InfeasibleRuntimeError(RuntimeError):
    """Raised when the unchanged workload is estimated to exceed 120 hours."""


@dataclass(frozen=True, slots=True)
class RemainingTask:
    task_id: str
    lane_id: str
    remaining_work_units: float
    seconds_per_work_unit: float | None
    fixed_overhead_seconds: float
    dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.lane_id.strip():
            raise ValueError("ETA task and lane IDs must not be blank")
        if not math.isfinite(self.remaining_work_units) or self.remaining_work_units < 0:
            raise ValueError("remaining work units must be finite and non-negative")
        if self.seconds_per_work_unit is not None and (
            not math.isfinite(self.seconds_per_work_unit) or self.seconds_per_work_unit <= 0
        ):
            raise ValueError("ETA work rate must be finite and positive")
        if not math.isfinite(self.fixed_overhead_seconds) or self.fixed_overhead_seconds < 0:
            raise ValueError("ETA fixed overhead must be finite and non-negative")
        if len(self.dependency_ids) != len(set(self.dependency_ids)):
            raise ValueError("ETA dependency IDs must be unique")

    @property
    def duration_seconds(self) -> float | None:
        if self.seconds_per_work_unit is None:
            return None
        return self.remaining_work_units * self.seconds_per_work_unit + self.fixed_overhead_seconds


@dataclass(frozen=True, slots=True)
class EtaEstimate:
    status: Literal["CALIBRATING", "READY"]
    remaining_seconds: float | None
    expected_completion_utc: datetime | None
    lower_seconds: float | None
    upper_seconds: float | None
    exceeds_24_hours: bool

    @property
    def infeasible_over_120_hours(self) -> bool:
        return self.remaining_seconds is not None and self.remaining_seconds > 120 * 3600


@dataclass(frozen=True, slots=True)
class TimingRate:
    world_id: str
    model_id: str
    seconds_per_work_unit_ewma: float
    observation_count: int

    def __post_init__(self) -> None:
        if not self.world_id.strip() or not self.model_id.strip():
            raise ValueError("timing rate world and model IDs must not be blank")
        if (
            not math.isfinite(self.seconds_per_work_unit_ewma)
            or self.seconds_per_work_unit_ewma <= 0
            or self.observation_count <= 0
        ):
            raise ValueError("timing rate values must be positive and finite")


@dataclass(frozen=True, slots=True)
class EwmaRateBook:
    rates: tuple[TimingRate, ...] = ()
    alpha: float = 0.3

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("EWMA alpha must be within (0, 1]")
        keys = tuple((rate.world_id, rate.model_id) for rate in self.rates)
        if len(keys) != len(set(keys)):
            raise ValueError("EWMA timing keys must be unique")

    def updated(
        self,
        world_id: str,
        model_id: str,
        *,
        elapsed_seconds: float,
        completed_work_units: float,
    ) -> EwmaRateBook:
        if (
            not math.isfinite(elapsed_seconds)
            or not math.isfinite(completed_work_units)
            or elapsed_seconds <= 0
            or completed_work_units <= 0
        ):
            raise ValueError("EWMA observation values must be finite and positive")
        observed = elapsed_seconds / completed_work_units
        by_key = {(rate.world_id, rate.model_id): rate for rate in self.rates}
        previous = by_key.get((world_id, model_id))
        value = (
            observed
            if previous is None
            else self.alpha * observed + (1.0 - self.alpha) * previous.seconds_per_work_unit_ewma
        )
        by_key[(world_id, model_id)] = TimingRate(
            world_id=world_id,
            model_id=model_id,
            seconds_per_work_unit_ewma=value,
            observation_count=1 if previous is None else previous.observation_count + 1,
        )
        return EwmaRateBook(
            rates=tuple(by_key[key] for key in sorted(by_key)),
            alpha=self.alpha,
        )

    def rate(self, world_id: str, model_id: str) -> float | None:
        return next(
            (
                rate.seconds_per_work_unit_ewma
                for rate in self.rates
                if rate.world_id == world_id and rate.model_id == model_id
            ),
            None,
        )


def _validated_now(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("ETA timestamp must be timezone-aware UTC")
    return value


def estimate_run_eta(
    tasks: tuple[RemainingTask, ...],
    *,
    now: datetime | None = None,
) -> EtaEstimate:
    estimated_at = _validated_now(now)
    task_ids = tuple(task.task_id for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("ETA task IDs must be unique")
    by_id = {task.task_id: task for task in tasks}
    unknown = {
        dependency
        for task in tasks
        for dependency in task.dependency_ids
        if dependency not in by_id
    }
    if unknown:
        raise ValueError("ETA graph contains unknown dependencies")
    if any(task.duration_seconds is None for task in tasks):
        return EtaEstimate(
            status="CALIBRATING",
            remaining_seconds=None,
            expected_completion_utc=None,
            lower_seconds=None,
            upper_seconds=None,
            exceeds_24_hours=False,
        )

    visiting: set[str] = set()
    memo: dict[str, float] = {}

    def dependency_path(task_id: str) -> float:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            raise ValueError("ETA dependency graph contains a cycle")
        visiting.add(task_id)
        task = by_id[task_id]
        dependency_seconds = max(
            (dependency_path(dependency) for dependency in task.dependency_ids),
            default=0.0,
        )
        duration = task.duration_seconds
        if duration is None:
            raise RuntimeError("calibrating ETA task reached ready calculation")
        total = dependency_seconds + duration
        visiting.remove(task_id)
        memo[task_id] = total
        return total

    critical_path = max((dependency_path(task.task_id) for task in tasks), default=0.0)
    lane_loads: dict[str, float] = {}
    for task in tasks:
        duration = task.duration_seconds
        if duration is None:
            raise RuntimeError("calibrating ETA task reached lane calculation")
        lane_loads[task.lane_id] = lane_loads.get(task.lane_id, 0.0) + duration
    remaining = max(critical_path, max(lane_loads.values(), default=0.0))
    return EtaEstimate(
        status="READY",
        remaining_seconds=remaining,
        expected_completion_utc=estimated_at + timedelta(seconds=remaining),
        lower_seconds=remaining * 0.8,
        upper_seconds=remaining * 1.25,
        exceeds_24_hours=remaining > 24 * 3600,
    )


def enforce_time_gate(
    estimate: EtaEstimate,
    *,
    authorized_over_24_hours: bool,
) -> None:
    if estimate.status != "READY" or estimate.remaining_seconds is None:
        raise RuntimeAuthorizationRequired("runtime is still calibrating")
    if estimate.remaining_seconds > 120 * 3600:
        raise InfeasibleRuntimeError(
            "unchanged workload exceeds the 120-hour hardware feasibility ceiling"
        )
    if estimate.exceeds_24_hours and not authorized_over_24_hours:
        raise RuntimeAuthorizationRequired(
            "runtime exceeds 24 hours and requires explicit user authorization"
        )
