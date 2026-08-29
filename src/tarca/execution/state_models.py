from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from pydantic import BaseModel

from tarca.contracts import ArtifactRef, canonical_json_bytes
from tarca.execution.contracts import ResourceAllocation, TaskSpec
from tarca.execution.telemetry import GpuSample, ResourceSample

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
class RunningAttempt:
    run_id: str
    attempt_id: str
    task: TaskSpec
    allocation: ResourceAllocation
    pid: int | None
    process_started_at_utc: datetime | None
    heartbeat_at_utc: datetime | None


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
        require_utc(self.process_started_at_utc)
        if not self.run_id.strip() or not self.task_id.strip():
            raise ValueError("process run and task identities must not be blank")


class ProcessProbe(Protocol):
    def inspect(self, pid: int) -> ProcessIdentity | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    live_task_ids: tuple[str, ...]
    stalled_task_ids: tuple[str, ...]


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("execution timestamps must be timezone-aware UTC")
    return value


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime) -> str:
    return require_utc(value).isoformat(timespec="microseconds")


def parse_timestamp(value: str) -> datetime:
    return require_utc(datetime.fromisoformat(value))


def safe_identifier(value: str, label: str) -> str:
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


def json_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        serializable: object = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        serializable = asdict(value)
    elif isinstance(value, Mapping):
        serializable = dict(value)
    else:
        raise TypeError("state event payload must be a model, dataclass, or mapping")
    return canonical_json_bytes(json_safe(serializable)).decode("utf-8")


def json_safe(value: object) -> object:
    if isinstance(value, datetime):
        result: object = timestamp(value)
    elif isinstance(value, StrEnum):
        result = value.value
    elif isinstance(value, Path):
        result = value.as_posix()
    elif is_dataclass(value) and not isinstance(value, type):
        result = json_safe(asdict(value))
    elif isinstance(value, Mapping):
        result = {str(key): json_safe(item) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        result = [json_safe(item) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
    else:
        raise TypeError(
            f"state event payload contains unsupported type: {type(value).__name__}"
        )
    return result


def resource_sample(payload_json: str) -> ResourceSample:
    payload = cast(dict[str, Any], json.loads(payload_json))
    gpu_samples = tuple(
        GpuSample(
            gpu_id=int(item["gpu_id"]),
            utilization_percent=float(item["utilization_percent"]),
            memory_used_bytes=int(item["memory_used_bytes"]),
            memory_total_bytes=int(item["memory_total_bytes"]),
            power_watts=float(item["power_watts"]),
            temperature_celsius=float(item["temperature_celsius"]),
            compute_pids=tuple(int(pid) for pid in item["compute_pids"]),
        )
        for item in payload["gpu_samples"]
    )
    process_pss = payload["process_pss_bytes"]
    return ResourceSample(
        sampled_at_utc=parse_timestamp(str(payload["sampled_at_utc"])),
        host_cpu_percent=float(payload["host_cpu_percent"]),
        effective_busy_cores=float(payload["effective_busy_cores"]),
        process_rss_bytes=int(payload["process_rss_bytes"]),
        process_pss_bytes=None if process_pss is None else int(process_pss),
        process_affinity_cpu_ids=tuple(
            int(cpu_id) for cpu_id in payload["process_affinity_cpu_ids"]
        ),
        host_memory_used_bytes=int(payload["host_memory_used_bytes"]),
        gpu_samples=gpu_samples,
        disk_read_bytes_per_second=float(payload["disk_read_bytes_per_second"]),
        disk_write_bytes_per_second=float(payload["disk_write_bytes_per_second"]),
    )
