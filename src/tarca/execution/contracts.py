from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from tarca.contracts import ArtifactRef, Sha256Hash, StrictContractModel, UtcDatetime


def _safe_logical_value(value: str) -> str:
    if (
        not value.strip()
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError("execution identifier must be a nonblank logical value")
    return value


class ScientificIdentity(StrictContractModel):
    protocol_id: str
    experiment_id: str
    task_id: str
    model_id: str
    data_id: str
    seed: int = Field(ge=0)

    @field_validator("protocol_id", "experiment_id", "task_id", "model_id", "data_id")
    @classmethod
    def _identifiers_are_safe(cls, value: str) -> str:
        return _safe_logical_value(value)


class ResourceRequest(StrictContractModel):
    cpu_threads: int = Field(gt=0)
    gpu_count: int = Field(ge=0)
    gpu_memory_gib: float = Field(ge=0.0)
    host_memory_gib: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _resources_are_coherent(self) -> Self:
        if not math.isfinite(self.gpu_memory_gib) or not math.isfinite(self.host_memory_gib):
            raise ValueError("resource memory requests must be finite")
        if (self.gpu_count == 0) != (self.gpu_memory_gib == 0.0):
            raise ValueError("GPU count and memory request must both be zero or positive")
        return self


class TaskSpec(StrictContractModel):
    identity: ScientificIdentity
    phase: str
    inputs: tuple[ArtifactRef, ...]
    output_artifact_type: str
    resource_request: ResourceRequest

    @field_validator("phase", "output_artifact_type")
    @classmethod
    def _task_text_is_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("inputs")
    @classmethod
    def _inputs_are_unique(cls, value: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        identities = tuple(item.identity_key() for item in value)
        if len(identities) != len(set(identities)):
            raise ValueError("task input artifacts must be unique")
        return value

    @property
    def task_id(self) -> str:
        return self.identity.task_id


class TaskManifest(StrictContractModel):
    manifest_id: str
    tasks: tuple[TaskSpec, ...]
    completed_task_policy: Literal["NEVER_RERUN"]

    @field_validator("manifest_id")
    @classmethod
    def _manifest_id_is_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("tasks")
    @classmethod
    def _tasks_are_unique(cls, value: tuple[TaskSpec, ...]) -> tuple[TaskSpec, ...]:
        task_ids = tuple(task.task_id for task in value)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task manifest task IDs must be unique")
        return value


class ResourceAllocation(StrictContractModel):
    cpu_threads: int = Field(gt=0)
    gpu_ids: tuple[int, ...]
    host_memory_gib_limit: float = Field(gt=0.0)
    worker_id: str

    @field_validator("gpu_ids")
    @classmethod
    def _gpu_ids_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(item) is not int or item < 0 for item in value):
            raise ValueError("GPU IDs must be non-negative integers")
        if len(value) != len(set(value)):
            raise ValueError("GPU IDs must be unique")
        return value

    @field_validator("worker_id")
    @classmethod
    def _worker_id_is_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("host_memory_gib_limit")
    @classmethod
    def _memory_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("host memory limit must be finite")
        return value


class PlannedTask(StrictContractModel):
    task_id: str
    attempt_id: str
    executor_key: str
    allocation: ResourceAllocation
    input_refs: tuple[ArtifactRef, ...]
    expected_output_artifact_type: str

    @field_validator("task_id", "attempt_id", "expected_output_artifact_type")
    @classmethod
    def _planned_values_are_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("executor_key")
    @classmethod
    def _executor_key_is_allowlist_shaped(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value) is None:
            raise ValueError("executor key must be a safe registry identifier")
        return value

    @field_validator("input_refs")
    @classmethod
    def _input_refs_are_unique(
        cls, value: tuple[ArtifactRef, ...]
    ) -> tuple[ArtifactRef, ...]:
        identities = tuple(item.identity_key() for item in value)
        if len(identities) != len(set(identities)):
            raise ValueError("planned task input artifacts must be unique")
        return value


class ExecutionPlan(StrictContractModel):
    plan_id: str
    task_manifest_id: str
    backend_id: str
    planned_tasks: tuple[PlannedTask, ...]
    max_concurrency: int = Field(gt=0)
    resource_snapshot_hash: Sha256Hash
    created_at: UtcDatetime

    @field_validator("plan_id", "task_manifest_id", "backend_id")
    @classmethod
    def _plan_values_are_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("planned_tasks")
    @classmethod
    def _planned_tasks_are_unique(
        cls, value: tuple[PlannedTask, ...]
    ) -> tuple[PlannedTask, ...]:
        task_ids = tuple(task.task_id for task in value)
        attempt_ids = tuple(task.attempt_id for task in value)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("execution plan task IDs must be unique")
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("execution plan attempt IDs must be unique")
        return value


class ExecutionContext(StrictContractModel):
    run_id: str
    task_id: str
    attempt_id: str
    runtime_identity: str
    worker_identity: str

    @field_validator("run_id", "task_id", "attempt_id", "runtime_identity", "worker_identity")
    @classmethod
    def _context_values_are_safe(cls, value: str) -> str:
        return _safe_logical_value(value)


class TaskState(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    PUBLISHING = "PUBLISHING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskResult(StrictContractModel):
    task_id: str
    attempt_id: str
    state: TaskState
    artifact: ArtifactRef | None

    @field_validator("task_id", "attempt_id")
    @classmethod
    def _result_values_are_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @model_validator(mode="after")
    def _completion_has_a_typed_artifact(self) -> Self:
        if self.state is TaskState.COMPLETED and self.artifact is None:
            raise ValueError("COMPLETED task result requires a verified ArtifactRef")
        if self.state is not TaskState.COMPLETED and self.artifact is not None:
            raise ValueError("only COMPLETED task results may bind an ArtifactRef")
        return self


class MonitoringSnapshot(StrictContractModel):
    phase: str
    terminal_status: str | None
    task_counts: Mapping[str, int]
    resource_summary: Mapping[str, float]
    heartbeat_age_seconds: float = Field(ge=0.0)
    eta_status: str

    @field_validator("phase", "eta_status")
    @classmethod
    def _snapshot_text_is_safe(cls, value: str) -> str:
        return _safe_logical_value(value)

    @field_validator("terminal_status")
    @classmethod
    def _terminal_status_is_safe(cls, value: str | None) -> str | None:
        return None if value is None else _safe_logical_value(value)

    @model_validator(mode="after")
    def _snapshot_is_finite_and_immutable(self) -> Self:
        if any(count < 0 for count in self.task_counts.values()):
            raise ValueError("monitoring task counts must be non-negative")
        if any(not math.isfinite(value) for value in self.resource_summary.values()):
            raise ValueError("monitoring resources must be finite")
        if not math.isfinite(self.heartbeat_age_seconds):
            raise ValueError("monitoring heartbeat age must be finite")
        object.__setattr__(self, "task_counts", MappingProxyType(dict(self.task_counts)))
        object.__setattr__(
            self,
            "resource_summary",
            MappingProxyType(dict(self.resource_summary)),
        )
        return self

    @field_serializer("task_counts", "resource_summary")
    def _serialize_mappings(
        self,
        value: Mapping[str, int] | Mapping[str, float],
    ) -> dict[str, int | float]:
        return dict(value)
