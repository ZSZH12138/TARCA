"""Immutable cross-plane identities and execution-safe value contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
COMPLETED_TASK_POLICY = "NEVER_RERUN"


class TaskState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_hash(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if _HASH_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a sha256:<64 hex> value")
    return text


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A content-identified artifact; a path is only a placement hint."""

    artifact_id: str
    artifact_type: str
    content_hash: str
    schema_version: str
    relative_path: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.artifact_type, "artifact_type")
        _require_hash(self.content_hash, "content_hash")
        _require_text(self.schema_version, "schema_version")
        if self.relative_path is not None:
            _require_text(self.relative_path, "relative_path")

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        """Return the stable identity without the filesystem placement."""

        return (self.artifact_id, self.artifact_type, self.content_hash, self.schema_version)

    def to_mapping(self) -> dict[str, str | None]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ArtifactRef:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            artifact_id=payload["artifact_id"],
            artifact_type=payload["artifact_type"],
            content_hash=payload["content_hash"],
            schema_version=payload["schema_version"],
            relative_path=payload.get("relative_path"),
        )


@dataclass(frozen=True, slots=True)
class ScientificIdentity:
    """Stable identity of a scientific task, independent of retry attempts."""

    protocol_id: str
    experiment_id: str
    task_id: str
    model_id: str
    data_id: str
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("protocol_id", "experiment_id", "task_id", "model_id", "data_id"):
            _require_text(getattr(self, field_name), field_name)
        _require_non_negative_int(self.seed, "seed")

    @property
    def identity_hash(self) -> str:
        return _canonical_hash(self.to_mapping())

    def to_mapping(self) -> dict[str, str | int]:
        return {
            "protocol_id": self.protocol_id,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "data_id": self.data_id,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ScientificIdentity:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            protocol_id=payload["protocol_id"],
            experiment_id=payload["experiment_id"],
            task_id=payload["task_id"],
            model_id=payload["model_id"],
            data_id=payload["data_id"],
            seed=payload["seed"],
        )


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Execution resource request; it cannot change scientific identity."""

    cpu_threads: int = 1
    gpu_count: int = 0
    gpu_memory_gib: float = 0.0
    host_memory_gib: float = 0.0

    def __post_init__(self) -> None:
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if self.gpu_count < 0 or self.gpu_memory_gib < 0 or self.host_memory_gib < 0:
            raise ValueError("resource values must be non-negative")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable scientific task manifest entry."""

    identity: ScientificIdentity
    phase: str
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifact_type: str = "task_result"
    resource_request: ResourceRequest = ResourceRequest()

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    def __post_init__(self) -> None:
        _require_text(self.phase, "phase")
        _require_text(self.output_artifact_type, "output_artifact_type")
        if not isinstance(self.identity, ScientificIdentity):
            raise TypeError("identity must be ScientificIdentity")
        if not isinstance(self.input_artifacts, tuple):
            raise TypeError("input_artifacts must be a tuple")
        if not all(isinstance(item, ArtifactRef) for item in self.input_artifacts):
            raise TypeError("input_artifacts must contain ArtifactRef values")


@dataclass(frozen=True, slots=True)
class TaskAttemptRecord:
    """Retry bookkeeping that cannot alter the scientific task identity."""

    task_id: str
    attempt_id: str
    attempt_number: int
    state: TaskState

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.attempt_id, "attempt_id")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if not isinstance(self.state, TaskState):
            raise TypeError("state must be TaskState")


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Published result metadata; a completed result is never silently rerun."""

    task_id: str
    attempt_id: str
    state: TaskState
    artifact: ArtifactRef | None = None

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.attempt_id, "attempt_id")
        if not isinstance(self.state, TaskState):
            raise TypeError("state must be TaskState")
        if self.state is TaskState.COMPLETED and self.artifact is None:
            raise ValueError("completed TaskResult requires an ArtifactRef")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Governance decision without exposing partial scientific payloads."""

    gate_id: str
    status: GateStatus
    rationale: str
    evidence: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.gate_id, "gate_id")
        _require_text(self.rationale, "rationale")
        if not isinstance(self.status, GateStatus):
            raise TypeError("status must be GateStatus")
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, ArtifactRef) for item in self.evidence
        ):
            raise TypeError("evidence must be a tuple of ArtifactRef values")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime identity bound to a task attempt, not to scientific inputs."""

    run_id: str
    task_id: str
    attempt_id: str
    runtime_identity: str
    worker_identity: str

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "task_id",
            "attempt_id",
            "runtime_identity",
            "worker_identity",
        ):
            _require_text(getattr(self, field_name), field_name)


def freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """Freeze terminal-safe metadata without introducing a cross-module dict type."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(dict(metadata))
