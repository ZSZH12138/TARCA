"""Scheduler skeleton; manifests are immutable inputs."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.common import TaskAttemptRecord, TaskSpec
from tarca.contracts.future import TaskManifest


class Scheduler(Protocol):
    def lease(self, manifest: TaskManifest) -> tuple[TaskSpec, ...]: ...

    def record_attempt(self, attempt: TaskAttemptRecord) -> None: ...


def validate_manifest_immutability(manifest: TaskManifest) -> TaskManifest:
    if not isinstance(manifest, TaskManifest):
        raise TypeError("manifest must be TaskManifest")
    return manifest
