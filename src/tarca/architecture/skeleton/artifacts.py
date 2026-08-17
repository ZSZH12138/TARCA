"""Append-only artifact skeleton."""

from __future__ import annotations

from typing import Protocol, TypeVar

from tarca.contracts.common import ArtifactRef
from tarca.contracts.errors import UnimplementedCapabilityError

T = TypeVar("T")


class ArtifactStore(Protocol):
    def publish_atomic(self, value: T, artifact_type: str) -> ArtifactRef: ...

    def verify_artifact(self, reference: ArtifactRef) -> bool: ...

    def load_typed(self, reference: ArtifactRef, expected_type: type[T]) -> T: ...

    def resolve_artifact(self, reference: ArtifactRef) -> ArtifactRef: ...


def publish_atomic(value: object, artifact_type: str) -> ArtifactRef:
    raise UnimplementedCapabilityError("artifacts.publish_atomic")


def verify_artifact(reference: ArtifactRef) -> bool:
    raise UnimplementedCapabilityError("artifacts.verify_artifact")


def load_typed(reference: ArtifactRef, expected_type: type[T]) -> T:
    raise UnimplementedCapabilityError("artifacts.load_typed")


def resolve_artifact(reference: ArtifactRef) -> ArtifactRef:
    raise UnimplementedCapabilityError("artifacts.resolve_artifact")


def validate_artifact_reference(reference: ArtifactRef) -> ArtifactRef:
    if not isinstance(reference, ArtifactRef):
        raise TypeError("reference must be ArtifactRef")
    return reference
