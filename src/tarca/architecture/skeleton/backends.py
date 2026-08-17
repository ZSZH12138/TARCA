"""Backend Protocols; third-party concrete types stop at this boundary."""

from __future__ import annotations

from typing import Protocol, TypeVar

from tarca.contracts.common import ArtifactRef
from tarca.contracts.future import InterventionResult

T = TypeVar("T")


class OTBackend(Protocol):
    def solve(self, source: object, target: object) -> object: ...


class InterventionBackend(Protocol):
    def apply(self, request: object) -> InterventionResult: ...


class StorageBackend(Protocol):
    def put(self, value: T) -> ArtifactRef: ...

    def get(self, reference: ArtifactRef, expected_type: type[T]) -> T: ...


def validate_backend(backend: object, method_name: str) -> object:
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("method_name must be a non-empty string")
    if not callable(getattr(backend, method_name, None)):
        raise TypeError(f"backend must expose {method_name}")
    return backend
