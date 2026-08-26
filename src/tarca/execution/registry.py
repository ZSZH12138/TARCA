from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Protocol

from tarca.contracts import ArtifactRef
from tarca.execution.contracts import ExecutionContext, TaskSpec


class ProgressSink(Protocol):
    def report(self, progress: object) -> None:
        raise NotImplementedError


Executor = Callable[[TaskSpec, ExecutionContext, ProgressSink], ArtifactRef]


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, Executor]) -> None:
        if not executors:
            raise ValueError("executor registry must not be empty")
        if any(re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", key) is None for key in executors):
            raise ValueError("executor key must be a safe registry identifier")
        if any(not callable(executor) for executor in executors.values()):
            raise TypeError("executor registry values must be callable")
        self._executors = MappingProxyType(dict(executors))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))

    def resolve(self, key: str) -> Executor:
        try:
            return self._executors[key]
        except KeyError as error:
            raise ValueError("executor key is not allowlisted") from error
