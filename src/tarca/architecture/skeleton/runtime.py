"""Runtime skeleton; execution cannot alter scientific identity."""

from __future__ import annotations

from collections.abc import Mapping

from tarca.contracts.common import ExecutionContext, TaskResult
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import TaskManifest


def inspect_resources() -> Mapping[str, float]:
    raise UnimplementedCapabilityError("runtime.inspect_resources")


def qualify_runtime() -> None:
    raise UnimplementedCapabilityError("runtime.qualify_runtime")


def plan_execution(manifest: TaskManifest) -> object:
    raise UnimplementedCapabilityError("runtime.plan_execution")


def execute_task(context: ExecutionContext) -> TaskResult:
    raise UnimplementedCapabilityError("runtime.execute_task")


def reconcile_run(manifest: TaskManifest) -> tuple[TaskResult, ...]:
    raise UnimplementedCapabilityError("runtime.reconcile_run")


def validate_execution_context(context: ExecutionContext) -> ExecutionContext:
    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be ExecutionContext")
    return context
