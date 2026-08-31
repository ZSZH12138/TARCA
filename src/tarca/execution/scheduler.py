from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import psutil

from tarca.contracts import canonical_json_hash
from tarca.execution.contracts import ExecutionContext, PlannedTask
from tarca.execution.registry import ExecutorRegistry
from tarca.execution.resources import HostAdmissionPolicy, ResourceCapacity, plan_resources
from tarca.execution.state import AttemptState, ExecutionStateStore, ProcessIdentity, QueuedTask
from tarca.execution.supervision import RuntimeSupervisor
from tarca.execution.worker import run_worker


class RunTerminalStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerHandle:
    attempt_id: str
    process: ProcessLike | None


class PsutilProcessProbe:
    """Read a worker's persisted identity from its immutable argv and start time."""

    def __init__(self, process_factory: Callable[[int], Any] = psutil.Process) -> None:
        self._process_factory = process_factory

    @staticmethod
    def _argument(arguments: tuple[str, ...], name: str) -> str | None:
        for index, value in enumerate(arguments):
            if value == name and index + 1 < len(arguments):
                return arguments[index + 1]
            prefix = f"{name}="
            if value.startswith(prefix):
                return value.removeprefix(prefix)
        return None

    def inspect(self, pid: int) -> ProcessIdentity | None:
        if pid <= 0:
            return None
        try:
            process = self._process_factory(pid)
            arguments = tuple(str(value) for value in process.cmdline())
            run_id = self._argument(arguments, "--run-id")
            task_id = self._argument(arguments, "--task-id")
            if run_id is None or task_id is None:
                return None
            started = datetime.fromtimestamp(float(process.create_time()), UTC)
            return ProcessIdentity(pid, started, run_id, task_id)
        except (psutil.Error, OSError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class SchedulerLaunch:
    task: PlannedTask
    scientific_identity_sha256: str
    handle: object


class WorkerBackend(Protocol):
    backend_id: str

    def launch(self, task: PlannedTask, database_path: Path) -> object: ...

    def poll(self) -> tuple[object, ...]: ...


class SynchronousTestBackend:
    backend_id = "synchronous-test"

    def __init__(self, store: ExecutionStateStore, registry: ExecutorRegistry) -> None:
        self._store = store
        self._registry = registry

    def launch(self, task: PlannedTask, database_path: Path) -> WorkerHandle:
        del database_path
        context = ExecutionContext(
            run_id=self._store.claimed_task(task.attempt_id).run_id,
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_identity="synchronous-test-runtime",
            worker_identity=task.allocation.worker_id,
        )
        run_worker(context, self._store, self._registry)
        return WorkerHandle(task.attempt_id, None)

    def poll(self) -> tuple[object, ...]:
        return ()


class LocalMultiProcessBackend:
    backend_id = "local-multiprocess"

    def __init__(
        self,
        repository_root: Path,
        *,
        python_executable: str | None = None,
        cpu_ids: tuple[int, ...] | None = None,
        environment_overrides: dict[str, str] | None = None,
        popen_factory: Any = subprocess.Popen,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.python_executable = python_executable or sys.executable
        self.cpu_ids = cpu_ids or tuple(psutil.Process().cpu_affinity())
        self.environment_overrides = dict(environment_overrides or {})
        self._popen_factory = popen_factory
        self._handles: list[WorkerHandle] = []
        self._next_cpu = 0

    def _allocate_cpu_ids(self, count: int) -> tuple[int, ...]:
        if count > len(self.cpu_ids):
            raise ValueError("worker CPU request exceeds the configured affinity pool")
        start = self._next_cpu
        selected = tuple(
            self.cpu_ids[(start + offset) % len(self.cpu_ids)] for offset in range(count)
        )
        self._next_cpu = (start + count) % len(self.cpu_ids)
        return selected

    def launch(self, task: PlannedTask, database_path: Path) -> WorkerHandle:
        arguments = (
            self.python_executable,
            "-m",
            "tarca.execution.worker_entry",
            "--database",
            str(database_path),
            "--repository-root",
            str(self.repository_root),
            "--run-id",
            self._run_id(task.attempt_id, database_path),
            "--task-id",
            task.task_id,
            "--attempt-id",
            task.attempt_id,
            "--worker-id",
            task.allocation.worker_id,
        )
        environment = dict(os.environ)
        environment.update(self.environment_overrides)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(str(item) for item in task.allocation.gpu_ids),
                "OMP_NUM_THREADS": str(task.allocation.cpu_threads),
                "MKL_NUM_THREADS": str(task.allocation.cpu_threads),
                "TARCA_CPU_AFFINITY": ",".join(
                    str(item) for item in self._allocate_cpu_ids(task.allocation.cpu_threads)
                ),
            }
        )
        process = self._popen_factory(
            arguments,
            shell=False,
            env=environment,
            cwd=self.repository_root,
            start_new_session=True,
        )
        try:
            started_at = datetime.fromtimestamp(psutil.Process(process.pid).create_time(), UTC)
        except (psutil.Error, OSError):
            started_at = datetime.now(UTC)
        ExecutionStateStore(database_path).bind_running_process(
            task.attempt_id,
            task.allocation.worker_id,
            process.pid,
            started_at,
        )
        handle = WorkerHandle(task.attempt_id, process)
        self._handles.append(handle)
        return handle

    @staticmethod
    def _run_id(attempt_id: str, database_path: Path) -> str:
        store = ExecutionStateStore(database_path)
        return store.claimed_task(attempt_id).run_id

    def poll(self) -> tuple[WorkerHandle, ...]:
        finished = tuple(
            handle
            for handle in self._handles
            if handle.process is not None and handle.process.poll() is not None
        )
        if finished:
            finished_ids = {handle.attempt_id for handle in finished}
            self._handles = [
                handle for handle in self._handles if handle.attempt_id not in finished_ids
            ]
        return finished

    def terminate_all(self, *, timeout_seconds: float = 5.0) -> tuple[str, ...]:
        if timeout_seconds <= 0.0:
            raise ValueError("worker termination timeout must be positive")
        active = tuple(
            handle
            for handle in self._handles
            if handle.process is not None and handle.process.poll() is None
        )
        for handle in active:
            assert handle.process is not None
            handle.process.terminate()
        deadline = time.monotonic() + timeout_seconds
        while (
            any(handle.process is not None and handle.process.poll() is None for handle in active)
            and time.monotonic() < deadline
        ):
            time.sleep(min(0.05, timeout_seconds))
        for handle in active:
            if handle.process is not None and handle.process.poll() is None:
                handle.process.kill()
        active_ids = {handle.attempt_id for handle in active}
        self._handles = [handle for handle in self._handles if handle.attempt_id not in active_ids]
        return tuple(handle.attempt_id for handle in active)


class Scheduler:
    """Runtime-only scheduler; it cannot inspect model scores or gate evidence."""

    visible_columns = (
        "run_id",
        "task_id",
        "attempt_id",
        "phase",
        "state",
        "worker_id",
        "pid",
        "heartbeat_at_utc",
        "error_category",
        "resource_request",
        "resource_allocation",
    )

    def __init__(
        self,
        store: ExecutionStateStore,
        backend: WorkerBackend,
        capacity: ResourceCapacity,
        *,
        policy: HostAdmissionPolicy | None = None,
        poll_interval_seconds: float = 0.2,
        supervisor: RuntimeSupervisor | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("scheduler poll interval must be positive")
        self.store = store
        self.backend = backend
        self.capacity = capacity
        self.policy = policy
        self.poll_interval_seconds = poll_interval_seconds
        self.supervisor = supervisor

    def tick(self, run_id: str) -> tuple[SchedulerLaunch, ...]:
        self.backend.poll()
        queued = tuple(
            sorted(
                self.store.ready_tasks(run_id),
                key=lambda item: item.task.resource_request.gpu_count == 0,
            )
        )
        active = self.store.running_attempts(run_id)
        allocations = (
            plan_resources(
                tuple(item.task for item in queued),
                self.capacity,
                self.policy,
                active=tuple((item.task, item.allocation) for item in active),
            )
            if queued
            else ()
        )
        launches: list[SchedulerLaunch] = []
        for allocation in allocations:
            index = int(allocation.worker_id.rsplit("-", 1)[-1])
            selected: QueuedTask = queued[index]
            worker_id = f"worker-{selected.attempt_id}"
            bound_allocation = allocation.model_copy(update={"worker_id": worker_id})
            claim = self.store.claim_attempt(selected.attempt_id, worker_id, bound_allocation)
            if claim is None:
                continue
            planned = PlannedTask(
                task_id=claim.task.task_id,
                attempt_id=claim.attempt_id,
                executor_key=claim.executor_key,
                allocation=bound_allocation,
                input_refs=claim.task.inputs,
                expected_output_artifact_type=claim.task.output_artifact_type,
            )
            handle = self.backend.launch(planned, self.store.database_path)
            launches.append(
                SchedulerLaunch(
                    task=planned,
                    scientific_identity_sha256=canonical_json_hash(claim.task),
                    handle=handle,
                )
            )
        if self.supervisor is not None:
            self.supervisor.sample_if_due(run_id, os.getpid())
        return tuple(launches)

    def run_until_terminal(
        self,
        run_id: str,
        *,
        maximum_wait_seconds: float | None = None,
    ) -> RunTerminalStatus:
        started = time.monotonic()
        while True:
            self.tick(run_id)
            retried = False
            for failed in self.store.latest_failed_attempts(run_id):
                retry = self.store.retry_attempt(
                    failed.attempt_id,
                    failed.error_category,
                    lower_packing_applied=failed.error_category == "CUDA_OOM",
                )
                retried = retried or retry is not None
            if retried:
                continue
            counts = self.store.run_attempt_counts(run_id)
            total = sum(counts.values())
            if total and counts.get(AttemptState.COMPLETED.value, 0) == total:
                return RunTerminalStatus.COMPLETED
            if (
                counts.get(AttemptState.FAILED.value, 0)
                or counts.get(AttemptState.STALLED.value, 0)
            ) and not counts.get(AttemptState.RUNNING.value, 0):
                return RunTerminalStatus.FAILED
            if (
                maximum_wait_seconds is not None
                and time.monotonic() - started >= maximum_wait_seconds
            ):
                return RunTerminalStatus.STOPPED
            time.sleep(self.poll_interval_seconds)
