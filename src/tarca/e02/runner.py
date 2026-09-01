from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.e02.jobs import e02_artifact_store, e02_executor_registry
from tarca.e02.tasks import E02Graph, compile_e02_ready
from tarca.execution import (
    ExecutionStateStore,
    ExecutorRegistry,
    HostAdmissionPolicy,
    ResourceCapacity,
    RunPlanNode,
    RunTerminalStatus,
    Scheduler,
    SynchronousTestBackend,
)
from tarca.execution.scheduler import WorkerBackend
from tarca.execution.supervision import RuntimeSupervisor
from tarca.execution.telemetry import (
    PsutilNvmlTelemetryProbe,
    TelemetryPolicy,
    TelemetryProbe,
)


@dataclass(frozen=True, slots=True)
class E02RunResult:
    run_id: str
    graph_id: str
    status: RunTerminalStatus
    completed: tuple[tuple[str, ArtifactRef], ...]


def e02_host_admission_policy() -> HostAdmissionPolicy:
    return HostAdmissionPolicy(
        scheduler_monitor_cores=1,
        system_io_reserved_cores=3,
        maximum_data_cores=24,
        maximum_host_memory_bytes=200 * 1024**3,
        minimum_local_storage_free_bytes=200 * 1024**3,
        initial_loader_workers_per_gpu_job=3,
    )


def e02_scientific_plan_hash(graph: E02Graph, gpu_order: tuple[int, int]) -> str:
    if tuple(sorted(gpu_order)) != (0, 1):
        raise ValueError("GPU order must contain devices zero and one")
    return canonical_json_hash({"graph_id": graph.graph_id, "config_sha256": graph.config_sha256})


def run_e02_formal(
    graph: E02Graph,
    capacity: ResourceCapacity,
    *,
    repository_root: Path,
    database_path: Path,
    registry: ExecutorRegistry | None = None,
    backend: WorkerBackend | None = None,
    telemetry_probe: TelemetryProbe | None = None,
    policy: HostAdmissionPolicy | None = None,
    maximum_wait_seconds: float = 1.0,
) -> E02RunResult:
    root = repository_root.resolve()
    resolved_registry = registry or e02_executor_registry(root)
    state = ExecutionStateStore(
        database_path, artifact_verifier=e02_artifact_store(root).verify_artifact
    )
    run_id = f"run-{graph.graph_id.removeprefix('e02-graph-')}"
    state.create_run(run_id, graph.graph_id)
    state.register_run_plan(
        run_id,
        tuple(
            RunPlanNode(
                identity=n.identity,
                phase=n.phase,
                resource_request=n.resource_request,
                dependency_task_ids=n.dependency_ids,
            )
            for n in graph.nodes
        ),
    )
    telemetry = telemetry_probe or PsutilNvmlTelemetryProbe()
    scheduler = Scheduler(
        state,
        backend or SynchronousTestBackend(state, resolved_registry),
        capacity,
        policy=policy or e02_host_admission_policy(),
        poll_interval_seconds=0.001,
        supervisor=RuntimeSupervisor(
            state,
            telemetry,
            TelemetryPolicy(sample_interval_seconds=2.0),
        ),
    )
    try:
        while True:
            completed = state.completed_artifacts(run_id)
            if len(completed) == len(graph.nodes):
                return E02RunResult(
                    run_id,
                    graph.graph_id,
                    RunTerminalStatus.COMPLETED,
                    tuple(sorted(completed.items())),
                )
            manifest = compile_e02_ready(graph, completed)
            node_by_id = {n.node_id: n for n in graph.nodes}
            for task in manifest.tasks:
                node = node_by_id[task.task_id]
                state.enqueue_task(
                    run_id, task, node.executor_key, dependency_task_ids=node.dependency_ids
                )
            if not manifest.tasks and not state.running_attempts(run_id):
                return E02RunResult(
                    run_id,
                    graph.graph_id,
                    RunTerminalStatus.FAILED,
                    tuple(sorted(completed.items())),
                )
            status = scheduler.run_until_terminal(
                run_id, maximum_wait_seconds=maximum_wait_seconds
            )
            if status is RunTerminalStatus.FAILED:
                return E02RunResult(
                    run_id, graph.graph_id, status, tuple(sorted(completed.items()))
                )
    finally:
        close_probe = getattr(telemetry, "close", None)
        if callable(close_probe):
            close_probe()
