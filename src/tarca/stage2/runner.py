from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.execution import (
    ExecutionStateStore,
    ExecutorRegistry,
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
from tarca.stage2.jobs import stage2_artifact_store, stage2_executor_registry
from tarca.stage2.tasks import Stage2Graph, compile_stage2_ready


@dataclass(frozen=True, slots=True)
class Stage2RunResult:
    run_id: str
    graph_id: str
    status: RunTerminalStatus
    completed: tuple[tuple[str, ArtifactRef], ...]


def stage2_scientific_plan_hash(graph: Stage2Graph, gpu_order: tuple[int, int]) -> str:
    if tuple(sorted(gpu_order)) != (0, 1):
        raise ValueError("GPU order must contain devices zero and one")
    return canonical_json_hash({"graph_id": graph.graph_id, "config_sha256": graph.config_sha256})


def run_stage2(
    graph: Stage2Graph,
    capacity: ResourceCapacity,
    *,
    repository_root: Path,
    database_path: Path,
    registry: ExecutorRegistry | None = None,
    backend: WorkerBackend | None = None,
    telemetry_probe: TelemetryProbe | None = None,
    maximum_wait_seconds: float = 1.0,
) -> Stage2RunResult:
    root = repository_root.resolve()
    resolved_registry = registry or stage2_executor_registry(root)
    verifier = stage2_artifact_store(root).verify_artifact
    state = ExecutionStateStore(database_path, artifact_verifier=verifier)
    run_id = f"run-{graph.graph_id.removeprefix('stage2-graph-')}"
    state.create_run(run_id, graph.graph_id)
    state.register_run_plan(
        run_id,
        tuple(
            RunPlanNode(
                identity=node.identity,
                phase=node.phase,
                resource_request=node.resource_request,
                dependency_task_ids=node.dependency_ids,
            )
            for node in graph.nodes
        ),
    )
    scheduler_backend = backend or SynchronousTestBackend(state, resolved_registry)
    telemetry = telemetry_probe or PsutilNvmlTelemetryProbe()
    scheduler = Scheduler(
        state,
        scheduler_backend,
        capacity,
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
                return Stage2RunResult(
                    run_id,
                    graph.graph_id,
                    RunTerminalStatus.COMPLETED,
                    tuple(sorted(completed.items())),
                )
            manifest = compile_stage2_ready(graph, completed)
            node_by_id = {node.node_id: node for node in graph.nodes}
            for task in manifest.tasks:
                node = node_by_id[task.task_id]
                state.enqueue_task(
                    run_id, task, node.executor_key, dependency_task_ids=node.dependency_ids
                )
            if not manifest.tasks and not state.running_attempts(run_id):
                return Stage2RunResult(
                    run_id,
                    graph.graph_id,
                    RunTerminalStatus.FAILED,
                    tuple(sorted(completed.items())),
                )
            status = scheduler.run_until_terminal(
                run_id, maximum_wait_seconds=maximum_wait_seconds
            )
            if status is RunTerminalStatus.FAILED:
                return Stage2RunResult(
                    run_id, graph.graph_id, status, tuple(sorted(completed.items()))
                )
    finally:
        close_probe = getattr(telemetry, "close", None)
        if callable(close_probe):
            close_probe()
