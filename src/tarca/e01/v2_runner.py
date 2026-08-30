from __future__ import annotations

import json
import os
import shutil
import signal
from pathlib import Path
from typing import Any, cast

import psutil

from tarca.e01.resources import worker_thread_environment
from tarca.e01.v2_config import E01V2Config
from tarca.e01.v2_jobs import e01_v2_artifact_store
from tarca.e01.v2_tasks import E01V2Graph, compile_e01_v2_graph
from tarca.execution import RunPlanNode
from tarca.execution.resources import HostAdmissionPolicy, ResourceCapacity
from tarca.execution.scheduler import (
    LocalMultiProcessBackend,
    PsutilProcessProbe,
    RunTerminalStatus,
    Scheduler,
)
from tarca.execution.state import ExecutionStateStore
from tarca.execution.supervision import RuntimeSupervisor
from tarca.execution.telemetry import PsutilNvmlTelemetryProbe, TelemetryPolicy


def _relative(root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository bind mount") from error


def _run_plan(graph: E01V2Graph) -> tuple[RunPlanNode, ...]:
    return tuple(
        RunPlanNode(
            identity=node.task.identity,
            phase=node.task.phase,
            resource_request=node.task.resource_request,
            dependency_task_ids=node.dependency_task_ids,
        )
        for node in graph.nodes
    )


def _enqueue_ready(graph: E01V2Graph, state: ExecutionStateStore, run_id: str) -> tuple[str, ...]:
    completed = state.completed_artifacts(run_id)
    enqueued: list[str] = []
    for node in graph.nodes:
        if node.task.task_id in completed or not set(node.dependency_task_ids).issubset(completed):
            continue
        task = node.task.model_copy(
            update={"inputs": tuple(completed[item] for item in node.dependency_task_ids)}
        )
        state.enqueue_task(
            run_id,
            task,
            node.executor_key,
            dependency_task_ids=node.dependency_task_ids,
        )
        enqueued.append(task.task_id)
    return tuple(enqueued)


def _capacity(preflight: dict[str, Any], artifact_root: Path) -> ResourceCapacity:
    inventory = cast(dict[str, Any], preflight["inventory"])
    disk = shutil.disk_usage(artifact_root)
    return ResourceCapacity(
        logical_cpu_count=int(inventory["logical_cpu_count"]),
        physical_cpu_count=int(inventory["physical_cpu_cores"]),
        available_memory_bytes=int(float(inventory["available_ram_gib"]) * 1024**3),
        gpu_memory_bytes=tuple(int(float(value) * 1024**3) for value in inventory["gpu_vram_gib"]),
        local_storage_available=True,
        local_storage_free_bytes=disk.free,
    )


def _read_final(root: Path, state: ExecutionStateStore, run_id: str) -> dict[str, Any]:
    reference = state.completed_artifacts(run_id).get("e01v2-aggregate")
    if reference is None:
        raise RuntimeError("E01-v2 final report artifact is missing")
    value = json.loads(e01_v2_artifact_store(root).load_bytes(reference))
    if not isinstance(value, dict):
        raise RuntimeError("E01-v2 final report artifact is invalid")
    return cast(dict[str, Any], value)


def run_formal_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    config: E01V2Config,
    prepared: dict[str, Any],
    preflight: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    del prepared
    root = repository_root.resolve()
    artifacts = artifact_root.resolve()
    os.environ.update(worker_thread_environment())
    os.environ.update(
        {
            "TARCA_EXECUTION_KIND": "e01-v2",
            "TARCA_E01_V2_CONFIG": _relative(root, config_path, "E01-v2 config"),
            "TARCA_E01_V2_ARTIFACT_ROOT": _relative(root, artifacts, "E01-v2 artifact root"),
            "TARCA_E01_DATABASE": str(artifacts / "runtime/execution.sqlite3"),
            "TARCA_E01_V2_GPU_BATCH_SIZE": str(preflight["capacity_plan"]["gpu_batch_size"]),
        }
    )
    graph = compile_e01_v2_graph(config)
    if graph.graph_id != preflight.get("graph_id"):
        raise RuntimeError("E01-v2 preflight graph identity drifted")
    run_id = f"run-{graph.graph_id.removeprefix('e01v2-graph-')}"
    database = artifacts / "runtime/execution.sqlite3"
    state = ExecutionStateStore(
        database,
        artifact_verifier=e01_v2_artifact_store(root).verify_artifact,
    )
    state.create_run(run_id, graph.graph_id)
    state.register_run_plan(run_id, _run_plan(graph))
    if resume:
        state.reconcile_processes(PsutilProcessProbe())
    if state.planned_task_count(run_id) != len(graph.nodes):
        raise RuntimeError("persisted E01-v2 run plan is incomplete")
    affinity = tuple(psutil.Process().cpu_affinity())
    service_cores = int(preflight["capacity_plan"]["service_cores"])
    worker_cpu_ids = affinity[service_cores:] if len(affinity) > service_cores else affinity
    backend = LocalMultiProcessBackend(root, cpu_ids=worker_cpu_ids)
    policy = HostAdmissionPolicy(
        scheduler_monitor_cores=1,
        system_io_reserved_cores=1,
        maximum_data_cores=int(preflight["capacity_plan"]["cpu_analysis_workers"]),
        maximum_host_memory_bytes=96 * 1024**3,
        minimum_local_storage_free_bytes=config.runtime_profile.minimum_free_storage_gib * 1024**3,
        initial_loader_workers_per_gpu_job=1,
    )
    telemetry = PsutilNvmlTelemetryProbe()
    supervisor = RuntimeSupervisor(
        state,
        telemetry,
        TelemetryPolicy(sample_interval_seconds=2.0),
    )
    scheduler = Scheduler(
        state,
        backend,
        _capacity(preflight, artifacts),
        policy=policy,
        supervisor=supervisor,
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_runtime(_signal_number: int, _frame: object) -> None:
        raise RuntimeError("E01-v2 runtime received a termination request")

    signal.signal(signal.SIGTERM, interrupt_runtime)
    try:
        while True:
            if len(state.completed_artifacts(run_id)) == len(graph.nodes):
                return _read_final(root, state, run_id)
            ready = _enqueue_ready(graph, state, run_id)
            if not ready and not state.running_attempts(run_id):
                failed = state.latest_failed_attempts(run_id)
                if failed:
                    categories = ",".join(sorted({item.error_category for item in failed}))
                    raise RuntimeError(f"E01-v2 execution has failed attempts: {categories}")
                raise RuntimeError("E01-v2 graph has no ready work before completion")
            status = scheduler.run_until_terminal(run_id, maximum_wait_seconds=1.0)
            if status is RunTerminalStatus.FAILED:
                raise RuntimeError("E01-v2 scheduled execution stopped with failed work")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        backend.terminate_all()
        telemetry.close()
