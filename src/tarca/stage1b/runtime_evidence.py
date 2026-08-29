from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.execution.resources import ResourceCapacity
from tarca.execution.scheduler import PsutilProcessProbe, Scheduler, WorkerBackend
from tarca.execution.state import ExecutionStateStore
from tarca.execution.supervision import RuntimeSupervisor
from tarca.execution.telemetry import (
    PsutilNvmlTelemetryProbe,
    TelemetryPolicy,
    TelemetryProbe,
)
from tarca.stage1b.evidence_io import sha256_bytes
from tarca.stage1b.hardware import inventory_hardware


def _runtime_receipt(runtime_root: Path, filename: str) -> tuple[dict[str, Any], str]:
    path = runtime_root.resolve() / filename
    if not path.is_file():
        raise RuntimeError(f"required runtime receipt is missing: {filename}")
    payload = path.read_bytes()
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"runtime receipt is not a JSON object: {filename}")
    return cast(dict[str, Any], decoded), sha256_bytes(payload)


def _runtime_plan_nodes(graph: Any) -> tuple[Any, ...]:
    from tarca.execution import RunPlanNode

    return tuple(
        RunPlanNode(
            identity=node.identity,
            phase=node.phase,
            resource_request=node.resource_request,
            dependency_task_ids=node.dependency_ids,
        )
        for node in graph.nodes
    )


def _qualification_execution_evidence(
    repository_root: Path,
    runtime_root: Path,
    graph: Any,
    completed: dict[str, ArtifactRef],
    capacity: Any,
) -> dict[str, Any]:
    """Build a hash-linked, science-blind audit envelope before the final receipt job."""
    from tarca.execution import (
        ExecutionPlan,
        PlannedTask,
        ResourceAllocation,
        TaskManifest,
        TaskSpec,
    )
    from tarca.stage1b.compiler import repository_v2_inputs
    from tarca.stage1b.runner import _jsonable, _write_json

    inputs = repository_v2_inputs(repository_root)
    final_node = graph.nodes[-1]
    expected_completed_ids = {node.node_id for node in graph.nodes[:-1]}
    if set(completed) != expected_completed_ids or final_node.phase != "QUALIFICATION_RECEIPT":
        raise RuntimeError("qualification execution evidence requires every pre-receipt task")
    if graph.config_sha256 != inputs.config_sha256 or graph.code_sha256 != inputs.code_sha256:
        raise RuntimeError("qualification graph identity drifted from the compiled inputs")

    environment, environment_hash = _runtime_receipt(runtime_root, "environment_receipt_v2.json")
    precision, precision_hash = _runtime_receipt(runtime_root, "precision_receipt_v2.json")
    official_sources, official_sources_hash = _runtime_receipt(
        runtime_root, "official_sources_receipt_v2.json"
    )
    hardware, hardware_hash = _runtime_receipt(runtime_root, "hardware_probe_v2.json")
    if environment.get("cuda_probe_passed") is not True:
        raise RuntimeError("server environment receipt did not pass both CUDA probes")
    if precision.get("selected") not in {"FP32", "AMP_FP16"}:
        raise RuntimeError("precision receipt has no approved selection")
    if hardware.get("decision", {}).get("feasible") is not True:
        raise RuntimeError("hardware receipt is not feasible")

    raw_sources = official_sources.get("sources")
    if not isinstance(raw_sources, list):
        raise RuntimeError("official source receipt contains no sources")
    received_source_commits = {
        str(item.get("source_id")): str(item.get("commit"))
        for item in raw_sources
        if isinstance(item, dict)
    }
    expected_source_commits = {
        source.source_id: source.commit for source in inputs.world_suite.sources
    }
    if received_source_commits != expected_source_commits:
        raise RuntimeError("official source receipt drifted from the v2 source manifest")

    all_tasks = tuple(
        TaskSpec(
            identity=node.identity,
            phase=node.phase,
            inputs=tuple(completed[dependency] for dependency in node.dependency_ids),
            output_artifact_type=node.output_artifact_type,
            resource_request=node.resource_request,
        )
        for node in graph.nodes
    )
    task_manifest = TaskManifest(
        manifest_id=f"stage1b-full-{graph.graph_id.removeprefix('stage1b-graph-')}",
        tasks=all_tasks,
        completed_task_policy="NEVER_RERUN",
    )
    task_manifest_hash = canonical_json_hash(task_manifest)
    gpu_index = 0
    planned_tasks: list[PlannedTask] = []
    for index, task in enumerate(all_tasks):
        request = task.resource_request
        gpu_ids: tuple[int, ...] = ()
        if request.gpu_count:
            if request.gpu_count != 1 or not capacity.gpu_memory_bytes:
                raise RuntimeError("qualification plan contains an unsupported GPU request")
            gpu_ids = (gpu_index % len(capacity.gpu_memory_bytes),)
            gpu_index += 1
        planned_tasks.append(
            PlannedTask(
                task_id=task.task_id,
                attempt_id=f"{task.task_id}-attempt-1",
                executor_key=next(
                    node.executor_key for node in graph.nodes if node.node_id == task.task_id
                ),
                allocation=ResourceAllocation(
                    cpu_threads=request.cpu_threads,
                    gpu_ids=gpu_ids,
                    host_memory_gib_limit=request.host_memory_gib,
                    worker_id=f"planned-worker-{index}",
                ),
                input_refs=task.inputs,
                expected_output_artifact_type=task.output_artifact_type,
            )
        )
    created_at_value = hardware.get("created_at_utc")
    if not isinstance(created_at_value, str):
        raise RuntimeError("hardware receipt creation time is missing")
    plan_identity_hash = canonical_json_hash(
        {"manifest": task_manifest_hash, "capacity": _jsonable(capacity)}
    )
    execution_plan = ExecutionPlan(
        plan_id=f"stage1b-plan-{plan_identity_hash}",
        task_manifest_id=task_manifest.manifest_id,
        backend_id="local-multiprocess",
        planned_tasks=tuple(planned_tasks),
        max_concurrency=max(1, len(capacity.gpu_memory_bytes)),
        resource_snapshot_hash=canonical_json_hash(_jsonable(capacity)),
        created_at=datetime.fromisoformat(created_at_value),
    )
    reproduction_refs = tuple(
        completed[node.node_id]
        for node in graph.nodes
        if node.output_artifact_type == "OFFICIAL_REPRODUCTION_RECEIPT"
    )
    if len(reproduction_refs) != len(inputs.reproduction_suite.cases):
        raise RuntimeError("official reproduction evidence is incomplete")
    evidence = {
        "official_source_receipt_sha256": official_sources_hash,
        "reproduction_receipt_sha256": canonical_json_hash(
            [ref.model_dump(mode="json") for ref in reproduction_refs]
        ),
        "environment_receipt_sha256": environment_hash,
        "precision_receipt_sha256": precision_hash,
        "run_graph_sha256": canonical_json_hash(_jsonable(graph)),
        "task_manifest_sha256": task_manifest_hash,
        "execution_plan_sha256": canonical_json_hash(execution_plan),
        "hardware_receipt_sha256": hardware_hash,
        "completed_task_count": len(graph.nodes),
        "expected_task_count": len(graph.nodes),
        "source_drift_detected": False,
        "identity_drift_detected": False,
    }
    document = {
        "schema_version": "2.0.0",
        "qualification_evidence": evidence,
        "run_graph": _jsonable(graph),
        "task_manifest": task_manifest.model_dump(mode="json"),
        "execution_plan": execution_plan.model_dump(mode="json"),
        "reproduction_artifacts": [ref.model_dump(mode="json") for ref in reproduction_refs],
    }
    _write_json(runtime_root / "qualification_execution_evidence_v2.json", document)
    return evidence


def _runtime_scheduler(
    state: ExecutionStateStore,
    backend: WorkerBackend,
    capacity: ResourceCapacity,
    *,
    telemetry_probe: TelemetryProbe | None = None,
) -> Scheduler:
    probe = telemetry_probe or PsutilNvmlTelemetryProbe()
    supervisor = RuntimeSupervisor(
        state,
        probe,
        TelemetryPolicy(sample_interval_seconds=2.0),
    )
    return Scheduler(state, backend, capacity, supervisor=supervisor)


def _enqueue_ready_tasks(
    graph: Any,
    state: ExecutionStateStore,
    run_id: str,
) -> tuple[str, ...]:
    from tarca.stage1b.compiler import compile_ready_manifest

    completed = state.completed_artifacts(run_id)
    manifest = compile_ready_manifest(graph, completed)
    node_by_id = {node.node_id: node for node in graph.nodes}
    for task in manifest.tasks:
        node = node_by_id[task.task_id]
        state.enqueue_task(
            run_id,
            task,
            node.executor_key,
            dependency_task_ids=node.dependency_ids,
        )
    return tuple(task.task_id for task in manifest.tasks)


def run_scheduled_qualification(
    repository_root: Path,
    artifact_root: Path,
    qualification_config: Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen official Stage1B graph through isolated local workers."""
    from tarca.execution.scheduler import LocalMultiProcessBackend, RunTerminalStatus
    from tarca.stage1b.compiler import (
        compile_stage1b_graph,
        repository_v2_inputs,
    )
    from tarca.stage1b.jobs import stage1b_artifact_store
    from tarca.stage1b.runner import _load_hardware_receipt

    root = repository_root.resolve()
    resolved_artifact_root = artifact_root.resolve()
    worlds_path = root / "configs/stage1b/worlds_v2.yaml"
    qualification_path = (
        qualification_config or root / "configs/stage1b/qualification_v2.yaml"
    ).resolve()
    artifact_relative = resolved_artifact_root.relative_to(root)
    qualification_relative = qualification_path.relative_to(root)
    os.environ["TARCA_STAGE1B_ARTIFACT_ROOT"] = artifact_relative.as_posix()
    os.environ["TARCA_STAGE1B_QUALIFICATION_CONFIG"] = qualification_relative.as_posix()
    inputs = repository_v2_inputs(root, qualification_path)
    _load_hardware_receipt(
        resolved_artifact_root / "runtime",
        worlds_path,
        qualification_path,
        inputs.world_suite.source_manifest_sha256(),
    )
    inventory = inventory_hardware()
    disk = shutil.disk_usage(resolved_artifact_root.parent)
    capacity = ResourceCapacity(
        logical_cpu_count=inventory.logical_cpu_count,
        physical_cpu_count=inventory.physical_cpu_count,
        available_memory_bytes=inventory.available_memory_bytes,
        gpu_memory_bytes=inventory.gpu_vram_bytes,
        local_storage_available=True,
        local_storage_free_bytes=disk.free,
    )
    graph = compile_stage1b_graph(inputs)
    run_id = f"run-{graph.graph_id.removeprefix('stage1b-graph-')}"
    database_path = resolved_artifact_root / "runtime/execution.sqlite3"
    verifier = stage1b_artifact_store(root).verify_artifact
    state = ExecutionStateStore(database_path, artifact_verifier=verifier)
    state.create_run(run_id, graph.graph_id)
    plan_nodes = _runtime_plan_nodes(graph)
    state.register_run_plan(run_id, plan_nodes)
    state.reconcile_processes(PsutilProcessProbe())
    if state.planned_task_count(run_id) != len(graph.nodes):
        raise RuntimeError("persisted Stage1B run plan is incomplete")
    backend = LocalMultiProcessBackend(root)
    scheduler = _runtime_scheduler(state, backend, capacity)
    node_by_id = {node.node_id: node for node in graph.nodes}
    while True:
        completed = state.completed_artifacts(run_id)
        if len(completed) == len(graph.nodes):
            final_node = graph.nodes[-1]
            final_ref = completed[final_node.node_id]
            payload = stage1b_artifact_store(root).load_bytes(final_ref)
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise RuntimeError("final Stage1B receipt artifact is invalid")
            return cast(dict[str, Any], result)
        ready_task_ids = _enqueue_ready_tasks(graph, state, run_id)
        if not ready_task_ids:
            raise RuntimeError("Stage1B task graph has no ready work before completion")
        if (
            len(ready_task_ids) == 1
            and node_by_id[ready_task_ids[0]].phase == "QUALIFICATION_RECEIPT"
        ):
            _qualification_execution_evidence(
                root,
                resolved_artifact_root / "runtime",
                graph,
                completed,
                capacity,
            )
        status = scheduler.run_until_terminal(run_id, maximum_wait_seconds=1.0)
        if status is RunTerminalStatus.FAILED:
            raise RuntimeError(f"Stage1B scheduled execution stopped with {status.value}")
