from pathlib import Path

from tarca.contracts import canonical_json_bytes
from tarca.e02.config import load_e02_config
from tarca.e02.jobs import e02_artifact_store, e02_executor_registry
from tarca.e02.runner import (
    e02_host_admission_policy,
    e02_scientific_plan_hash,
    run_e02_formal,
)
from tarca.e02.tasks import FrozenStage2Input, compile_e02_graph, compile_e02_ready
from tarca.execution import (
    ExecutionStateStore,
    ExecutorRegistry,
    GpuSample,
    HostTelemetry,
    ResourceCapacity,
    RunTerminalStatus,
    plan_resources,
)
from tests.e02.test_tasks import ROOT, _ref


def _graph():
    return compile_e02_graph(
        load_e02_config(ROOT / "configs/e02/e02_v1.yaml"),
        FrozenStage2Input(
            freeze_receipt=_ref("freeze", "STAGE2_FREEZE_RECEIPT"),
            sealed_access_grant=_ref("grant", "SEALED_ACCESS_GRANT"),
            frozen=True,
        ),
    )


def test_every_e02_executor_is_exactly_allowlisted() -> None:
    graph = _graph()
    assert set(e02_executor_registry(Path(ROOT)).keys) == {
        node.executor_key for node in graph.nodes
    }


def test_e02_science_hash_is_gpu_placement_invariant() -> None:
    graph = _graph()
    assert e02_scientific_plan_hash(graph, (0, 1)) == e02_scientific_plan_hash(graph, (1, 0))


def test_e02_capacity_policy_reserves_four_cores_and_requires_200_gib_storage() -> None:
    policy = e02_host_admission_policy()

    assert policy.maximum_data_cores == 24
    assert policy.scheduler_monitor_cores == 1
    assert policy.system_io_reserved_cores == 3
    assert policy.maximum_host_memory_bytes == 200 * 1024**3
    assert policy.minimum_local_storage_free_bytes == 200 * 1024**3


def test_e02_prediction_wave_fills_both_gpus_and_backfills_linear_cpu_work() -> None:
    graph = _graph()
    completed = {
        node.node_id: _ref(node.node_id, node.output_artifact_type)
        for node in graph.nodes
        if node.phase in {"GRANT_VERIFY", "STAGE2_VERIFY", "FORMAL_OPEN"}
    }
    ready = compile_e02_ready(graph, completed)
    predictions = tuple(task for task in ready.tasks if task.phase == "FORMAL_PREDICT")
    ordered = tuple(sorted(predictions, key=lambda task: task.resource_request.gpu_count == 0))
    capacity = ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=300 * 1024**3,
    )

    allocations = plan_resources(ordered, capacity, e02_host_admission_policy())
    launched = tuple((ordered[int(item.worker_id.rsplit("-", 1)[1])], item) for item in allocations)

    assert len(launched) == 3
    assert {item.gpu_ids for _, item in launched if item.gpu_ids} == {(0,), (1,)}
    assert any(task.identity.model_id == "STRONGEST_LINEAR" for task, _ in launched)
    assert sum(item.cpu_threads for _, item in launched) == 16
    assert sum(item.host_memory_gib_limit for _, item in launched) == 96.0


def test_e02_runner_executes_complete_graph_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARCA_E02_ARTIFACT_ROOT", "artifacts/e02")
    graph = _graph()

    def execute(task, context, progress):
        del context, progress
        return e02_artifact_store(tmp_path, task).publish_bytes(
            canonical_json_bytes({"task_id": task.task_id}) + b"\n",
            task.output_artifact_type,
            "application/json",
            "test-v1",
        )

    registry = ExecutorRegistry(
        {node.executor_key: execute for node in graph.nodes}
    )
    capacity = ResourceCapacity(
        logical_cpu_count=32,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=300 * 1024**3,
    )

    result = run_e02_formal(
        graph,
        capacity,
        repository_root=tmp_path,
        database_path=tmp_path / "state.sqlite3",
        registry=registry,
    )

    assert result.status is RunTerminalStatus.COMPLETED
    assert len(result.completed) == len(graph.nodes)


class _TelemetryProbe:
    def __init__(self) -> None:
        self.closed = False

    def host_snapshot(self, process_id: int) -> HostTelemetry:
        assert process_id > 0
        return HostTelemetry(
            host_cpu_percent=40.0,
            effective_busy_cores=6.0,
            process_rss_bytes=1024**3,
            process_pss_bytes=512 * 1024**2,
            process_affinity_cpu_ids=tuple(range(24)),
            host_memory_used_bytes=40 * 1024**3,
            disk_read_bytes_per_second=512.0,
            disk_write_bytes_per_second=1024.0,
        )

    def gpu_samples(self) -> tuple[GpuSample, ...]:
        return (
            GpuSample(0, 70.0, 10 * 1024**3, 24 * 1024**3, 280.0, 64.0, ()),
            GpuSample(1, 65.0, 9 * 1024**3, 24 * 1024**3, 260.0, 62.0, ()),
        )

    def close(self) -> None:
        self.closed = True


def test_e02_runner_persists_runtime_telemetry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TARCA_E02_ARTIFACT_ROOT", "artifacts/e02")
    graph = _graph()

    def execute(task, context, progress):
        del context, progress
        return e02_artifact_store(tmp_path, task).publish_bytes(
            canonical_json_bytes({"task_id": task.task_id}) + b"\n",
            task.output_artifact_type,
            "application/json",
            "test-v1",
        )

    registry = ExecutorRegistry({node.executor_key: execute for node in graph.nodes})
    capacity = ResourceCapacity(
        logical_cpu_count=32,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=300 * 1024**3,
    )
    probe = _TelemetryProbe()
    database = tmp_path / "telemetry.sqlite3"

    result = run_e02_formal(
        graph,
        capacity,
        repository_root=tmp_path,
        database_path=database,
        registry=registry,
        telemetry_probe=probe,
    )

    samples = ExecutionStateStore(database).resource_samples(result.run_id, attempt_id=None)
    assert samples[-1].gpu_samples[1].utilization_percent == 65.0
    assert probe.closed is True
