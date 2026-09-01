from pathlib import Path

from tarca.contracts import canonical_json_bytes
from tarca.e02.config import load_e02_config
from tarca.e02.jobs import e02_artifact_store, e02_executor_registry
from tarca.e02.runner import e02_scientific_plan_hash, run_e02_formal
from tarca.e02.tasks import FrozenStage2Input, compile_e02_graph
from tarca.execution import (
    ExecutionStateStore,
    ExecutorRegistry,
    GpuSample,
    HostTelemetry,
    ResourceCapacity,
    RunTerminalStatus,
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
