from pathlib import Path

from tarca.contracts import canonical_json_bytes
from tarca.execution import (
    ExecutionStateStore,
    ExecutorRegistry,
    GpuSample,
    HostTelemetry,
    ResourceCapacity,
    RunTerminalStatus,
)
from tarca.stage2.config import load_stage2_config
from tarca.stage2.jobs import stage2_artifact_store, stage2_executor_registry
from tarca.stage2.runner import run_stage2, stage2_scientific_plan_hash
from tarca.stage2.tasks import compile_stage2_graph
from tests.stage2.test_tasks import ROOT, _inputs


def test_every_stage2_executor_is_exactly_allowlisted() -> None:
    graph = compile_stage2_graph(
        load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml"), _inputs()
    )
    assert set(stage2_executor_registry(Path(ROOT)).keys) == {
        node.executor_key for node in graph.nodes
    }


def test_stage2_science_hash_is_gpu_placement_invariant() -> None:
    config = load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml")
    graph = compile_stage2_graph(config, _inputs())
    assert stage2_scientific_plan_hash(graph, (0, 1)) == stage2_scientific_plan_hash(graph, (1, 0))


def test_stage2_runner_executes_complete_graph_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2")
    graph = compile_stage2_graph(
        load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml"), _inputs()
    )

    def execute(task, context, progress):
        del context, progress
        return stage2_artifact_store(tmp_path, task).publish_bytes(
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

    result = run_stage2(
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
            host_cpu_percent=50.0,
            effective_busy_cores=8.0,
            process_rss_bytes=2 * 1024**3,
            process_pss_bytes=1024**3,
            process_affinity_cpu_ids=tuple(range(24)),
            host_memory_used_bytes=48 * 1024**3,
            disk_read_bytes_per_second=1024.0,
            disk_write_bytes_per_second=2048.0,
        )

    def gpu_samples(self) -> tuple[GpuSample, ...]:
        return (
            GpuSample(0, 80.0, 12 * 1024**3, 24 * 1024**3, 320.0, 68.0, ()),
            GpuSample(1, 75.0, 11 * 1024**3, 24 * 1024**3, 300.0, 66.0, ()),
        )

    def close(self) -> None:
        self.closed = True


def test_stage2_runner_persists_runtime_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2")
    graph = compile_stage2_graph(
        load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml"), _inputs()
    )

    def execute(task, context, progress):
        del context, progress
        return stage2_artifact_store(tmp_path, task).publish_bytes(
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

    result = run_stage2(
        graph,
        capacity,
        repository_root=tmp_path,
        database_path=database,
        registry=registry,
        telemetry_probe=probe,
    )

    samples = ExecutionStateStore(database).resource_samples(result.run_id, attempt_id=None)
    assert samples[-1].gpu_samples[0].utilization_percent == 80.0
    assert probe.closed is True
