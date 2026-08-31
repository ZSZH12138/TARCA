from pathlib import Path

from tarca.contracts import canonical_json_bytes
from tarca.e02.config import load_e02_config
from tarca.e02.jobs import e02_artifact_store, e02_executor_registry
from tarca.e02.runner import e02_scientific_plan_hash, run_e02_formal
from tarca.e02.tasks import FrozenStage2Input, compile_e02_graph
from tarca.execution import ExecutorRegistry, ResourceCapacity, RunTerminalStatus
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
