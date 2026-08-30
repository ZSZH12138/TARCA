from __future__ import annotations

from pathlib import Path

from tarca.e01.v2_config import E01V2Config, load_e01_v2_config
from tarca.e01.v2_tasks import compile_e01_v2_graph

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e01/e01_v2.yaml"


def test_v2_graph_pipelines_fifty_gpu_generations_into_fifty_cpu_analyses() -> None:
    config = load_e01_v2_config(CONFIG)
    graph = compile_e01_v2_graph(config)
    generation = tuple(node for node in graph.nodes if node.phase == "E01_A_V2_GPU_GENERATE")
    analysis = tuple(node for node in graph.nodes if node.phase == "E01_A_V2_CPU_ANALYZE")
    aggregate = tuple(node for node in graph.nodes if node.phase == "E01_V2_AGGREGATE")

    assert len(graph.nodes) == 101
    assert len(generation) == len(analysis) == 50
    assert len(aggregate) == 1
    assert {node.task.identity.seed for node in generation} == set(config.formal_seeds)
    assert all(node.task.resource_request.gpu_count == 1 for node in generation)
    assert all(node.task.resource_request.cpu_threads == 1 for node in analysis)
    assert all(len(node.dependency_task_ids) == 1 for node in analysis)
    assert set(aggregate[0].dependency_task_ids) == {node.task.task_id for node in analysis}
    assert {node.task.task_id for node in graph.ready_nodes} == {
        node.task.task_id for node in generation
    }


def test_v2_runtime_changes_do_not_change_graph_scientific_identity() -> None:
    config = load_e01_v2_config(CONFIG)
    payload = config.model_dump(mode="json")
    payload["runtime_profile"]["expected_ram_gib"] = 120
    changed = E01V2Config.model_validate(payload)

    first = compile_e01_v2_graph(config)
    second = compile_e01_v2_graph(changed)

    assert first.graph_id == second.graph_id
    assert tuple(node.task.identity for node in first.nodes) == tuple(
        node.task.identity for node in second.nodes
    )


def test_v2_graph_never_requeues_hash_verified_completed_tasks() -> None:
    config = load_e01_v2_config(CONFIG)
    graph = compile_e01_v2_graph(config)
    generated = graph.ready_nodes[0]
    dependent = next(
        node for node in graph.nodes if generated.task.task_id in node.dependency_task_ids
    )

    resumed = compile_e01_v2_graph(
        config,
        completed_task_ids=frozenset({generated.task.task_id}),
    )

    ready_ids = {node.task.task_id for node in resumed.ready_nodes}
    assert generated.task.task_id not in ready_ids
    assert dependent.task.task_id in ready_ids
    assert len(resumed.nodes) == 101
