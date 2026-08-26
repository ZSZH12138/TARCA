from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tarca.contracts import ArtifactRef
from tarca.stage1b.compiler import (
    compile_ready_manifest,
    compile_stage1b_graph,
    repository_v2_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _artifact(node_id: str, artifact_type: str, marker: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"{node_id}-output",
        artifact_type=artifact_type,
        content_hash=marker * 64,
        schema_version="2.0.0",
        relative_path=f"stage1b/tasks/{marker}.json",
    )


def test_primary_training_graph_contains_twelve_gpu_nodes() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    training = tuple(node for node in graph.nodes if node.phase == "NEURAL_TRAIN")
    assert len(training) == 12
    assert all(node.resource_request.gpu_count == 1 for node in training)
    assert {
        (node.identity.data_id, node.identity.model_id, node.identity.seed) for node in training
    } == {
        (world, model, seed)
        for world in ("lorenz96_f10_v2", "lorenz96_twoscale_v2")
        for model in ("patchtst_reference", "itransformer_reference")
        for seed in (104729, 130363, 155921)
    }


def test_graph_is_acyclic_prehashed_and_excludes_formal_experiments() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    positions = {node.node_id: index for index, node in enumerate(graph.nodes)}
    assert len(positions) == len(graph.nodes)
    assert all(
        positions[dependency] < positions[node.node_id]
        for node in graph.nodes
        for dependency in node.dependency_ids
    )
    assert all(node.identity.task_id == node.node_id for node in graph.nodes)
    serialized = repr(graph)
    assert "E01" not in serialized and "E02" not in serialized


def test_ready_manifest_materializes_only_verified_dependency_outputs() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    roots = tuple(node for node in graph.nodes if not node.dependency_ids)
    initial = compile_ready_manifest(graph, {})
    assert {task.task_id for task in initial.tasks} == {node.node_id for node in roots}
    assert initial.completed_task_policy == "NEVER_RERUN"

    completed = {
        node.node_id: _artifact(node.node_id, node.output_artifact_type, hex(index + 1)[2:])
        for index, node in enumerate(roots)
    }
    next_manifest = compile_ready_manifest(graph, completed)
    assert all(task.task_id not in completed for task in next_manifest.tasks)
    assert all(
        tuple(ref.artifact_type for ref in task.inputs)
        == next(node.expected_input_types for node in graph.nodes if node.node_id == task.task_id)
        for task in next_manifest.tasks
    )


def test_ready_manifest_rejects_mismatched_or_unknown_outputs() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    root = next(node for node in graph.nodes if not node.dependency_ids)
    with pytest.raises(ValueError, match="artifact type"):
        compile_ready_manifest(graph, {root.node_id: _artifact(root.node_id, "wrong", "a")})
    with pytest.raises(ValueError, match="unknown"):
        compile_ready_manifest(graph, {"unknown": _artifact("unknown", "wrong", "b")})


def test_code_change_changes_graph_identity_but_resource_change_does_not() -> None:
    inputs = repository_v2_inputs(REPOSITORY_ROOT)
    original = compile_stage1b_graph(inputs)
    changed_code = compile_stage1b_graph(replace(inputs, code_sha256="f" * 64))
    assert changed_code.graph_id != original.graph_id
    first = original.nodes[0]
    resource_only = replace(
        first,
        resource_request=first.resource_request.model_copy(update={"cpu_threads": 27}),
    )
    assert resource_only.identity == first.identity
