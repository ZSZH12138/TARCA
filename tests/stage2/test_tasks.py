from pathlib import Path

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.stage2.config import load_stage2_config
from tarca.stage2.tasks import Stage2GraphInputs, compile_stage2_graph, compile_stage2_ready

ROOT = Path(__file__).resolve().parents[2]


def _ref(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        artifact_type=name.upper(),
        content_hash=canonical_json_hash(name),
        schema_version="1.0.0",
        relative_path=f"artifacts/{name}.json",
    )


def _inputs() -> Stage2GraphInputs:
    return Stage2GraphInputs(
        stage1b_manifest=_ref("stage1b"),
        e01_receipt=_ref("e01"),
        source_capsule=_ref("source"),
        formal_access_event_count=0,
    )


def test_stage2_graph_has_six_independent_large_gpu_training_tasks() -> None:
    graph = compile_stage2_graph(
        load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml"), _inputs()
    )
    gpu_train = tuple(node for node in graph.nodes if node.phase == "NEURAL_TRAIN")
    assert len(gpu_train) == 6
    assert all(node.resource_request.gpu_count == 1 for node in gpu_train)
    assert all(node.resource_request.cpu_threads == 4 for node in gpu_train)
    assert len({node.identity.seed for node in gpu_train}) == 3


def test_stage2_graph_never_mentions_formal_partitions_or_seeds() -> None:
    config = load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml")
    graph = compile_stage2_graph(config, _inputs())
    serialized = repr(graph)
    assert "TEST_SEEN_REGIME" not in serialized
    assert "TEST_UNSEEN_REGIME" not in serialized
    assert all(str(seed) not in serialized for seed in (1729, 2718, 3141, 5772, 8111))


def test_stage2_ready_manifest_is_never_rerun_and_uses_exact_artifacts() -> None:
    graph = compile_stage2_graph(
        load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml"), _inputs()
    )
    manifest = compile_stage2_ready(graph, {})
    assert manifest.completed_task_policy == "NEVER_RERUN"
    assert {task.phase for task in manifest.tasks} == {"SOURCE_VERIFY", "UPSTREAM_VERIFY"}
    source_tasks = tuple(task for task in manifest.tasks if task.phase == "SOURCE_VERIFY")
    assert all(task.inputs == (_inputs().source_capsule,) for task in source_tasks)
