from pathlib import Path

from tarca.stage2.config import load_stage2_config
from tarca.stage2.jobs import stage2_executor_registry
from tarca.stage2.runner import stage2_scientific_plan_hash
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
