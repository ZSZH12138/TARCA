from pathlib import Path

from tarca.e02.config import load_e02_config
from tarca.e02.jobs import e02_executor_registry
from tarca.e02.runner import e02_scientific_plan_hash
from tarca.e02.tasks import FrozenStage2Input, compile_e02_graph
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
