from pathlib import Path

import pytest

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.e02.config import load_e02_config
from tarca.e02.tasks import FrozenStage2Input, compile_e02_graph

ROOT = Path(__file__).resolve().parents[2]


def _ref(name: str, artifact_type: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        artifact_type=artifact_type,
        content_hash=canonical_json_hash(name),
        schema_version="1.0.0",
        relative_path=f"artifacts/{name}.json",
    )


def test_e02_graph_requires_frozen_stage2_and_grant() -> None:
    config = load_e02_config(ROOT / "configs/e02/e02_v1.yaml")
    with pytest.raises(PermissionError):
        compile_e02_graph(
            config,
            FrozenStage2Input(
                freeze_receipt=_ref("freeze", "STAGE2_FREEZE_RECEIPT"),
                sealed_access_grant=None,
                frozen=False,
            ),
        )


def test_e02_graph_contains_fixed_four_predictors_and_no_selection_task() -> None:
    graph = compile_e02_graph(
        load_e02_config(ROOT / "configs/e02/e02_v1.yaml"),
        FrozenStage2Input(
            freeze_receipt=_ref("freeze", "STAGE2_FREEZE_RECEIPT"),
            sealed_access_grant=_ref("grant", "SEALED_ACCESS_GRANT"),
            frozen=True,
        ),
    )
    predictions = tuple(node for node in graph.nodes if node.phase == "FORMAL_PREDICT")
    assert len(predictions) == 4
    assert {node.identity.model_id for node in predictions} == {
        "STRONGEST_LINEAR",
        "ITRANSFORMER_INIT_0",
        "ITRANSFORMER_INIT_1",
        "ITRANSFORMER_INIT_2",
    }
    assert all(node.phase != "MODEL_SELECT" for node in graph.nodes)
