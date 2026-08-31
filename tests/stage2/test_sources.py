from __future__ import annotations

from pathlib import Path

from tarca.stage2.config import load_stage2_config
from tarca.stage2.sources import dlinear_model_config, stage2_sources

STAGE2_CONFIG = Path("configs/stage2/stage2_v1.yaml")


def test_stage2_source_set_and_dlinear_identity_are_exact() -> None:
    config = load_stage2_config(STAGE2_CONFIG)

    sources = stage2_sources(config)
    dlinear = dlinear_model_config(config, dimension=8)

    assert tuple(source.source_id for source in sources) == (
        "dlinear",
        "itransformer",
        "patchtst",
        "scoring_rules_l96",
    )
    assert dlinear.sequence_length == 64
    assert dlinear.prediction_length == 24
    assert dlinear.dimension == 8
    assert dlinear.asset_sha256 == (
        "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc"
    )

