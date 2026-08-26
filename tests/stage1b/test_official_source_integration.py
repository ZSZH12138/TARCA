from __future__ import annotations

from pathlib import Path

import pytest

from tarca.stage1b.config import load_world_suite
from tarca.stage1b.reproduction import (
    ReproductionKind,
    load_reproduction_suite,
    run_reproduction,
)
from tarca.stage1b.sources import (
    MaterializedSources,
    SubprocessGitRunner,
    materialize_source,
    verify_materialized_source,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CACHE = REPOSITORY_ROOT / "third_party/stage1b"


def _materialized_sources() -> MaterializedSources:
    if not SOURCE_CACHE.is_dir():
        pytest.skip("run scripts/materialize_stage1b_sources.py before official integration tests")
    worlds = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    runner = SubprocessGitRunner.discover()
    return MaterializedSources(
        receipts=tuple(
            materialize_source(source, SOURCE_CACHE, runner) for source in worlds.sources
        )
    )


@pytest.mark.official_source
def test_all_registered_official_reproductions_match_pinned_upstream() -> None:
    suite = load_reproduction_suite(
        REPOSITORY_ROOT / "configs/stage1b/official_reproduction_v2.yaml"
    )
    sources = _materialized_sources()

    receipts = tuple(
        run_reproduction(
            case,
            sources,
            input_root=REPOSITORY_ROOT,
        )
        for case in suite.cases
    )

    assert len(receipts) == 6
    assert all(receipt.passed for receipt in receipts)
    assert max(
        receipt.maximum_absolute_error
        for receipt, case in zip(receipts, suite.cases, strict=True)
        if case.kind is ReproductionKind.MODEL_FORWARD
    ) <= 1e-6
    assert all(
        verify_materialized_source(receipt, SOURCE_CACHE) == receipt.checkout_root
        for receipt in sources.receipts
    )
