from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.official_worlds import (
    baseline_concept_schedule,
    build_official_world,
    concept_pair_schedules,
)
from tarca.stage1b.oracle import OraclePairRequest, paired_rollout
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
from tarca.stage1b.worlds import SimulationRequest

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


@pytest.fixture(scope="module")
def official_sources() -> MaterializedSources:
    return _materialized_sources()


@pytest.mark.official_source
def test_all_registered_official_reproductions_match_pinned_upstream(
    official_sources: MaterializedSources,
) -> None:
    suite = load_reproduction_suite(
        REPOSITORY_ROOT / "configs/stage1b/official_reproduction_v2.yaml"
    )
    receipts = tuple(
        run_reproduction(
            case,
            official_sources,
            input_root=REPOSITORY_ROOT,
        )
        for case in suite.cases
    )

    assert len(receipts) == 6
    assert all(receipt.passed for receipt in receipts)
    assert (
        max(
            receipt.maximum_absolute_error
            for receipt, case in zip(receipts, suite.cases, strict=True)
            if case.kind is ReproductionKind.MODEL_FORWARD
        )
        <= 1e-6
    )
    assert all(
        verify_materialized_source(receipt, SOURCE_CACHE) == receipt.checkout_root
        for receipt in official_sources.receipts
    )


@pytest.mark.official_source
@pytest.mark.filterwarnings("error:Excess work done on this call")
@pytest.mark.parametrize(
    "world_id",
    [
        "control_var_v2",
        "lorenz96_f10_v2",
        "lorenz96_f40_v2",
        "lorenz96_twoscale_v2",
        "gvar_predator_prey_v2",
        "corrected_cml_v2",
    ],
)
def test_official_oracle_drivers_are_bitwise_identity_replay(
    world_id: str,
    official_sources: MaterializedSources,
) -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    config = suite.world(world_id)
    request = SimulationRequest(
        seed=104729,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=config.regimes[0].regime_id,
        length=4,
    )
    schedule = baseline_concept_schedule(config, request)
    driver = build_official_world(config, official_sources)

    pair = paired_rollout(
        driver,
        OraclePairRequest(
            base=request,
            factual_schedule=schedule,
            counterfactual_schedule=schedule,
            changed_concept="identity",
        ),
    )

    assert torch.equal(pair.factual.values, pair.counterfactual.values)
    assert torch.equal(pair.factual.future_noise, pair.counterfactual.future_noise)
    assert pair.factual.values.shape == (4, config.dimension)
    assert bool(torch.isfinite(pair.factual.values).all())
    assert all(
        verify_materialized_source(receipt, SOURCE_CACHE) == receipt.checkout_root
        for receipt in official_sources.receipts
    )


@pytest.mark.official_source
@pytest.mark.parametrize(
    ("world_id", "pair_id"),
    [
        ("lorenz96_f10_v2", "trend_primary"),
        ("lorenz96_f10_v2", "scale_primary"),
        ("lorenz96_twoscale_v2", "trend_primary"),
        ("lorenz96_twoscale_v2", "scale_primary"),
    ],
)
def test_primary_official_concept_pairs_share_initial_state_and_noise(
    world_id: str,
    pair_id: str,
    official_sources: MaterializedSources,
) -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    config = suite.world(world_id)
    request = SimulationRequest(
        seed=104729,
        partition=QualificationPartition.QUAL_SEEN,
        regime_id=config.regimes[0].regime_id,
        length=4,
    )
    factual, counterfactual, changed_concept = concept_pair_schedules(
        config,
        request,
        pair_id,
    )

    pair = paired_rollout(
        build_official_world(config, official_sources),
        OraclePairRequest(
            base=request,
            factual_schedule=factual,
            counterfactual_schedule=counterfactual,
            changed_concept=changed_concept,
        ),
    )

    assert torch.equal(pair.factual.initial_state, pair.counterfactual.initial_state)
    assert torch.equal(pair.factual.future_noise, pair.counterfactual.future_noise)
    assert not torch.equal(pair.factual.values, pair.counterfactual.values)
