from __future__ import annotations

from datetime import UTC, datetime, timedelta

import torch
from tests.fakes.fakes import (
    FakeArtifactStore,
    FakeConceptExtractor,
    FakeHighLevelInterventionModel,
    FakePredictor,
)

from tarca.contracts.common import GateDecision, GateStatus
from tarca.contracts.data import WindowBatch
from tarca.contracts.effects import EffectSignature
from tarca.contracts.future import ExperimentSummary, LocalizationResult, LocalizationStage
from tarca.contracts.manifests import MetricRecord
from tarca.contracts.types import SplitPartition


def _batch() -> WindowBatch:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    prediction = start + timedelta(hours=2)
    return WindowBatch(
        x=torch.zeros((1, 2, 1)),
        y=torch.zeros((1, 2, 1)),
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=None,
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=("window-1",),
        input_feature_names=("x",),
        target_names=("y",),
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=(start,),
        feature_end=(start + timedelta(hours=1),),
        prediction_start=(prediction,),
        label_end=(prediction + timedelta(hours=1),),
        forecast_time=((prediction, prediction + timedelta(hours=1)),),
        metadata={},
    )


def test_contract_only_end_to_end_smoke_does_not_claim_scientific_success() -> None:
    batch = _batch()
    factual = FakePredictor().predict_distribution(batch)
    concepts = FakeConceptExtractor().compute(batch)
    intervened = FakeHighLevelInterventionModel().intervene(
        factual,
        factual,
        concept_intervention=__import__(
            "tarca.contracts.future", fromlist=["ConceptIntervention"]
        ).ConceptIntervention("fake_concept", 0.0),
    )
    effect = EffectSignature(
        delta_mean=intervened.mean - factual.mean,
        delta_scale=None,
        delta_quantiles={0.5: intervened.quantiles[0.5] - factual.quantiles[0.5]},
        horizon=2,
    )
    localization = LocalizationResult(
        stage=LocalizationStage.COARSE_LAYER, candidate_ids=("contract-only",)
    )
    metric = MetricRecord(
        experiment_id="contract-smoke",
        run_id="contract-smoke-run",
        split=SplitPartition.TEST,
        metric="contract_smoke_only",
        value=0.0,
        regime=None,
        horizon=2,
        concept="fake_concept",
    )
    summary = ExperimentSummary(experiment_id="contract-smoke", results=())
    gate = GateDecision(
        gate_id="contract-smoke-gate",
        status=GateStatus.BLOCKED,
        rationale="Contract-only smoke; no scientific algorithm or result is asserted",
    )
    artifact = FakeArtifactStore().publish_atomic(summary, "experiment_summary")
    assert concepts.computed_from_history_only is True
    assert effect.horizon == 2
    assert localization.stage is LocalizationStage.COARSE_LAYER
    assert metric.metric == "contract_smoke_only"
    assert gate.status is GateStatus.BLOCKED
    assert artifact.artifact_type == "experiment_summary"
