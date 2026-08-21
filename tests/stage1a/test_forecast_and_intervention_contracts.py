from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import torch
from pydantic import ValidationError

from tarca.contracts import (
    CONTRACT_SCHEMA_VERSION,
    PROTOCOL_ID,
    ConceptBatch,
    ConceptSpec,
    ForecastDistribution,
    InterventionKind,
    InterventionPair,
    InterventionPairSet,
    InterventionSite,
    InterventionSpec,
    MetricContext,
    MetricRecord,
    RegimeRelation,
    ResolvedInterventionPairBatch,
    SplitPartition,
    WindowBatch,
    validate_concept_batch,
    validate_forecast_distribution,
    validate_intervention_pair_set,
    validate_intervention_site,
    validate_intervention_spec,
    validate_resolved_intervention_pair_batch,
)

HASH_A = "a" * 64


def _forecast() -> ForecastDistribution:
    mean = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
    return ForecastDistribution(
        mean=mean,
        scale=torch.ones_like(mean),
        quantiles={0.1: mean - 1.0, 0.9: mean + 1.0},
        logits=None,
        samples=torch.stack((mean, mean + 0.25)),
        window_id=("window-0", "window-1"),
        target_names=("load",),
    )


def test_forecast_validation_preserves_mean_identity() -> None:
    forecast = _forecast()
    before = (
        forecast.mean,
        forecast.mean.data_ptr(),
        forecast.mean.dtype,
        forecast.mean.device,
        forecast.mean.requires_grad,
    )

    assert validate_forecast_distribution(forecast) is forecast
    assert (
        forecast.mean,
        forecast.mean.data_ptr(),
        forecast.mean.dtype,
        forecast.mean.device,
        forecast.mean.requires_grad,
    ) == before


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"scale": torch.zeros((2, 2, 1))}, "strictly positive"),
        ({"quantiles": {0.0: torch.ones((2, 2, 1))}}, "between 0 and 1"),
        (
            {
                "quantiles": {
                    0.1: torch.full((2, 2, 1), 2.0),
                    0.9: torch.full((2, 2, 1), 1.0),
                }
            },
            "must not cross",
        ),
        ({"scale": torch.ones((2, 2, 1), dtype=torch.float64)}, "dtype"),
        ({"logits": torch.tensor(1.0)}, "rank-4"),
        ({"samples": torch.tensor(1.0)}, "rank-4"),
    ),
)
def test_forecast_validation_rejects_invalid_distribution(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_forecast_distribution(replace(_forecast(), **change))


def test_concept_contracts_validate_range_shape_and_mask() -> None:
    spec = ConceptSpec(
        name="trend",
        definition_version="1.0.0",
        required_history=2,
        history_only=True,
        source_kind="ANALYTIC",
        intervention_semantics="additive trend delta",
        valid_range=(-10.0, 10.0),
        expected_effect_components=("mean",),
        definition_hash=HASH_A,
    )
    values = torch.tensor([[1.0], [2.0]], requires_grad=True)
    batch = ConceptBatch(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool),
        names=(spec.name,),
        window_id=("window-0", "window-1"),
        computed_from_history_only=True,
        definition_version=spec.definition_version,
    )

    assert validate_concept_batch(batch) is batch
    assert batch.values is values
    with pytest.raises(ValueError, match="bool"):
        validate_concept_batch(replace(batch, valid_mask=torch.ones_like(values)))
    with pytest.raises(ValidationError, match="valid_range"):
        ConceptSpec(**{**spec.model_dump(), "valid_range": (10.0, -10.0)})


def test_intervention_site_rejects_duplicate_or_out_of_range_axes() -> None:
    site = InterventionSite(
        site_name="encoder.block.0",
        layer=0,
        tensor_rank=3,
        batch_axis=0,
        variable_axis=1,
        patch_axis=None,
        feature_axis=2,
        shape_template=(None, None, 16),
    )

    assert validate_intervention_site(site) is site
    with pytest.raises(ValueError, match="axes must be unique"):
        validate_intervention_site(replace(site, feature_axis=1))
    with pytest.raises(ValueError, match="within tensor rank"):
        validate_intervention_site(replace(site, feature_axis=3))


def test_intervention_spec_enforces_subspace_rules_without_mutation() -> None:
    basis = torch.eye(2, requires_grad=True)
    spec = InterventionSpec(
        site_name="encoder.block.0",
        layer=0,
        variable_index=0,
        patch_index=None,
        lag=1,
        subspace_basis=basis,
        intervention_kind=InterventionKind.SUBSPACE_SWAP,
    )

    assert validate_intervention_spec(spec, orthogonality_tolerance=1e-6) is spec
    assert spec.subspace_basis is basis
    with pytest.raises(ValueError, match="must not carry"):
        validate_intervention_spec(
            replace(spec, intervention_kind=InterventionKind.FULL_SWAP),
            orthogonality_tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="orthonormal"):
        validate_intervention_spec(
            replace(spec, subspace_basis=torch.tensor([[1.0, 1.0], [0.0, 1.0]])),
            orthogonality_tolerance=1e-6,
        )


def test_intervention_pair_and_metric_records_reject_invalid_scientific_values() -> None:
    pair_values = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "pair_id": "c" * 64,
        "partition": SplitPartition.TRAIN,
        "base_window_id": "window-0",
        "source_window_id": "window-1",
        "concept_name": "trend",
        "regime_relation": RegimeRelation.SAME,
        "matching_distance": 0.25,
        "concept_delta": 1.5,
    }
    assert InterventionPair.model_validate(pair_values).pair_id == "c" * 64
    with pytest.raises(ValidationError, match="must differ"):
        InterventionPair.model_validate({**pair_values, "source_window_id": "window-0"})
    with pytest.raises(ValidationError, match="non-negative"):
        InterventionPair.model_validate({**pair_values, "matching_distance": -0.1})

    context = MetricContext(
        experiment_id="stage1a",
        run_id="fixture",
        split=SplitPartition.TRAIN,
        data_hash=HASH_A,
        model_id=None,
        protocol_id=PROTOCOL_ID,
        gate_scope=None,
    )
    assert context.protocol_id == PROTOCOL_ID
    with pytest.raises(ValidationError, match="finite"):
        MetricRecord(
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            split=context.split,
            metric_name="mae",
            value=float("inf"),
            regime=None,
            horizon=None,
            concept=None,
        )


def test_intervention_pair_set_requires_unique_stable_ids() -> None:
    pair_id = "c" * 64
    pair_set = InterventionPairSet(pair_ids=(pair_id,), source_label="train-pairs")

    assert validate_intervention_pair_set(pair_set) is pair_set
    with pytest.raises(ValueError, match="must be unique"):
        validate_intervention_pair_set(
            InterventionPairSet(pair_ids=(pair_id, pair_id), source_label="train-pairs")
        )


def _pair_window_batch() -> WindowBatch:
    start = datetime(2026, 8, 21, tzinfo=UTC)
    return WindowBatch(
        x=torch.tensor([[[1.0]], [[2.0]]]),
        y=torch.tensor([[[2.0]], [[3.0]]]),
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=None,
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=("window-0", "window-1"),
        input_feature_names=("load",),
        target_names=("load",),
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=(start, start),
        feature_end=(start + timedelta(hours=1),) * 2,
        prediction_start=(start + timedelta(hours=2),) * 2,
        label_end=(start + timedelta(hours=2),) * 2,
        forecast_time=((start + timedelta(hours=2),),) * 2,
        metadata={"partition": "TRAIN"},
    )


def test_resolved_pair_batch_validates_direction_indices_and_partition() -> None:
    pair = InterventionPair(
        schema_version=CONTRACT_SCHEMA_VERSION,
        pair_id="d" * 64,
        partition=SplitPartition.TRAIN,
        base_window_id="window-0",
        source_window_id="window-1",
        concept_name="trend",
        regime_relation=RegimeRelation.SAME,
        matching_distance=0.25,
        concept_delta=1.0,
    )
    windows = _pair_window_batch()
    resolved = ResolvedInterventionPairBatch(
        pairs=(pair,),
        base=windows,
        source=windows,
        base_row_for_pair=(0,),
        source_row_for_pair=(1,),
        dataset_hash=HASH_A,
    )

    assert validate_resolved_intervention_pair_batch(resolved) is resolved
    assert resolved.base.x is windows.x
    with pytest.raises(ValueError, match="source window direction"):
        validate_resolved_intervention_pair_batch(
            replace(resolved, source_row_for_pair=(0,))
        )
