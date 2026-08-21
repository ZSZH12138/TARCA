from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import torch
from mypy import api as mypy_api

from tarca.contracts import (
    ConceptBatch,
    ConceptExtractor,
    ForecastDistribution,
    ForecastPredictor,
    LeakageAudit,
    MechanisticModelAdapter,
    WindowBatch,
    validate_concept_batch,
    validate_forecast_distribution,
    validate_window_batch,
)


def _batch() -> WindowBatch:
    start = datetime(2026, 8, 21, tzinfo=UTC)
    return WindowBatch(
        x=torch.tensor([[[1.0], [2.0]]], requires_grad=True),
        y=torch.tensor([[[3.0]]]),
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=None,
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=("window-0",),
        input_feature_names=("load",),
        target_names=("load",),
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=(start,),
        feature_end=(start + timedelta(hours=1),),
        prediction_start=(start + timedelta(hours=2),),
        label_end=(start + timedelta(hours=2),),
        forecast_time=((start + timedelta(hours=2),),),
        metadata={"partition": "TRAIN"},
    )


class _FakePredictor:
    adapter_name = "fake-predictor"
    model_hash = "a" * 64
    is_frozen = True

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        assert batch.y is not None
        return ForecastDistribution(
            mean=batch.y,
            scale=torch.ones_like(batch.y),
            quantiles={},
            logits=None,
            samples=None,
            window_id=batch.window_id,
            target_names=batch.target_names,
        )


class _FakeConceptExtractor:
    def compute(self, batch: WindowBatch) -> ConceptBatch:
        return ConceptBatch(
            values=batch.x[:, -1, :],
            valid_mask=torch.ones((batch.x.shape[0], 1), dtype=torch.bool),
            names=("last-value",),
            window_id=batch.window_id,
            computed_from_history_only=True,
            definition_version="1.0.0",
        )

    def leakage_audit(self, batch: WindowBatch) -> LeakageAudit:
        validate_window_batch(batch)
        return LeakageAudit(passed=True, findings=())


def test_predictor_and_concept_protocols_support_behavior_checked_fakes() -> None:
    batch = _batch()
    predictor: ForecastPredictor = _FakePredictor()
    extractor: ConceptExtractor = _FakeConceptExtractor()
    x_identity = (batch.x, batch.x.data_ptr(), batch.x.requires_grad)

    forecast = predictor.predict_distribution(batch)
    concepts = extractor.compute(batch)

    assert validate_forecast_distribution(forecast) is forecast
    assert validate_concept_batch(concepts) is concepts
    assert extractor.leakage_audit(batch).passed is True
    assert (batch.x, batch.x.data_ptr(), batch.x.requires_grad) == x_identity


def test_mechanistic_adapter_is_not_implied_by_plain_predictor() -> None:
    predictor = cast(ForecastPredictor, _FakePredictor())

    assert isinstance(predictor, ForecastPredictor)
    assert not isinstance(predictor, MechanisticModelAdapter)


def test_static_protocol_rejects_wrong_prediction_return_type(tmp_path: Path) -> None:
    source = """
from tarca.contracts import ForecastDistribution, ForecastPredictor, WindowBatch

class BadPredictor:
    adapter_name = 'bad'
    model_hash = 'a' * 64
    is_frozen = True
    def predict_distribution(self, batch: WindowBatch) -> str:
        return 'wrong'

def accepts(value: ForecastPredictor) -> None:
    pass

accepts(BadPredictor())
"""

    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\nstrict = True\n", encoding="utf-8")
    stdout, stderr, status = mypy_api.run(["--config-file", str(config), "-c", source])

    assert status == 1
    assert "incompatible type" in stdout.lower()
    assert stderr == ""
