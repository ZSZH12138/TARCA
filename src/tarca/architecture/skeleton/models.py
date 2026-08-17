"""Model capability skeleton; existing model implementations remain untouched."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torch import Tensor

from tarca.contracts.adapters import ForecastPredictor
from tarca.contracts.data import WindowBatch
from tarca.contracts.forecast import ForecastDistribution
from tarca.contracts.interventions import InterventionSite, InterventionSpec


class MechanisticModelAdapter(Protocol):
    @property
    def adapter_name(self) -> str: ...

    @property
    def model_hash(self) -> str: ...

    @property
    def is_frozen(self) -> bool: ...

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution: ...

    def list_intervention_sites(self) -> tuple[InterventionSite, ...]: ...

    def capture(
        self, batch: WindowBatch, sites: tuple[InterventionSite, ...]
    ) -> Mapping[str, Tensor]: ...

    def intervene(
        self, base: WindowBatch, source: WindowBatch, spec: InterventionSpec
    ) -> ForecastDistribution: ...


class ModelRegistry(Protocol):
    def resolve_predictor(self, model_id: str) -> ForecastPredictor: ...

    def resolve_mechanistic_adapter(self, model_id: str) -> MechanisticModelAdapter: ...


def validate_predictor(predictor: ForecastPredictor) -> ForecastPredictor:
    if not callable(getattr(predictor, "predict_distribution", None)):
        raise TypeError("predictor must expose predict_distribution")
    return predictor
