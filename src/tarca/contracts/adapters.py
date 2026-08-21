from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from torch import Tensor

from .data import WindowBatch
from .forecasts import ForecastDistribution
from .interventions import InterventionSite, InterventionSpec


@runtime_checkable
class ForecastPredictor(Protocol):
    @property
    def adapter_name(self) -> str: ...

    @property
    def model_hash(self) -> str: ...

    @property
    def is_frozen(self) -> bool: ...

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution: ...


@runtime_checkable
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
        self,
        batch: WindowBatch,
        sites: tuple[InterventionSite, ...],
    ) -> Mapping[str, Tensor]: ...

    def intervene(
        self,
        base: WindowBatch,
        source: WindowBatch,
        spec: InterventionSpec,
    ) -> ForecastDistribution: ...
