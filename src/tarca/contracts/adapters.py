"""Static protocol for forecast-model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torch import Tensor

from .data import WindowBatch
from .forecast import ForecastDistribution
from .interventions import InterventionSite, InterventionSpec


class ForecastModelAdapter(Protocol):
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
