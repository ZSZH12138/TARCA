"""Concept extraction skeleton."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.concepts import ConceptBatch
from tarca.contracts.data import WindowBatch
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.forecast import ForecastDistribution
from tarca.contracts.future import ConceptIntervention, LeakageAudit


class ConceptExtractor(Protocol):
    def compute(self, batch: WindowBatch) -> ConceptBatch: ...

    def leakage_audit(self, batch: WindowBatch) -> LeakageAudit: ...


class HighLevelInterventionModel(Protocol):
    def intervene(
        self,
        base: ForecastDistribution,
        source: ForecastDistribution,
        concept_intervention: ConceptIntervention,
    ) -> ForecastDistribution: ...


def compute(batch: WindowBatch) -> ConceptBatch:
    raise UnimplementedCapabilityError("concepts.compute")


def leakage_audit(batch: WindowBatch) -> LeakageAudit:
    raise UnimplementedCapabilityError("concepts.leakage_audit")


def validate_concept_batch(batch: ConceptBatch) -> ConceptBatch:
    if not isinstance(batch, ConceptBatch):
        raise TypeError("batch must be ConceptBatch")
    return batch
