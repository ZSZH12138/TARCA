"""Intervention adapter and engine skeleton."""

from __future__ import annotations

from typing import Protocol

from tarca.architecture.skeleton.models import MechanisticModelAdapter
from tarca.contracts.concepts import ConceptBatch
from tarca.contracts.data import WindowBatch
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import InterventionPairSet, InterventionResult
from tarca.contracts.interventions import InterventionSpec


class InterventionAdapter(Protocol):
    def build_pairs(self, base: ConceptBatch, source: ConceptBatch) -> InterventionPairSet: ...


class InterventionEngine(Protocol):
    def apply_intervention(
        self,
        pair_id: str,
        base: WindowBatch,
        source: WindowBatch,
        spec: InterventionSpec,
        model: MechanisticModelAdapter,
    ) -> InterventionResult: ...


def build_pairs(base: ConceptBatch, source: ConceptBatch) -> InterventionPairSet:
    raise UnimplementedCapabilityError("interventions.build_pairs")


def apply_intervention(
    pair_id: str,
    base: WindowBatch,
    source: WindowBatch,
    spec: InterventionSpec,
    model: MechanisticModelAdapter,
) -> InterventionResult:
    raise UnimplementedCapabilityError("interventions.apply_intervention")


def validate_intervention_result(result: InterventionResult) -> InterventionResult:
    if not isinstance(result, InterventionResult):
        raise TypeError("result must be InterventionResult")
    return result
