"""Experiment compiler and gate skeleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tarca.contracts.common import GateDecision, TaskResult
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import ExperimentSpec, ExperimentSummary, TaskManifest
from tarca.contracts.manifests import MetricRecord


class ExperimentCompiler(Protocol):
    def compile_experiment(self, spec: ExperimentSpec) -> TaskManifest: ...

    def validate_experiment(self, spec: ExperimentSpec) -> None: ...


def compile_experiment(spec: ExperimentSpec) -> TaskManifest:
    raise UnimplementedCapabilityError("experiments.compile_experiment")


def validate_experiment(spec: ExperimentSpec) -> None:
    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be ExperimentSpec")


def reduce_results(results: Sequence[TaskResult]) -> ExperimentSummary:
    raise UnimplementedCapabilityError("experiments.reduce_results")


def evaluate_gate(records: Sequence[MetricRecord]) -> GateDecision:
    raise UnimplementedCapabilityError("experiments.evaluate_gate")
