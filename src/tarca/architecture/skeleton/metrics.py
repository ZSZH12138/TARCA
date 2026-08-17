"""Pure metric-consumer skeleton."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.forecast import ForecastDistribution
from tarca.contracts.manifests import MetricRecord


class MetricComputer(Protocol):
    def compute(
        self, forecast: ForecastDistribution, target: Sequence[float]
    ) -> tuple[MetricRecord, ...]: ...


def forecasting_metrics(
    forecast: ForecastDistribution, target: Sequence[float]
) -> tuple[MetricRecord, ...]:
    raise UnimplementedCapabilityError("metrics.forecasting_metrics")


def abstraction_metrics(effect: object) -> tuple[MetricRecord, ...]:
    raise UnimplementedCapabilityError("metrics.abstraction_metrics")


def localization_metrics(result: object) -> tuple[MetricRecord, ...]:
    raise UnimplementedCapabilityError("metrics.localization_metrics")


def calibration_metrics(forecast: ForecastDistribution) -> tuple[MetricRecord, ...]:
    raise UnimplementedCapabilityError("metrics.calibration_metrics")


def statistical_test(records: Sequence[MetricRecord]) -> Mapping[str, float]:
    raise UnimplementedCapabilityError("metrics.statistical_test")


def validate_metric_records(records: Sequence[MetricRecord]) -> tuple[MetricRecord, ...]:
    normalized = tuple(records)
    if not all(isinstance(record, MetricRecord) for record in normalized):
        raise TypeError("records must contain MetricRecord values")
    return normalized
