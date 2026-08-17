"""Robustness environment and solver skeleton."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.effects import EffectSignature
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import EnvironmentSpec, RobustnessSpec


class RobustnessSolver(Protocol):
    def fit(self, train: EnvironmentSpec) -> RobustnessSolver: ...

    def transform(
        self, validation: EnvironmentSpec, test: EnvironmentSpec
    ) -> tuple[EnvironmentSpec, EnvironmentSpec]: ...


class EnvironmentDefinition(Protocol):
    def define(self, environment_id: str) -> EnvironmentSpec: ...


def fit_robustness(spec: RobustnessSpec) -> RobustnessSolver:
    raise UnimplementedCapabilityError("robustness.fit")


def transform_robustness(
    signature: EffectSignature, environment: EnvironmentSpec
) -> EffectSignature:
    raise UnimplementedCapabilityError("robustness.transform")


def validate_environment(environment: EnvironmentSpec) -> EnvironmentSpec:
    if not isinstance(environment, EnvironmentSpec):
        raise TypeError("environment must be EnvironmentSpec")
    return environment
