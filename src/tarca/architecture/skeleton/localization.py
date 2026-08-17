"""Localization state-machine skeleton."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.effects import EffectSignature
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import LocalizationResult, LocalizationStage


class LocalizationBackend(Protocol):
    def localize(
        self, signature: EffectSignature, stage: LocalizationStage
    ) -> LocalizationResult: ...


def localize(signature: EffectSignature, stage: LocalizationStage) -> LocalizationResult:
    raise UnimplementedCapabilityError("localization.localize")


def validate_localization_result(result: LocalizationResult) -> LocalizationResult:
    if not isinstance(result, LocalizationResult):
        raise TypeError("result must be LocalizationResult")
    return result
