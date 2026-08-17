"""Effect signature normalization skeleton."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from tarca.contracts.effects import EffectSignature
from tarca.contracts.errors import UnimplementedCapabilityError


class EffectNormalizer(Protocol):
    def fit(self, train_signatures: Iterable[EffectSignature]) -> EffectNormalizer: ...

    def transform(self, signatures: Iterable[EffectSignature]) -> tuple[EffectSignature, ...]: ...


def fit_normalizer(train_signatures: Iterable[EffectSignature]) -> EffectNormalizer:
    raise UnimplementedCapabilityError("effects.fit_normalizer")


def normalize(
    signatures: Iterable[EffectSignature], normalizer: EffectNormalizer
) -> tuple[EffectSignature, ...]:
    raise UnimplementedCapabilityError("effects.normalize")


def validate_effect_signature(signature: EffectSignature) -> EffectSignature:
    if not isinstance(signature, EffectSignature):
        raise TypeError("signature must be EffectSignature")
    return signature
