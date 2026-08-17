"""Typed effect signatures; no calibration or scientific fitting is provided."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import Tensor


@dataclass(frozen=True, slots=True)
class EffectSignature:
    """Forecast-space intervention effect signature."""

    delta_mean: Tensor
    delta_scale: Tensor | None
    delta_quantiles: Mapping[float, Tensor]
    horizon: int

    def __post_init__(self) -> None:
        if not isinstance(self.delta_mean, Tensor):
            raise TypeError("delta_mean must be a torch.Tensor")
        if self.delta_scale is not None and not isinstance(self.delta_scale, Tensor):
            raise TypeError("delta_scale must be a torch.Tensor or None")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not isinstance(self.delta_quantiles, Mapping):
            raise TypeError("delta_quantiles must be a mapping")
        for quantile, value in self.delta_quantiles.items():
            if not isinstance(quantile, float) or not 0.0 <= quantile <= 1.0:
                raise ValueError("quantile keys must be floats in [0, 1]")
            if not isinstance(value, Tensor):
                raise TypeError("delta_quantiles values must be tensors")
        object.__setattr__(self, "delta_quantiles", MappingProxyType(dict(self.delta_quantiles)))
