from __future__ import annotations

import torch
from torch import Tensor


def recover_lag(effect_curve: Tensor) -> int:
    """Return the one-based horizon with the largest absolute effect."""

    if not isinstance(effect_curve, Tensor) or effect_curve.ndim != 1 or effect_curve.numel() == 0:
        raise ValueError("effect curve must be a nonempty rank-1 tensor")
    if not effect_curve.is_floating_point() or not bool(torch.isfinite(effect_curve).all()):
        raise ValueError("effect curve must be finite and floating")
    return int(torch.argmax(effect_curve.abs()).item()) + 1
