from __future__ import annotations


class DeviceContractError(RuntimeError):
    """Raised when a model and its runtime tensors do not share one device."""
