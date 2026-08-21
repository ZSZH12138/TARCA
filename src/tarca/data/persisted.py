from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from tarca.contracts.base import canonical_json_bytes
from tarca.contracts.data import WindowBatch, validate_window_batch

from .payload import PersistedPayloadFile, PersistedWindowMetadata


class PayloadBackend(Protocol):
    def read_bytes(self, path: Path) -> bytes: ...


class LocalPayloadBackend:
    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()


def load_window_batch(
    files: dict[str, PersistedPayloadFile],
    payloads: dict[str, bytes],
) -> WindowBatch:
    metadata = _load_metadata(files["metadata"], payloads["metadata"])

    def tensor(role: str) -> torch.Tensor | None:
        if role not in files:
            return None
        array = _load_npy_bytes(payloads[role])
        if array.dtype.hasobject:
            raise ValueError("object dtype is forbidden; arrays require allow_pickle=False")
        return torch.from_numpy(array)

    x = tensor("x")
    if x is None:  # pragma: no cover - guaranteed by payload contract
        raise ValueError("persisted partition has no x array")
    batch = WindowBatch(
        x=x,
        y=tensor("y"),
        observed_covariates=tensor("observed_covariates"),
        known_future_covariates=tensor("known_future_covariates"),
        x_observed_mask=tensor("x_observed_mask"),
        y_observed_mask=tensor("y_observed_mask"),
        observed_covariates_mask=tensor("observed_covariates_mask"),
        known_future_covariates_mask=tensor("known_future_covariates_mask"),
        regime=tensor("regime"),
        window_id=metadata.window_id,
        input_feature_names=metadata.input_feature_names,
        target_names=metadata.target_names,
        observed_covariate_names=metadata.observed_covariate_names,
        known_future_covariate_names=metadata.known_future_covariate_names,
        feature_start=metadata.feature_start,
        feature_end=metadata.feature_end,
        prediction_start=metadata.prediction_start,
        label_end=metadata.label_end,
        forecast_time=metadata.forecast_time,
        metadata=MappingProxyType(dict(metadata.metadata)),
    )
    return validate_window_batch(batch)


def _load_npy_bytes(payload: bytes) -> NDArray[Any]:
    value = np.load(BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError("persisted array must be a single .npy ndarray")
    if value.dtype.hasobject:
        raise ValueError("object dtype is forbidden; arrays require allow_pickle=False")
    return value


def _load_metadata(descriptor: PersistedPayloadFile, payload: bytes) -> PersistedWindowMetadata:
    metadata = PersistedWindowMetadata.model_validate_json(payload)
    if canonical_json_bytes(metadata) + b"\n" != payload:
        raise ValueError(f"window metadata is not canonical: {descriptor.relative_path}")
    return metadata
