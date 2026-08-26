from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from tarca.contracts import (
    ForecastDistribution,
    WindowBatch,
    validate_forecast_distribution,
    validate_window_batch,
)


def _tensor_bytes(tensor: Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


@dataclass(frozen=True, slots=True)
class _VarCandidate:
    lag: int
    ridge: float
    coefficients: Tensor
    intercept: Tensor
    tune_mse: float


@dataclass(frozen=True, slots=True)
class TunedVAR:
    coefficients: Tensor
    intercept: Tensor
    residual_scale: Tensor
    selected_lag: int
    selected_ridge: float
    target_names: tuple[str, ...]

    @property
    def adapter_name(self) -> str:
        return "TunedVAR"

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def model_hash(self) -> str:
        metadata = json.dumps(
            {
                "adapter": self.adapter_name,
                "lag": self.selected_lag,
                "ridge": self.selected_ridge,
                "target_names": self.target_names,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(metadata)
        for tensor in (self.coefficients, self.intercept, self.residual_scale):
            digest.update(_tensor_bytes(tensor))
        return digest.hexdigest()

    @staticmethod
    def _fit_candidate(
        train_x: Tensor,
        train_y: Tensor,
        tune_x: Tensor,
        tune_y: Tensor,
        lag: int,
        ridge: float,
    ) -> _VarCandidate:
        dimension = train_x.shape[2]
        design = train_x[:, -lag:, :].reshape(train_x.shape[0], lag * dimension)
        response = train_y[:, 0, :]
        ones = torch.ones((design.shape[0], 1), dtype=torch.float64)
        augmented = torch.cat((design.to(torch.float64), ones), dim=1)
        penalty = torch.eye(augmented.shape[1], dtype=torch.float64) * ridge
        penalty[-1, -1] = 0.0
        gram = augmented.transpose(0, 1) @ augmented + penalty
        beta = torch.linalg.solve(gram, augmented.transpose(0, 1) @ response.to(torch.float64))
        coefficients = beta[:-1].transpose(0, 1).contiguous()
        intercept = beta[-1].contiguous()
        tune_mean = TunedVAR._recursive_mean(
            tune_x.to(torch.float64), coefficients, intercept, lag, tune_y.shape[1]
        )
        tune_mse = float(torch.mean((tune_mean - tune_y.to(torch.float64)) ** 2))
        return _VarCandidate(
            lag=lag,
            ridge=ridge,
            coefficients=coefficients,
            intercept=intercept,
            tune_mse=tune_mse,
        )

    @staticmethod
    def _recursive_mean(
        histories: Tensor,
        coefficients: Tensor,
        intercept: Tensor,
        lag: int,
        horizon: int,
    ) -> Tensor:
        rolling = histories
        forecasts: list[Tensor] = []
        for _ in range(horizon):
            features = rolling[:, -lag:, :].reshape(rolling.shape[0], -1)
            next_value = features @ coefficients.transpose(0, 1) + intercept
            forecasts.append(next_value)
            rolling = torch.cat((rolling, next_value.unsqueeze(1)), dim=1)
        return torch.stack(forecasts, dim=1)

    @classmethod
    def fit(
        cls,
        train_x: Tensor,
        train_y: Tensor,
        tune_x: Tensor,
        tune_y: Tensor,
        lag_orders: tuple[int, ...],
        ridge_values: tuple[float, ...],
        target_names: tuple[str, ...],
    ) -> TunedVAR:
        _validate_fit_tensors(train_x, train_y, tune_x, tune_y, target_names)
        candidates = tuple(
            cls._fit_candidate(train_x, train_y, tune_x, tune_y, lag, ridge)
            for lag in lag_orders
            for ridge in ridge_values
            if lag <= train_x.shape[1]
        )
        if not candidates:
            raise ValueError("VAR search contains no lag compatible with the history")
        best = min(candidates, key=lambda item: (item.tune_mse, item.lag, item.ridge))
        tune_mean = cls._recursive_mean(
            tune_x.to(torch.float64),
            best.coefficients,
            best.intercept,
            best.lag,
            tune_y.shape[1],
        )
        residual_scale = (
            (tune_y.to(torch.float64) - tune_mean).std(dim=0, unbiased=False).clamp_min(1e-4)
        )
        return cls(
            coefficients=best.coefficients,
            intercept=best.intercept,
            residual_scale=residual_scale,
            selected_lag=best.lag,
            selected_ridge=best.ridge,
            target_names=target_names,
        )

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        validate_window_batch(batch)
        if batch.x.shape[1] < self.selected_lag:
            raise ValueError("WindowBatch history is shorter than the selected VAR lag")
        if batch.x.shape[2] != self.coefficients.shape[0]:
            raise ValueError("WindowBatch dimension does not match the VAR model")
        horizon = batch.y.shape[1] if batch.y is not None else len(batch.forecast_time[0])
        if horizon != self.residual_scale.shape[0]:
            raise ValueError("WindowBatch horizon does not match the fitted VAR scale")
        mean = self._recursive_mean(
            batch.x.to(torch.float64),
            self.coefficients,
            self.intercept,
            self.selected_lag,
            horizon,
        ).to(dtype=batch.x.dtype, device=batch.x.device)
        scale = self.residual_scale.to(dtype=batch.x.dtype, device=batch.x.device)
        scale = scale.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
        return validate_forecast_distribution(
            ForecastDistribution(
                mean=mean,
                scale=scale,
                quantiles=MappingProxyType({}),
                logits=None,
                samples=None,
                window_id=batch.window_id,
                target_names=self.target_names,
            )
        )

    def predict_tensors(self, histories: Tensor, horizon: int) -> ForecastDistribution:
        if histories.ndim != 3 or histories.shape[1] < self.selected_lag:
            raise ValueError("VAR tensor histories are incompatible with the selected lag")
        if histories.shape[2] != self.coefficients.shape[0]:
            raise ValueError("VAR tensor dimension does not match the fitted model")
        if horizon != self.residual_scale.shape[0]:
            raise ValueError("VAR tensor horizon does not match the fitted scale")
        mean = self._recursive_mean(
            histories.to(torch.float64),
            self.coefficients,
            self.intercept,
            self.selected_lag,
            horizon,
        ).to(dtype=histories.dtype, device=histories.device)
        scale = self.residual_scale.to(dtype=histories.dtype, device=histories.device)
        scale = scale.unsqueeze(0).expand(histories.shape[0], -1, -1)
        return validate_forecast_distribution(
            ForecastDistribution(
                mean=mean,
                scale=scale,
                quantiles=MappingProxyType({}),
                logits=None,
                samples=None,
                window_id=None,
                target_names=self.target_names,
            )
        )


def _validate_fit_tensors(
    train_x: Tensor,
    train_y: Tensor,
    tune_x: Tensor,
    tune_y: Tensor,
    target_names: tuple[str, ...],
) -> None:
    tensors = (train_x, train_y, tune_x, tune_y)
    if any(tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("VAR fit tensors must be finite rank-three tensors")
    if train_x.shape[0] != train_y.shape[0] or tune_x.shape[0] != tune_y.shape[0]:
        raise ValueError("VAR histories and targets must align by sample")
    dimension = train_x.shape[2]
    if any(tensor.shape[2] != dimension for tensor in tensors):
        raise ValueError("VAR fit tensors must share a feature dimension")
    if train_y.shape[1:] != tune_y.shape[1:]:
        raise ValueError("VAR train and tune horizons must match")
    if len(target_names) != dimension or len(set(target_names)) != dimension:
        raise ValueError("VAR target names must be unique and match the feature dimension")
