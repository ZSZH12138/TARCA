from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tarca.contracts import ForecastDistribution, WindowBatch, validate_window_batch
from tarca.stage1b.metrics import gaussian_crps
from tarca.stage2.distributions import gaussian_forecast, residual_scale, scale_ceiling


def _tensor_bytes(tensor: Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _model_hash(
    adapter: str,
    metadata: dict[str, object],
    tensors: tuple[Tensor, ...],
) -> str:
    payload = json.dumps(
        {"adapter": adapter, **metadata},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(payload)
    for tensor in tensors:
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _validate_fit_tensors(
    train_x: Tensor,
    train_y: Tensor,
    target_names: tuple[str, ...],
    validation_x: Tensor | None = None,
    validation_y: Tensor | None = None,
) -> None:
    tensors = tuple(
        tensor
        for tensor in (train_x, train_y, validation_x, validation_y)
        if tensor is not None
    )
    if any(
        tensor.ndim != 3
        or not tensor.is_floating_point()
        or not bool(torch.isfinite(tensor).all())
        for tensor in tensors
    ):
        raise ValueError("baseline fit tensors must be finite floating rank-three tensors")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("TRAIN histories and targets must align by sample")
    dimension = train_x.shape[2]
    if train_y.shape[2] != dimension:
        raise ValueError("TRAIN histories and targets must share a feature dimension")
    if len(target_names) != dimension or len(set(target_names)) != dimension:
        raise ValueError("target names must be unique and match the feature dimension")
    if (validation_x is None) != (validation_y is None):
        raise ValueError("validation histories and targets must be supplied together")
    if validation_x is not None and validation_y is not None:
        if validation_x.shape[0] != validation_y.shape[0]:
            raise ValueError("validation histories and targets must align by sample")
        if validation_x.shape[2] != dimension or validation_y.shape[2] != dimension:
            raise ValueError("validation tensors must match the TRAIN feature dimension")
        if validation_y.shape[1] != train_y.shape[1]:
            raise ValueError("TRAIN and validation horizons must match")


def _horizon(batch: WindowBatch) -> int:
    if batch.y is not None:
        return batch.y.shape[1]
    lengths = {len(times) for times in batch.forecast_time}
    if len(lengths) != 1:
        raise ValueError("forecast times must share a horizon")
    return next(iter(lengths))


def _expanded_scale(scale: Tensor, mean: Tensor) -> Tensor:
    return scale.to(dtype=mean.dtype, device=mean.device).unsqueeze(0).expand_as(mean)


def _check_prediction_batch(
    batch: WindowBatch,
    *,
    target_names: tuple[str, ...],
    fitted_horizon: int,
) -> int:
    validate_window_batch(batch)
    horizon = _horizon(batch)
    if horizon != fitted_horizon:
        raise ValueError("WindowBatch horizon does not match the fitted baseline")
    if batch.x.shape[2] != len(target_names):
        raise ValueError("WindowBatch dimension does not match the fitted baseline")
    if batch.target_names != target_names:
        raise ValueError("WindowBatch target names do not match the fitted baseline")
    return horizon


def _last_mean(histories: Tensor, horizon: int) -> Tensor:
    return histories[:, -1:, :].expand(-1, horizon, -1)


def _seasonal_mean(histories: Tensor, lag: int, horizon: int) -> Tensor:
    if lag <= 0 or lag > histories.shape[1]:
        raise ValueError("seasonal lag must fit within the available history")
    rolling = histories
    forecasts: list[Tensor] = []
    for _ in range(horizon):
        next_value = rolling[:, -lag, :]
        forecasts.append(next_value)
        rolling = torch.cat((rolling, next_value.unsqueeze(1)), dim=1)
    return torch.stack(forecasts, dim=1)


@dataclass(frozen=True, slots=True)
class LastValueGaussian:
    scale: Tensor
    target_names: tuple[str, ...]
    scale_source: str = "TRAIN_ONLY"

    @property
    def adapter_name(self) -> str:
        return "LAST_VALUE_GAUSSIAN"

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def model_hash(self) -> str:
        return _model_hash(
            self.adapter_name,
            {"target_names": self.target_names, "scale_source": self.scale_source},
            (self.scale,),
        )

    @classmethod
    def fit(
        cls,
        train_x: Tensor,
        train_y: Tensor,
        target_names: tuple[str, ...],
        *,
        floor: float = 1e-4,
        ceiling_multiplier: float = 10.0,
        absolute_ceiling: float = 10.0,
    ) -> LastValueGaussian:
        _validate_fit_tensors(train_x, train_y, target_names)
        train_mean = _last_mean(train_x.to(torch.float64), train_y.shape[1])
        scale = residual_scale(
            train_y.to(torch.float64) - train_mean,
            floor=floor,
            ceiling=scale_ceiling(
                train_y,
                multiplier=ceiling_multiplier,
                absolute_ceiling=absolute_ceiling,
            ),
        )
        return cls(scale=scale, target_names=target_names)

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        horizon = _check_prediction_batch(
            batch,
            target_names=self.target_names,
            fitted_horizon=self.scale.shape[0],
        )
        mean = _last_mean(batch.x, horizon)
        return gaussian_forecast(
            mean,
            _expanded_scale(self.scale, mean),
            window_id=batch.window_id,
            target_names=self.target_names,
        )


@dataclass(frozen=True, slots=True)
class SeasonalNaiveGaussian:
    selected_lag: int
    scale: Tensor
    target_names: tuple[str, ...]
    validation_crps: float
    scale_source: str = "TRAIN_ONLY"

    @property
    def adapter_name(self) -> str:
        return "SEASONAL_NAIVE_GAUSSIAN"

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def model_hash(self) -> str:
        return _model_hash(
            self.adapter_name,
            {
                "lag": self.selected_lag,
                "target_names": self.target_names,
                "scale_source": self.scale_source,
            },
            (self.scale,),
        )

    @classmethod
    def fit(
        cls,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        *,
        lags: tuple[int, ...],
        target_names: tuple[str, ...],
        floor: float = 1e-4,
        ceiling_multiplier: float = 10.0,
        absolute_ceiling: float = 10.0,
    ) -> SeasonalNaiveGaussian:
        _validate_fit_tensors(train_x, train_y, target_names, validation_x, validation_y)
        candidates = tuple(sorted(set(lags)))
        if not candidates or any(lag <= 0 or lag > train_x.shape[1] for lag in candidates):
            raise ValueError("seasonal lags must be unique positive values within the history")
        ceiling = scale_ceiling(
            train_y,
            multiplier=ceiling_multiplier,
            absolute_ceiling=absolute_ceiling,
        )
        scored: list[tuple[float, int, Tensor]] = []
        selection_horizon = min(6, train_y.shape[1])
        for lag in candidates:
            train_mean = _seasonal_mean(train_x.to(torch.float64), lag, train_y.shape[1])
            scale = residual_scale(
                train_y.to(torch.float64) - train_mean,
                floor=floor,
                ceiling=ceiling,
            )
            validation_mean = _seasonal_mean(
                validation_x.to(torch.float64), lag, validation_y.shape[1]
            )
            validation_scale = scale.unsqueeze(0).expand_as(validation_mean)
            score = float(
                gaussian_crps(
                    validation_mean[:, :selection_horizon],
                    validation_scale[:, :selection_horizon],
                    validation_y.to(torch.float64)[:, :selection_horizon],
                ).mean()
            )
            scored.append((score, lag, scale))
        score, selected_lag, scale = min(scored, key=lambda item: (item[0], item[1]))
        return cls(
            selected_lag=selected_lag,
            scale=scale,
            target_names=target_names,
            validation_crps=score,
        )

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        horizon = _check_prediction_batch(
            batch,
            target_names=self.target_names,
            fitted_horizon=self.scale.shape[0],
        )
        mean = _seasonal_mean(batch.x, self.selected_lag, horizon)
        return gaussian_forecast(
            mean,
            _expanded_scale(self.scale, mean),
            window_id=batch.window_id,
            target_names=self.target_names,
        )


@dataclass(frozen=True, slots=True)
class _VARCandidate:
    lag: int
    ridge: float
    coefficients: Tensor
    intercept: Tensor
    scale: Tensor
    validation_crps: float


def _fit_var_coefficients(
    train_x: Tensor,
    train_y: Tensor,
    lag: int,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    dimension = train_x.shape[2]
    design = train_x[:, -lag:, :].reshape(train_x.shape[0], lag * dimension).to(torch.float64)
    response = train_y[:, 0, :].to(torch.float64)
    ones = torch.ones((design.shape[0], 1), dtype=torch.float64, device=design.device)
    augmented = torch.cat((design, ones), dim=1)
    penalty = torch.eye(augmented.shape[1], dtype=torch.float64, device=design.device) * ridge
    penalty[-1, -1] = 0.0
    gram = augmented.transpose(0, 1) @ augmented + penalty
    beta = torch.linalg.solve(gram, augmented.transpose(0, 1) @ response)
    return beta[:-1].transpose(0, 1).contiguous(), beta[-1].contiguous()


def _var_mean(
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


@dataclass(frozen=True, slots=True)
class Stage2VARGaussian:
    coefficients: Tensor
    intercept: Tensor
    scale: Tensor
    selected_lag: int
    selected_ridge: float
    target_names: tuple[str, ...]
    validation_crps: float
    scale_source: str = "TRAIN_ONLY"

    @property
    def adapter_name(self) -> str:
        return "STAGE2_VAR_GAUSSIAN"

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def model_hash(self) -> str:
        return _model_hash(
            self.adapter_name,
            {
                "lag": self.selected_lag,
                "ridge": self.selected_ridge,
                "target_names": self.target_names,
                "scale_source": self.scale_source,
            },
            (self.coefficients, self.intercept, self.scale),
        )

    @classmethod
    def fit(
        cls,
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        *,
        lag_orders: tuple[int, ...],
        ridge_values: tuple[float, ...],
        target_names: tuple[str, ...],
        floor: float = 1e-4,
        ceiling_multiplier: float = 10.0,
        absolute_ceiling: float = 10.0,
    ) -> Stage2VARGaussian:
        _validate_fit_tensors(train_x, train_y, target_names, validation_x, validation_y)
        lags = tuple(sorted(set(lag_orders)))
        ridges = tuple(sorted(set(ridge_values)))
        if not lags or any(lag <= 0 or lag > train_x.shape[1] for lag in lags):
            raise ValueError("VAR lags must be unique positive values within the history")
        if not ridges or any(not math.isfinite(ridge) or ridge < 0 for ridge in ridges):
            raise ValueError("VAR ridge values must be unique finite nonnegative values")
        ceiling = scale_ceiling(
            train_y,
            multiplier=ceiling_multiplier,
            absolute_ceiling=absolute_ceiling,
        )
        selection_horizon = min(6, train_y.shape[1])
        candidates: list[_VARCandidate] = []
        for lag in lags:
            for ridge in ridges:
                coefficients, intercept = _fit_var_coefficients(train_x, train_y, lag, ridge)
                train_mean = _var_mean(
                    train_x.to(torch.float64), coefficients, intercept, lag, train_y.shape[1]
                )
                scale = residual_scale(
                    train_y.to(torch.float64) - train_mean,
                    floor=floor,
                    ceiling=ceiling,
                )
                validation_mean = _var_mean(
                    validation_x.to(torch.float64),
                    coefficients,
                    intercept,
                    lag,
                    validation_y.shape[1],
                )
                score = float(
                    gaussian_crps(
                        validation_mean[:, :selection_horizon],
                        scale.unsqueeze(0).expand_as(validation_mean)[:, :selection_horizon],
                        validation_y.to(torch.float64)[:, :selection_horizon],
                    ).mean()
                )
                candidates.append(
                    _VARCandidate(lag, ridge, coefficients, intercept, scale, score)
                )
        best = min(
            candidates,
            key=lambda item: (item.validation_crps, item.lag, item.ridge),
        )
        return cls(
            coefficients=best.coefficients,
            intercept=best.intercept,
            scale=best.scale,
            selected_lag=best.lag,
            selected_ridge=best.ridge,
            target_names=target_names,
            validation_crps=best.validation_crps,
        )

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        horizon = _check_prediction_batch(
            batch,
            target_names=self.target_names,
            fitted_horizon=self.scale.shape[0],
        )
        if batch.x.shape[1] < self.selected_lag:
            raise ValueError("WindowBatch history is shorter than the selected VAR lag")
        mean = _var_mean(
            batch.x.to(torch.float64),
            self.coefficients,
            self.intercept,
            self.selected_lag,
            horizon,
        ).to(dtype=batch.x.dtype, device=batch.x.device)
        return gaussian_forecast(
            mean,
            _expanded_scale(self.scale, mean),
            window_id=batch.window_id,
            target_names=self.target_names,
        )
