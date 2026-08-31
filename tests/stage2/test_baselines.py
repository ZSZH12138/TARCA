from __future__ import annotations

import torch

from tarca.contracts import ForecastPredictor, validate_forecast_distribution
from tarca.stage2.baselines import (
    LastValueGaussian,
    SeasonalNaiveGaussian,
    Stage2VARGaussian,
)
from tests.stage1b.model_helpers import window_batch

TARGET_NAMES = ("x0", "x1")


def _var1_windows() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(90210)
    coefficient = torch.tensor([[0.72, 0.08], [-0.04, 0.61]])
    histories: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for _ in range(18):
        values = [torch.randn(2, generator=generator) * 0.1]
        for _ in range(23):
            values.append(
                coefficient @ values[-1] + torch.randn(2, generator=generator) * 0.01
            )
        trajectory = torch.stack(values)
        for start in range(15):
            histories.append(trajectory[start : start + 6])
            targets.append(trajectory[start + 6 : start + 9])
    return torch.stack(histories), torch.stack(targets)


def _seasonal_windows() -> tuple[torch.Tensor, torch.Tensor]:
    histories: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for phase in range(12):
        base = torch.tensor(
            [[float((step + phase) % 2), float((step + phase + 1) % 2)] for step in range(9)]
        )
        histories.append(base[:6])
        targets.append(base[6:9])
    return torch.stack(histories), torch.stack(targets)


def test_last_value_scale_is_exactly_train_residual_rms() -> None:
    train_x, train_y = _var1_windows()

    model = LastValueGaussian.fit(train_x[:180], train_y[:180], TARGET_NAMES)

    train_mean = train_x[:180, -1:, :].expand_as(train_y[:180])
    expected = torch.sqrt(
        torch.mean((train_y[:180].double() - train_mean.double()).square(), dim=0)
    ).clamp_min(1e-4)
    assert torch.equal(model.scale, expected)
    assert model.scale_source == "TRAIN_ONLY"


def test_seasonal_lag_uses_validation_but_scale_uses_train() -> None:
    x, y = _seasonal_windows()

    model = SeasonalNaiveGaussian.fit(
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        lags=(1, 2),
        target_names=TARGET_NAMES,
    )

    assert model.selected_lag == 2
    assert model.scale_source == "TRAIN_ONLY"
    assert torch.equal(model.scale, torch.full((3, 2), 1e-4, dtype=torch.float64))


def test_stage2_var_validation_targets_cannot_change_train_residual_scale() -> None:
    x, y = _var1_windows()
    train_x, validation_x = x[:180], x[180:]
    train_y, validation_y = y[:180], y[180:]

    normal = Stage2VARGaussian.fit(
        train_x,
        train_y,
        validation_x,
        validation_y,
        lag_orders=(1,),
        ridge_values=(1e-6,),
        target_names=TARGET_NAMES,
    )
    changed = Stage2VARGaussian.fit(
        train_x,
        train_y,
        validation_x,
        validation_y + 1_000_000.0,
        lag_orders=(1,),
        ridge_values=(1e-6,),
        target_names=TARGET_NAMES,
    )

    assert torch.equal(normal.scale, changed.scale)
    assert normal.scale_source == "TRAIN_ONLY"


def test_all_baselines_emit_valid_gaussian_forecast_contracts() -> None:
    x, y = _var1_windows()
    train_x, validation_x = x[:180], x[180:]
    train_y, validation_y = y[:180], y[180:]
    predictors: tuple[ForecastPredictor, ...] = (
        LastValueGaussian.fit(train_x, train_y, TARGET_NAMES),
        SeasonalNaiveGaussian.fit(
            train_x,
            train_y,
            validation_x,
            validation_y,
            lags=(1, 2, 4),
            target_names=TARGET_NAMES,
        ),
        Stage2VARGaussian.fit(
            train_x,
            train_y,
            validation_x,
            validation_y,
            lag_orders=(1, 2),
            ridge_values=(1e-6,),
            target_names=TARGET_NAMES,
        ),
    )
    batch = window_batch(validation_x[:4], validation_y[:4])

    for predictor in predictors:
        forecast = validate_forecast_distribution(predictor.predict_distribution(batch))
        assert forecast.mean.shape == forecast.scale.shape == (4, 3, 2)
        assert tuple(forecast.quantiles) == (0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975)
        assert predictor.is_frozen
