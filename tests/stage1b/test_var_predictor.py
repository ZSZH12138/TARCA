from __future__ import annotations

import torch

from tarca.contracts import validate_forecast_distribution
from tarca.stage1b.predictors import TunedVAR

from .model_helpers import window_batch


def _var1_windows() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(801)
    coefficient = torch.tensor([[0.75, 0.10], [-0.05, 0.65]])
    trajectories: list[torch.Tensor] = []
    for _ in range(16):
        values = [torch.randn(2, generator=generator) * 0.1]
        for _ in range(39):
            values.append(coefficient @ values[-1] + torch.randn(2, generator=generator) * 0.01)
        trajectories.append(torch.stack(values))
    histories: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for trajectory in trajectories:
        for start in range(0, 32):
            histories.append(trajectory[start : start + 6])
            targets.append(trajectory[start + 6 : start + 8])
    return torch.stack(histories), torch.stack(targets)


def test_var_recovers_var1_and_emits_positive_probabilistic_scale() -> None:
    histories, targets = _var1_windows()
    train_x, tune_x = histories[:400], histories[400:]
    train_y, tune_y = targets[:400], targets[400:]

    predictor = TunedVAR.fit(
        train_x=train_x,
        train_y=train_y,
        tune_x=tune_x,
        tune_y=tune_y,
        lag_orders=(1, 2, 4),
        ridge_values=(1e-6, 1e-3),
        target_names=("x0", "x1"),
    )
    batch = window_batch(tune_x[:4], tune_y[:4])
    forecast = validate_forecast_distribution(predictor.predict_distribution(batch))

    assert forecast.mean.shape == (4, 2, 2)
    assert forecast.scale is not None
    assert bool((forecast.scale > 0).all())
    assert predictor.selected_lag == 1
    assert predictor.is_frozen

