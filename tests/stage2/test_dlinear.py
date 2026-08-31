from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from tarca.contracts import validate_forecast_distribution
from tarca.stage2.dlinear import (
    DLinearGaussian,
    DLinearModelConfig,
    dlinear_fold_index,
    dlinear_state_sha256,
    fit_dlinear_cross_fitted,
    load_dlinear_checkpoint,
    load_official_dlinear,
    save_dlinear_checkpoint,
)
from tests.stage1b.model_helpers import window_batch


class _RepeatLast(nn.Module):
    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, -1:, :].expand(-1, self.horizon, -1)


class _TinyMean(nn.Module):
    def __init__(self, history: int, horizon: int, dimension: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.dimension = dimension
        self.linear = nn.Linear(history * dimension, horizon * dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flattened = values.reshape(values.shape[0], -1)
        return self.linear(flattened).reshape(-1, self.horizon, self.dimension)


def _synthetic_source(path: Path) -> DLinearModelConfig:
    source = path / "models" / "DLinear.py"
    source.parent.mkdir(parents=True)
    content = b"""\
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        return x[:, -1:, :].expand(-1, self.pred_len, -1)
"""
    source.write_bytes(content)
    return DLinearModelConfig(
        sequence_length=4,
        prediction_length=3,
        dimension=2,
        individual=False,
        moving_average_kernel=3,
        asset_relative_path="models/DLinear.py",
        asset_sha256=hashlib.sha256(content).hexdigest(),
    )


def test_official_dlinear_asset_hash_is_required(tmp_path: Path) -> None:
    model_config = _synthetic_source(tmp_path)
    (tmp_path / "models" / "DLinear.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="DLinear source hash"):
        load_official_dlinear(tmp_path, model_config)


def test_verified_official_dlinear_loads_expected_shape(tmp_path: Path) -> None:
    model = load_official_dlinear(tmp_path, _synthetic_source(tmp_path))

    output = model(torch.arange(16, dtype=torch.float32).reshape(2, 4, 2))

    assert output.shape == (2, 3, 2)
    assert not (tmp_path / "models" / "__pycache__").exists()


def test_dlinear_distribution_matches_contract() -> None:
    predictor = DLinearGaussian(
        mean_model=_RepeatLast(horizon=3),
        scale=torch.full((3, 2), 0.25, dtype=torch.float64),
        target_names=("x0", "x1"),
        checkpoint_sha256="a" * 64,
    )
    x = torch.randn((2, 4, 2))
    y = torch.randn((2, 3, 2))

    forecast = validate_forecast_distribution(
        predictor.predict_distribution(window_batch(x, y))
    )

    assert forecast.mean.shape == forecast.scale.shape == (2, 3, 2)
    assert bool((forecast.scale > 0).all())
    assert len(forecast.quantiles) == 8


def test_dlinear_checkpoint_round_trip_reproduces_forecast(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = _TinyMean(4, 3, 2)
    predictor = DLinearGaussian(
        mean_model=model,
        scale=torch.full((3, 2), 0.25, dtype=torch.float64),
        target_names=("x0", "x1"),
        checkpoint_sha256=dlinear_state_sha256(model),
    )
    checkpoint = tmp_path / "dlinear.pt"
    file_sha256 = save_dlinear_checkpoint(predictor, checkpoint)

    reloaded = load_dlinear_checkpoint(
        checkpoint,
        lambda: _TinyMean(4, 3, 2),
        expected_file_sha256=file_sha256,
    )
    x = torch.randn((2, 4, 2))
    y = torch.randn((2, 3, 2))
    batch = window_batch(x, y)

    expected = predictor.predict_distribution(batch)
    observed = reloaded.predict_distribution(batch)
    assert torch.equal(expected.mean, observed.mean)
    assert torch.equal(expected.scale, observed.scale)
    assert reloaded.checkpoint_sha256 == predictor.checkpoint_sha256


def _fold_balanced_ids(per_fold: int) -> tuple[str, ...]:
    selected: list[str] = []
    counts = [0] * 5
    candidate = 0
    while min(counts) < per_fold:
        identifier = f"trajectory-{candidate}"
        fold = dlinear_fold_index(identifier, fold_count=5)
        if counts[fold] < per_fold:
            selected.append(identifier)
            counts[fold] += 1
        candidate += 1
    return tuple(selected)


def test_cross_fitted_scale_is_deterministic_and_validation_isolated() -> None:
    generator = torch.Generator().manual_seed(4455)
    train_x = torch.randn((10, 4, 2), generator=generator)
    train_y = torch.randn((10, 3, 2), generator=generator)
    validation_x = torch.randn((4, 4, 2), generator=generator)
    validation_y = torch.randn((4, 3, 2), generator=generator)
    identifiers = _fold_balanced_ids(per_fold=2)

    def fit(targets: torch.Tensor):
        return fit_dlinear_cross_fitted(
            lambda: _TinyMean(4, 3, 2),
            train_x,
            train_y,
            identifiers,
            validation_x,
            targets,
            target_names=("x0", "x1"),
            fold_seeds=(11, 22, 33, 44, 55),
            final_seed=66,
            batch_size=5,
            max_epochs=1,
            patience=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        )

    first = fit(validation_y)
    replay = fit(validation_y)
    changed_validation = fit(validation_y + 1_000_000.0)

    assert first.checkpoint_sha256 == replay.checkpoint_sha256
    assert first.cross_fit_scale_sha256 == replay.cross_fit_scale_sha256
    assert first.cross_fit_scale_sha256 == changed_validation.cross_fit_scale_sha256
    assert first.predictor.scale_source == "CROSS_FITTED_TRAIN_ONLY"
