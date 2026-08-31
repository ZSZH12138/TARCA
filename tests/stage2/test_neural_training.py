from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarca.stage1b.neural import ITransformerReference
from tarca.stage2.training import (
    Stage2TrainingPolicy,
    train_stage2_neural,
)


def _model(*, dropout: float = 0.0) -> ITransformerReference:
    return ITransformerReference(
        history_length=8,
        horizon=2,
        input_dimension=3,
        d_model=12,
        n_layers=1,
        n_heads=3,
        d_ff=24,
        dropout=dropout,
    )


def _data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31415)
    train_x = torch.randn((24, 8, 3), generator=generator)
    train_y = train_x[:, -1:, :].repeat(1, 2, 1).square()
    validation_x = torch.randn((8, 8, 3), generator=generator)
    validation_y = validation_x[:, -1:, :].repeat(1, 2, 1).square()
    return train_x, train_y, validation_x, validation_y


def _policy(root: Path, *, max_epochs: int = 3) -> Stage2TrainingPolicy:
    return Stage2TrainingPolicy(
        model_id="ITRANSFORMER",
        device="cpu",
        precision="FP32",
        batch_size=8,
        max_epochs=max_epochs,
        patience=max_epochs - 1,
        learning_rate=1e-3,
        dataloader_workers=0,
        checkpoint_root=root,
    )


def test_stage2_training_policy_is_exact(tmp_path: Path) -> None:
    policy = _policy(tmp_path)

    assert policy.optimizer == "ADAMW"
    assert policy.betas == (0.9, 0.999)
    assert policy.epsilon == 1e-8
    assert policy.weight_decay == 0.01
    assert policy.gradient_clip_norm == 1.0
    assert policy.scheduler == "NONE"
    assert policy.deterministic_algorithms is True
    assert policy.cudnn_deterministic is True
    assert policy.cudnn_benchmark is False


def test_stage2_policy_rejects_optimizer_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen optimizer"):
        Stage2TrainingPolicy(
            model_id="ITRANSFORMER",
            device="cpu",
            precision="FP32",
            batch_size=8,
            max_epochs=3,
            patience=2,
            learning_rate=1e-3,
            dataloader_workers=0,
            checkpoint_root=tmp_path,
            weight_decay=0.0,
        )


def test_resume_preserves_checkpoint_and_fixed_batch_forecast(tmp_path: Path) -> None:
    train_x, train_y, validation_x, validation_y = _data()
    uninterrupted = train_stage2_neural(
        _model(dropout=0.1),
        train_x,
        train_y,
        validation_x,
        validation_y,
        policy=_policy(tmp_path / "full"),
        seed=1797287582,
    )
    interrupted = train_stage2_neural(
        _model(dropout=0.1),
        train_x,
        train_y,
        validation_x,
        validation_y,
        policy=_policy(tmp_path / "resume"),
        seed=1797287582,
        stop_after_epoch=1,
    )
    assert interrupted.completed is False
    assert interrupted.checkpoint is not None
    resumed = train_stage2_neural(
        _model(dropout=0.1),
        train_x,
        train_y,
        validation_x,
        validation_y,
        policy=_policy(tmp_path / "resume"),
        seed=1797287582,
        resume_from=interrupted.checkpoint,
    )

    assert resumed.completed is True
    assert resumed.model_sha256 == uninterrupted.model_sha256
    assert resumed.fixed_batch_forecast_sha256 == (
        uninterrupted.fixed_batch_forecast_sha256
    )
    assert resumed.best_validation_nll == uninterrupted.best_validation_nll


def test_stage2_training_emits_positive_finite_probability_and_checkpoint(
    tmp_path: Path,
) -> None:
    train_x, train_y, validation_x, validation_y = _data()

    result = train_stage2_neural(
        _model(),
        train_x,
        train_y,
        validation_x,
        validation_y,
        policy=_policy(tmp_path, max_epochs=2),
        seed=1797287582,
    )
    forecast = result.model.forward_distribution(validation_x[:2])

    assert result.completed is True
    assert result.checkpoint is not None and result.checkpoint.is_file()
    assert result.checkpoint_sha256 is not None
    assert result.best_epoch >= 0
    assert torch.isfinite(forecast.mean).all()
    assert forecast.scale is not None
    assert torch.isfinite(forecast.scale).all()
    assert bool((forecast.scale > 0).all())
    assert forecast.mean.shape == forecast.scale.shape == (2, 2, 3)

