from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from tarca.stage1b.neural import ITransformerReference
from tarca.stage1b.training import (
    TrainingPolicy,
    TrainingProgress,
    train_candidate,
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
    generator = torch.Generator().manual_seed(902)
    train_x = torch.randn(24, 8, 3, generator=generator)
    train_y = train_x[:, -1:, :].repeat(1, 2, 1) ** 2
    tune_x = torch.randn(8, 8, 3, generator=generator)
    tune_y = tune_x[:, -1:, :].repeat(1, 2, 1) ** 2
    return train_x, train_y, tune_x, tune_y


@dataclass(slots=True)
class _ProgressRecorder:
    values: list[TrainingProgress] = field(default_factory=list)

    def report(self, progress: TrainingProgress) -> None:
        self.values.append(progress)


def test_same_seed_produces_identical_receipt_and_predictions() -> None:
    train_x, train_y, tune_x, tune_y = _data()

    first = train_candidate(
        _model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
    )
    second = train_candidate(
        _model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
    )
    assert first.receipt == second.receipt
    torch.testing.assert_close(
        first.model.forward_distribution(tune_x).mean,
        second.model.forward_distribution(tune_x).mean,
        rtol=0.0,
        atol=0.0,
    )


def test_training_receipt_binds_device_progress_and_atomic_checkpoint(tmp_path: Path) -> None:
    train_x, train_y, tune_x, tune_y = _data()
    progress = _ProgressRecorder()
    policy = TrainingPolicy(
        device="cpu",
        precision="FP32",
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        dataloader_workers=0,
        checkpoint_root=tmp_path,
    )
    result = train_candidate(
        _model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        policy=policy,
        progress=progress,
    )
    assert result.receipt.device == "cpu"
    assert result.receipt.precision == "FP32"
    assert result.checkpoint is not None and result.checkpoint.is_file()
    assert result.receipt.checkpoint_sha256 == hashlib.sha256(
        result.checkpoint.read_bytes()
    ).hexdigest()
    assert len(result.receipt.checkpoint_sha256) == 64
    assert not tuple(tmp_path.glob("*.tmp-*"))
    assert progress.values
    assert progress.values[-1].completed_steps <= progress.values[-1].total_steps


def test_resumed_training_matches_uninterrupted(tmp_path: Path) -> None:
    train_x, train_y, tune_x, tune_y = _data()

    def policy(root: Path) -> TrainingPolicy:
        return TrainingPolicy(
            device="cpu",
            precision="FP32",
            batch_size=8,
            max_epochs=3,
            patience=2,
            learning_rate=1e-3,
            dataloader_workers=0,
            checkpoint_root=root,
        )

    uninterrupted = train_candidate(
        _model(dropout=0.1),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        policy=policy(tmp_path / "full"),
    )
    interrupted = train_candidate(
        _model(dropout=0.1),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        policy=policy(tmp_path / "resume"),
        stop_after_epoch=1,
    )
    assert interrupted.checkpoint is not None
    assert interrupted.receipt.completed is False
    resumed = train_candidate(
        _model(dropout=0.1),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        policy=policy(tmp_path / "resume"),
        resume_from=interrupted.checkpoint,
    )
    assert resumed.receipt.completed is True
    assert resumed.receipt.model_sha256 == uninterrupted.receipt.model_sha256
    torch.testing.assert_close(
        resumed.model.forward_distribution(tune_x).mean,
        uninterrupted.model.forward_distribution(tune_x).mean,
        atol=0.0,
        rtol=0.0,
    )


def test_resume_rejects_changed_training_data(tmp_path: Path) -> None:
    train_x, train_y, tune_x, tune_y = _data()
    policy = TrainingPolicy(
        device="cpu",
        precision="FP32",
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        dataloader_workers=0,
        checkpoint_root=tmp_path,
    )
    interrupted = train_candidate(
        _model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        policy=policy,
        stop_after_epoch=1,
    )
    assert interrupted.checkpoint is not None
    with pytest.raises(ValueError, match="training data"):
        train_candidate(
            _model(),
            train_x + 1.0,
            train_y,
            tune_x,
            tune_y,
            seed=903,
            policy=policy,
            resume_from=interrupted.checkpoint,
        )


def test_amp_is_rejected_without_cuda(tmp_path: Path) -> None:
    train_x, train_y, tune_x, tune_y = _data()
    policy = TrainingPolicy(
        device="cpu",
        precision="AMP_FP16",
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        dataloader_workers=0,
        checkpoint_root=tmp_path,
    )
    with pytest.raises(ValueError, match="AMP_FP16 requires CUDA"):
        train_candidate(
            _model(),
            train_x,
            train_y,
            tune_x,
            tune_y,
            seed=903,
            policy=policy,
        )
