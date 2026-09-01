from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from tarca.contracts import ForecastDistribution
from tarca.execution.errors import DeviceContractError
from tarca.stage1b.modeling import OfficialOperablePredictor
from tarca.stage1b.neural import OperableNeuralPredictor
from tarca.stage1b.training import (
    Precision,
    ProgressSink,
    TrainingPolicy,
    train_candidate,
)

Stage2NeuralModel = OperableNeuralPredictor | OfficialOperablePredictor
Stage2NeuralModelId = Literal["PATCHTST", "ITRANSFORMER"]


@dataclass(frozen=True, slots=True)
class Stage2TrainingPolicy:
    model_id: Stage2NeuralModelId
    device: str
    precision: Precision
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    dataloader_workers: int
    checkpoint_root: Path
    checkpoint_every_epochs: int = 1
    optimizer: Literal["ADAMW"] = "ADAMW"
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    scheduler: Literal["NONE"] = "NONE"
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    def __post_init__(self) -> None:
        exact = (
            self.optimizer,
            self.betas,
            self.epsilon,
            self.weight_decay,
            self.gradient_clip_norm,
            self.scheduler,
            self.deterministic_algorithms,
            self.cudnn_deterministic,
            self.cudnn_benchmark,
        )
        expected = (
            "ADAMW",
            (0.9, 0.999),
            1e-8,
            0.01,
            1.0,
            "NONE",
            True,
            True,
            False,
        )
        if exact != expected:
            raise ValueError(
                "Stage 2 training must use the frozen optimizer and determinism policy"
            )
        object.__setattr__(self, "checkpoint_root", self.checkpoint_root.resolve())
        self.as_stage1b_policy()

    def as_stage1b_policy(self) -> TrainingPolicy:
        return TrainingPolicy(
            device=self.device,
            precision=self.precision,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            learning_rate=self.learning_rate,
            dataloader_workers=self.dataloader_workers,
            checkpoint_root=self.checkpoint_root,
            checkpoint_every_epochs=self.checkpoint_every_epochs,
            optimizer=self.optimizer,
            betas=self.betas,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            scheduler=self.scheduler,
            deterministic_algorithms=self.deterministic_algorithms,
            cudnn_deterministic=self.cudnn_deterministic,
            cudnn_benchmark=self.cudnn_benchmark,
        )


@dataclass(frozen=True, slots=True)
class Stage2TrainingResult:
    model: Stage2NeuralModel
    model_sha256: str
    checkpoint: Path | None
    checkpoint_sha256: str | None
    fixed_batch_forecast_sha256: str
    best_epoch: int
    best_validation_nll: float
    epochs_completed: int
    completed: bool
    device: str
    precision: Precision


def _forecast_sha256(mean: Tensor, scale: Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("mean", mean), ("scale", scale)):
        resolved = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(resolved.dtype).encode())
        digest.update(str(tuple(resolved.shape)).encode())
        digest.update(resolved.numpy().tobytes())
    return digest.hexdigest()


def _validate_model_identity(model: Stage2NeuralModel, model_id: Stage2NeuralModelId) -> None:
    expected = {
        "PATCHTST": "PatchTSTReference",
        "ITRANSFORMER": "ITransformerReference",
    }[model_id]
    if model.adapter_name != expected:
        raise ValueError(f"Stage 2 {model_id} policy does not match {model.adapter_name}")


def _single_model_device(model: Stage2NeuralModel) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise DeviceContractError("Stage 2 neural model must occupy exactly one device")
    return next(iter(devices))


def forecast_fixed_batch_on_model_device(
    model: Stage2NeuralModel,
    validation_x: Tensor,
) -> ForecastDistribution:
    """Run the fixed validation forecast on the model's actual device."""
    model_device = _single_model_device(model)
    fixed_batch = validation_x.to(device=model_device)
    if fixed_batch.device != model_device:
        raise DeviceContractError("fixed validation batch did not reach the model device")
    model.eval()
    with torch.inference_mode():
        return model.forward_distribution(fixed_batch)


def train_stage2_neural(
    model: Stage2NeuralModel,
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    *,
    policy: Stage2TrainingPolicy,
    seed: int,
    progress: ProgressSink | None = None,
    resume_from: Path | None = None,
    resume_if_available: bool = False,
    require_complete_resume: bool = False,
    stop_after_epoch: int | None = None,
) -> Stage2TrainingResult:
    _validate_model_identity(model, policy.model_id)
    trained = train_candidate(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=seed,
        policy=policy.as_stage1b_policy(),
        progress=progress,
        resume_from=resume_from,
        resume_if_available=resume_if_available,
        require_complete_resume=require_complete_resume,
        stop_after_epoch=stop_after_epoch,
    )
    fixed_count = min(2, validation_x.shape[0])
    forecast = forecast_fixed_batch_on_model_device(
        trained.model, validation_x[:fixed_count]
    )
    if forecast.scale is None or forecast.mean.shape != validation_y[:fixed_count].shape:
        raise RuntimeError("Stage 2 neural checkpoint has an invalid probability shape")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (forecast.mean, forecast.scale)):
        raise RuntimeError("Stage 2 neural checkpoint produced non-finite probabilities")
    if bool((forecast.scale <= 0).any()):
        raise RuntimeError("Stage 2 neural checkpoint produced a non-positive scale")
    receipt = trained.receipt
    if not math.isfinite(receipt.best_tune_nll):
        raise RuntimeError("Stage 2 neural checkpoint has a non-finite validation NLL")
    return Stage2TrainingResult(
        model=trained.model,
        model_sha256=receipt.model_sha256,
        checkpoint=trained.checkpoint,
        checkpoint_sha256=receipt.checkpoint_sha256,
        fixed_batch_forecast_sha256=_forecast_sha256(forecast.mean, forecast.scale),
        best_epoch=receipt.best_epoch,
        best_validation_nll=receipt.best_tune_nll,
        epochs_completed=receipt.epochs_completed,
        completed=receipt.completed,
        device=receipt.device,
        precision=receipt.precision,
    )
