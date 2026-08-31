from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tarca.contracts import canonical_json_hash
from tarca.stage1b.modeling import OfficialOperablePredictor
from tarca.stage1b.neural import OperableNeuralPredictor
from tarca.stage1b.training_checkpoints import (
    atomic_torch_save as _atomic_torch_save,
)
from tarca.stage1b.training_checkpoints import (
    checkpoint_path as _checkpoint_path,
)
from tarca.stage1b.training_checkpoints import (
    checkpoint_payload as _checkpoint_payload,
)
from tarca.stage1b.training_checkpoints import (
    load_checkpoint as _load_checkpoint,
)
from tarca.stage1b.training_checkpoints import (
    policy_hash as _policy_hash,
)
from tarca.stage1b.training_checkpoints import (
    resolve_resume_path as _resolve_resume_path,
)
from tarca.stage1b.training_checkpoints import (
    restore_checkpoint as _restore_checkpoint,
)
from tarca.stage1b.training_checkpoints import (
    training_data_hash as _training_data_hash,
)

Stage1BNeuralPredictor = OperableNeuralPredictor | OfficialOperablePredictor
Precision = Literal["FP32", "AMP_FP16"]


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
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
        if re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", self.device) is None:
            raise ValueError("training device must be cpu, cuda, or cuda:<index>")
        if self.precision not in ("FP32", "AMP_FP16"):
            raise ValueError("unsupported training precision")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("training batch size and epochs must be positive")
        if not 0 <= self.patience < self.max_epochs:
            raise ValueError("training patience must be non-negative and below max_epochs")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("training learning rate must be finite and positive")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers must be non-negative")
        if self.checkpoint_every_epochs <= 0:
            raise ValueError("checkpoint interval must be positive")
        if self.optimizer != "ADAMW" or self.scheduler != "NONE":
            raise ValueError("training supports only AdamW without a scheduler")
        if (
            len(self.betas) != 2
            or not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in self.betas)
            or self.betas[0] >= self.betas[1]
        ):
            raise ValueError("training AdamW betas are invalid")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("training AdamW epsilon must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("training weight decay must be finite and nonnegative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("training gradient clip norm must be finite and positive")
        if any(
            type(value) is not bool
            for value in (
                self.deterministic_algorithms,
                self.cudnn_deterministic,
                self.cudnn_benchmark,
            )
        ):
            raise ValueError("training deterministic flags must be boolean")
        object.__setattr__(self, "checkpoint_root", self.checkpoint_root.resolve())


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    epoch: int
    batch: int
    completed_steps: int
    total_steps: int
    samples_per_second: float


class ProgressSink(Protocol):
    def report(self, progress: TrainingProgress) -> None:
        raise NotImplementedError


class _NullProgressSink:
    def report(self, progress: TrainingProgress) -> None:
        del progress


@dataclass(frozen=True, slots=True)
class TrainingReceipt:
    adapter_name: str
    seed: int
    epochs_completed: int
    best_epoch: int
    best_tune_nll: float
    train_sample_count: int
    tune_sample_count: int
    model_hash: str
    device: str
    precision: Precision
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    completed: bool

    @property
    def model_sha256(self) -> str:
        return self.model_hash


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: Stage1BNeuralPredictor
    receipt: TrainingReceipt
    checkpoint: Path | None = None


def _gaussian_nll(mean: Tensor, scale: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.log(scale) + 0.5 * ((target - mean) / scale) ** 2)


def _legacy_policy(
    *,
    batch_size: int | None,
    max_epochs: int | None,
    patience: int | None,
    learning_rate: float | None,
) -> TrainingPolicy:
    if None in (batch_size, max_epochs, patience, learning_rate):
        raise ValueError("legacy training requires the complete scalar budget")
    return TrainingPolicy(
        device="cpu",
        precision="FP32",
        batch_size=cast(int, batch_size),
        max_epochs=cast(int, max_epochs),
        patience=cast(int, patience),
        learning_rate=cast(float, learning_rate),
        dataloader_workers=0,
        checkpoint_root=Path.cwd() / ".tarca-disabled-checkpoints",
    )


def _validate_training_tensors(
    train_x: Tensor,
    train_y: Tensor,
    tune_x: Tensor,
    tune_y: Tensor,
) -> None:
    if any(tensor.ndim != 3 for tensor in (train_x, train_y, tune_x, tune_y)):
        raise ValueError("neural training tensors must have rank three")
    if train_x.shape[0] != train_y.shape[0] or tune_x.shape[0] != tune_y.shape[0]:
        raise ValueError("neural histories and targets must align by sample")
    if train_x.shape[0] == 0 or tune_x.shape[0] == 0:
        raise ValueError("neural training partitions must not be empty")
    if any(not tensor.is_floating_point() for tensor in (train_x, train_y, tune_x, tune_y)):
        raise ValueError("neural training tensors must use floating-point values")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in (train_x, train_y, tune_x, tune_y)):
        raise ValueError("neural training tensors must contain only finite values")


def _evaluate_tune_nll(
    model: Stage1BNeuralPredictor,
    loader: DataLoader[Any],
    device: torch.device,
    non_blocking: bool,
) -> float:
    weighted_loss = 0.0
    element_count = 0
    model.eval()
    with torch.no_grad():
        for tune_batch_x, tune_batch_y in loader:
            batch_x = tune_batch_x.to(device=device, non_blocking=non_blocking)
            batch_y = tune_batch_y.to(device=device, non_blocking=non_blocking)
            distribution = model.forward_distribution(batch_x)
            if distribution.scale is None:
                raise RuntimeError("neural candidate did not emit probabilistic scale")
            batch_loss = _gaussian_nll(distribution.mean, distribution.scale, batch_y)
            elements = batch_y.numel()
            weighted_loss += float(batch_loss) * elements
            element_count += elements
    if element_count == 0 or not math.isfinite(weighted_loss):
        raise RuntimeError("neural tuning produced no finite loss")
    return weighted_loss / element_count


def train_candidate(
    model: Stage1BNeuralPredictor,
    train_x: Tensor,
    train_y: Tensor,
    tune_x: Tensor,
    tune_y: Tensor,
    *,
    seed: int,
    policy: TrainingPolicy | None = None,
    progress: ProgressSink | None = None,
    resume_from: Path | None = None,
    resume_if_available: bool = False,
    stop_after_epoch: int | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    learning_rate: float | None = None,
) -> TrainingResult:
    _validate_training_tensors(train_x, train_y, tune_x, tune_y)
    explicit_policy = policy is not None
    if explicit_policy and any(
        value is not None for value in (batch_size, max_epochs, patience, learning_rate)
    ):
        raise ValueError("training policy cannot be combined with legacy scalar budgets")
    resolved_policy = policy or _legacy_policy(
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    if type(resume_if_available) is not bool:
        raise ValueError("resume_if_available must be boolean")
    if resume_from is not None and resume_if_available:
        raise ValueError("explicit and automatic checkpoint resume are mutually exclusive")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= resolved_policy.max_epochs:
        raise ValueError("stop_after_epoch must be within the training budget")
    device = torch.device(resolved_policy.device)
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but CUDA is unavailable")
    if resolved_policy.precision == "AMP_FP16" and not use_cuda:
        raise ValueError("AMP_FP16 requires CUDA")
    if not explicit_policy and (
        resume_from is not None or resume_if_available or stop_after_epoch is not None
    ):
        raise ValueError("checkpoint and interruption controls require an explicit policy")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(resolved_policy.deterministic_algorithms)
    torch.backends.cudnn.benchmark = resolved_policy.cudnn_benchmark
    torch.backends.cudnn.deterministic = resolved_policy.cudnn_deterministic
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed_all(seed)
        model.reset_for_training(seed)
        initial_model_hash = model.model_hash
        data_sha256 = _training_data_hash(
            {"train_x": train_x, "train_y": train_y, "tune_x": tune_x, "tune_y": tune_y}
        )
        config_sha256 = _policy_hash(resolved_policy)
        precision_sha256 = canonical_json_hash(
            {"device": resolved_policy.device, "precision": resolved_policy.precision}
        )
        task_sha256 = canonical_json_hash(
            {
                "adapter_name": model.adapter_name,
                "initial_model_sha256": initial_model_hash,
                "seed": seed,
                "data_sha256": data_sha256,
                "config_sha256": config_sha256,
                "precision_sha256": precision_sha256,
            }
        )
        checkpoint = _checkpoint_path(resolved_policy, task_sha256) if explicit_policy else None
        resolved_resume = resume_from
        if resume_if_available and checkpoint is not None and checkpoint.is_file():
            resolved_resume = checkpoint
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved_policy.learning_rate,
            betas=resolved_policy.betas,
            eps=resolved_policy.epsilon,
            weight_decay=resolved_policy.weight_decay,
        )
        amp_enabled = resolved_policy.precision == "AMP_FP16"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        shuffle_generator = torch.Generator().manual_seed(seed)
        train_dataset = TensorDataset(train_x.detach().cpu(), train_y.detach().cpu())
        tune_dataset = TensorDataset(tune_x.detach().cpu(), tune_y.detach().cpu())
        train_loader = DataLoader(
            train_dataset,
            batch_size=resolved_policy.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=resolved_policy.dataloader_workers,
            pin_memory=use_cuda,
            persistent_workers=resolved_policy.dataloader_workers > 0,
        )
        tune_loader = DataLoader(
            tune_dataset,
            batch_size=resolved_policy.batch_size,
            shuffle=False,
            num_workers=resolved_policy.dataloader_workers,
            pin_memory=use_cuda,
            persistent_workers=resolved_policy.dataloader_workers > 0,
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, Tensor] = {}
        stale_epochs = 0
        epochs_completed = 0
        start_epoch = 0
        if resolved_resume is not None:
            resume_path = _resolve_resume_path(resolved_resume, resolved_policy.checkpoint_root)
            payload = _load_checkpoint(resume_path)
            start_epoch, best_loss, best_epoch, best_state, stale_epochs = _restore_checkpoint(
                payload,
                expected_task_sha256=task_sha256,
                expected_data_sha256=data_sha256,
                expected_config_sha256=config_sha256,
                expected_precision_sha256=precision_sha256,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                shuffle_generator=shuffle_generator,
                use_cuda=use_cuda,
            )
            epochs_completed = start_epoch

        total_steps = len(train_loader) * resolved_policy.max_epochs
        completed_steps = len(train_loader) * start_epoch
        processed_samples = 0
        progress_sink = progress or _NullProgressSink()
        started = time.perf_counter()
        interrupted = False
        checkpoint_sha256: str | None = None
        for epoch in range(start_epoch, resolved_policy.max_epochs):
            model.train()
            for batch_index, (train_batch_x, train_batch_y) in enumerate(train_loader, start=1):
                batch_x = train_batch_x.to(device=device, non_blocking=use_cuda)
                batch_y = train_batch_y.to(device=device, non_blocking=use_cuda)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    distribution = model.forward_distribution(batch_x)
                    if distribution.scale is None:
                        raise RuntimeError("neural candidate did not emit probabilistic scale")
                    loss = _gaussian_nll(distribution.mean, distribution.scale, batch_y)
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=resolved_policy.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                completed_steps += 1
                processed_samples += batch_x.shape[0]
                elapsed = max(time.perf_counter() - started, 1e-9)
                progress_sink.report(
                    TrainingProgress(
                        epoch=epoch + 1,
                        batch=batch_index,
                        completed_steps=completed_steps,
                        total_steps=total_steps,
                        samples_per_second=processed_samples / elapsed,
                    )
                )
            tune_loss = _evaluate_tune_nll(model, tune_loader, device, use_cuda)
            epochs_completed = epoch + 1
            if tune_loss < best_loss:
                best_loss = tune_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            should_interrupt = stop_after_epoch == epochs_completed
            should_checkpoint = (
                explicit_policy
                and checkpoint is not None
                and (
                    epochs_completed % resolved_policy.checkpoint_every_epochs == 0
                    or should_interrupt
                )
            )
            if should_checkpoint:
                assert checkpoint is not None
                checkpoint_sha256 = _atomic_torch_save(
                    _checkpoint_payload(
                        task_sha256=task_sha256,
                        data_sha256=data_sha256,
                        config_sha256=config_sha256,
                        precision_sha256=precision_sha256,
                        seed=seed,
                        epoch=epochs_completed,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        shuffle_generator=shuffle_generator,
                        best_loss=best_loss,
                        best_epoch=best_epoch,
                        best_state=best_state,
                        stale_epochs=stale_epochs,
                        status="IN_PROGRESS",
                    ),
                    checkpoint,
                )
            if should_interrupt:
                interrupted = True
                break
            if stale_epochs >= resolved_policy.patience:
                break

        if not best_state or not math.isfinite(best_loss):
            raise RuntimeError("neural training produced no finite checkpoint")
        model.load_state_dict(best_state)
        if explicit_policy and checkpoint is not None and not interrupted:
            checkpoint_sha256 = _atomic_torch_save(
                _checkpoint_payload(
                    task_sha256=task_sha256,
                    data_sha256=data_sha256,
                    config_sha256=config_sha256,
                    precision_sha256=precision_sha256,
                    seed=seed,
                    epoch=epochs_completed,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    shuffle_generator=shuffle_generator,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    stale_epochs=stale_epochs,
                    status="COMPLETE",
                ),
                checkpoint,
            )
        model.freeze()
        receipt = TrainingReceipt(
            adapter_name=model.adapter_name,
            seed=seed,
            epochs_completed=epochs_completed,
            best_epoch=best_epoch,
            best_tune_nll=best_loss,
            train_sample_count=train_x.shape[0],
            tune_sample_count=tune_x.shape[0],
            model_hash=model.model_hash,
            device=resolved_policy.device,
            precision=resolved_policy.precision,
            checkpoint_path=checkpoint.as_posix() if checkpoint is not None else None,
            checkpoint_sha256=checkpoint_sha256,
            completed=not interrupted,
        )
        return TrainingResult(model=model, receipt=receipt, checkpoint=checkpoint)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
