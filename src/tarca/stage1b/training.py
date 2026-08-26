from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tarca.stage1b.modeling import OfficialOperablePredictor
from tarca.stage1b.neural import OperableNeuralPredictor

Stage1BNeuralPredictor = OperableNeuralPredictor | OfficialOperablePredictor


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


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: Stage1BNeuralPredictor
    receipt: TrainingReceipt


def _gaussian_nll(mean: Tensor, scale: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.log(scale) + 0.5 * ((target - mean) / scale) ** 2)


def train_candidate(
    model: Stage1BNeuralPredictor,
    train_x: Tensor,
    train_y: Tensor,
    tune_x: Tensor,
    tune_y: Tensor,
    *,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
) -> TrainingResult:
    if any(tensor.ndim != 3 for tensor in (train_x, train_y, tune_x, tune_y)):
        raise ValueError("neural training tensors must have rank three")
    if train_x.shape[0] != train_y.shape[0] or tune_x.shape[0] != tune_y.shape[0]:
        raise ValueError("neural histories and targets must align by sample")
    if batch_size <= 0 or max_epochs <= 0 or not 0 <= patience < max_epochs:
        raise ValueError("invalid neural training budget")
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model.reset_for_training(seed)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        shuffle_generator = torch.Generator().manual_seed(seed)
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0
        epochs_completed = 0
        for epoch in range(max_epochs):
            model.train()
            permutation = torch.randperm(train_x.shape[0], generator=shuffle_generator)
            for start in range(0, train_x.shape[0], batch_size):
                rows = permutation[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                distribution = model.forward_distribution(train_x[rows])
                if distribution.scale is None:
                    raise RuntimeError("neural candidate did not emit probabilistic scale")
                loss = _gaussian_nll(distribution.mean, distribution.scale, train_y[rows])
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                tune_distribution = model.forward_distribution(tune_x)
                if tune_distribution.scale is None:
                    raise RuntimeError("neural candidate did not emit probabilistic scale")
                tune_loss = float(
                    _gaussian_nll(tune_distribution.mean, tune_distribution.scale, tune_y)
                )
            epochs_completed = epoch + 1
            if tune_loss < best_loss:
                best_loss = tune_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is None:
            raise RuntimeError("neural training produced no finite checkpoint")
        model.load_state_dict(best_state)
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
        )
        return TrainingResult(model=model, receipt=receipt)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
