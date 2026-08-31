from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import cast

import torch
from torch import Tensor, nn

from tarca.contracts import ForecastDistribution, WindowBatch, validate_window_batch
from tarca.stage2.distributions import gaussian_forecast, residual_scale, scale_ceiling


@dataclass(frozen=True, slots=True)
class DLinearModelConfig:
    sequence_length: int
    prediction_length: int
    dimension: int
    individual: bool
    moving_average_kernel: int
    asset_relative_path: str
    asset_sha256: str

    def __post_init__(self) -> None:
        if min(
            self.sequence_length,
            self.prediction_length,
            self.dimension,
            self.moving_average_kernel,
        ) <= 0:
            raise ValueError("DLinear dimensions and moving average kernel must be positive")
        path = PurePosixPath(self.asset_relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.asset_relative_path:
            raise ValueError("DLinear asset path must stay below the source root")
        if len(self.asset_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.asset_sha256
        ):
            raise ValueError("DLinear asset hash must be a lowercase SHA-256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_dlinear(source_root: Path, config: DLinearModelConfig) -> nn.Module:
    resolved_root = source_root.resolve()
    source = (resolved_root / config.asset_relative_path).resolve()
    try:
        source.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("DLinear source path escapes the verified source root") from error
    if not source.is_file():
        raise ValueError("DLinear source asset is missing")
    source_bytes = source.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != config.asset_sha256:
        raise ValueError(
            f"DLinear source hash mismatch: expected {config.asset_sha256}, got {actual_hash}"
        )
    module = ModuleType(f"tarca_official_dlinear_{actual_hash[:16]}")
    module.__file__ = str(source)
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("DLinear official source must be UTF-8") from error
    exec(compile(source_text, str(source), "exec"), module.__dict__)
    model_type = getattr(module, "Model", None)
    if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
        raise ValueError("DLinear official source does not expose an nn.Module Model")
    official_config = SimpleNamespace(
        seq_len=config.sequence_length,
        pred_len=config.prediction_length,
        enc_in=config.dimension,
        individual=config.individual,
        moving_avg=config.moving_average_kernel,
    )
    model = model_type(official_config)
    if not isinstance(model, nn.Module):
        raise TypeError("DLinear Model construction did not return an nn.Module")
    return model


def dlinear_fold_index(trajectory_id: str, *, fold_count: int) -> int:
    if not trajectory_id.strip() or fold_count < 2:
        raise ValueError("DLinear fold assignment requires an identity and at least two folds")
    return int(hashlib.sha256(trajectory_id.encode()).hexdigest(), 16) % fold_count


def dlinear_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    return fallback if parameter is None else parameter.device


@dataclass(frozen=True, slots=True)
class DLinearGaussian:
    mean_model: nn.Module
    scale: Tensor
    target_names: tuple[str, ...]
    checkpoint_sha256: str
    scale_source: str = "CROSS_FITTED_TRAIN_ONLY"

    @property
    def adapter_name(self) -> str:
        return "OFFICIAL_DLINEAR_GAUSSIAN"

    @property
    def is_frozen(self) -> bool:
        return True

    @property
    def model_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.adapter_name.encode())
        digest.update(self.checkpoint_sha256.encode())
        digest.update(_tensor_sha256(self.scale).encode())
        digest.update("|".join(self.target_names).encode())
        return digest.hexdigest()

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        validate_window_batch(batch)
        horizon = batch.y.shape[1] if batch.y is not None else len(batch.forecast_time[0])
        if horizon != self.scale.shape[0] or batch.x.shape[2] != self.scale.shape[1]:
            raise ValueError("WindowBatch shape does not match the fitted DLinear model")
        if batch.target_names != self.target_names:
            raise ValueError("WindowBatch target names do not match DLinear")
        device = _model_device(self.mean_model, batch.x.device)
        self.mean_model.eval()
        with torch.no_grad():
            output = self.mean_model(batch.x.to(device))
        expected = (batch.x.shape[0], horizon, len(self.target_names))
        if not isinstance(output, Tensor) or tuple(output.shape) != expected:
            raise ValueError(f"DLinear mean output shape must be {expected}")
        mean = output.to(dtype=batch.x.dtype, device=batch.x.device)
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("DLinear mean output must be finite")
        scale = self.scale.to(dtype=mean.dtype, device=mean.device).unsqueeze(0).expand_as(mean)
        return gaussian_forecast(
            mean,
            scale,
            window_id=batch.window_id,
            target_names=self.target_names,
        )


def save_dlinear_checkpoint(predictor: DLinearGaussian, path: Path) -> str:
    observed_state_hash = dlinear_state_sha256(predictor.mean_model)
    if observed_state_hash != predictor.checkpoint_sha256:
        raise ValueError("DLinear predictor state does not match its checkpoint identity")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "schema_version": "tarca-stage2-dlinear-v1",
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in predictor.mean_model.state_dict().items()
                },
                "scale": predictor.scale.detach().cpu().clone(),
                "target_names": predictor.target_names,
                "checkpoint_sha256": predictor.checkpoint_sha256,
                "scale_source": predictor.scale_source,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def load_dlinear_checkpoint(
    path: Path,
    model_factory: Callable[[], nn.Module],
    *,
    expected_file_sha256: str,
) -> DLinearGaussian:
    if _sha256_file(path) != expected_file_sha256:
        raise ValueError("DLinear checkpoint file SHA-256 mismatch")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("DLinear checkpoint must contain an object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema_version") != "tarca-stage2-dlinear-v1":
        raise ValueError("DLinear checkpoint schema version is unsupported")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in raw_state.items()
    ):
        raise ValueError("DLinear checkpoint state_dict is invalid")
    state = cast(dict[str, Tensor], raw_state)
    scale = payload.get("scale")
    raw_names = payload.get("target_names")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(scale, Tensor) or scale.ndim != 2:
        raise ValueError("DLinear checkpoint scale is invalid")
    if not isinstance(raw_names, (tuple, list)) or any(
        not isinstance(name, str) for name in raw_names
    ):
        raise ValueError("DLinear checkpoint target names are invalid")
    if not isinstance(checkpoint_sha256, str):
        raise ValueError("DLinear checkpoint state identity is invalid")
    model = model_factory()
    model.load_state_dict(state, strict=True)
    if dlinear_state_sha256(model) != checkpoint_sha256:
        raise ValueError("DLinear checkpoint state SHA-256 mismatch")
    return DLinearGaussian(
        mean_model=model,
        scale=scale.to(torch.float64).clone(),
        target_names=tuple(cast(list[str] | tuple[str, ...], raw_names)),
        checkpoint_sha256=checkpoint_sha256,
    )


@dataclass(frozen=True, slots=True)
class DLinearTrainingResult:
    predictor: DLinearGaussian
    checkpoint_sha256: str
    cross_fit_scale_sha256: str
    best_epoch: int
    best_validation_mse: float


@dataclass(frozen=True, slots=True)
class _FitResult:
    model: nn.Module
    best_epoch: int
    best_loss: float


def _set_seed(seed: int) -> None:
    if seed <= 0:
        raise ValueError("DLinear training seeds must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_mean_model(
    model_factory: Callable[[], nn.Module],
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    *,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> _FitResult:
    _set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    best_state: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for start in range(0, train_x.shape[0], batch_size):
            end = min(start + batch_size, train_x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[start:end].to(device))
            loss = torch.mean((prediction - train_y[start:end].to(device)) ** 2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("DLinear training produced a non-finite loss")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x.to(device))
            validation_loss = float(
                torch.mean((validation_prediction - validation_y.to(device)) ** 2)
            )
        if not math.isfinite(validation_loss):
            raise RuntimeError("DLinear validation produced a non-finite loss")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("DLinear training did not retain a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return _FitResult(model=model, best_epoch=best_epoch, best_loss=best_loss)


def fit_dlinear_cross_fitted(
    model_factory: Callable[[], nn.Module],
    train_x: Tensor,
    train_y: Tensor,
    trajectory_ids: tuple[str, ...],
    validation_x: Tensor,
    validation_y: Tensor,
    *,
    target_names: tuple[str, ...],
    fold_seeds: tuple[int, ...],
    final_seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    floor: float = 1e-4,
    ceiling_multiplier: float = 10.0,
    absolute_ceiling: float = 10.0,
    device: str | torch.device = "cpu",
) -> DLinearTrainingResult:
    tensors = (train_x, train_y, validation_x, validation_y)
    if any(tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("DLinear fit tensors must be finite rank-three tensors")
    if train_x.shape[0] != train_y.shape[0] or len(trajectory_ids) != train_x.shape[0]:
        raise ValueError("DLinear TRAIN samples and trajectory IDs must align")
    if validation_x.shape[0] != validation_y.shape[0]:
        raise ValueError("DLinear validation samples must align")
    if train_y.shape[1:] != validation_y.shape[1:]:
        raise ValueError("DLinear TRAIN and validation targets must share shape")
    if len(target_names) != train_y.shape[2] or len(set(target_names)) != len(target_names):
        raise ValueError("DLinear target names must be unique and match target dimension")
    if len(fold_seeds) < 2 or len(set(fold_seeds)) != len(fold_seeds):
        raise ValueError("DLinear cross-fitting requires unique fold seeds")
    if min(batch_size, max_epochs, patience) <= 0:
        raise ValueError("DLinear training counts must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("DLinear optimizer values are invalid")
    resolved_device = torch.device(device)
    fold_assignments = torch.tensor(
        [dlinear_fold_index(item, fold_count=len(fold_seeds)) for item in trajectory_ids]
    )
    out_of_fold = torch.empty_like(train_y, dtype=torch.float64)
    for fold, seed in enumerate(fold_seeds):
        held_out = fold_assignments == fold
        training = ~held_out
        if not bool(held_out.any()) or not bool(training.any()):
            raise ValueError("each DLinear cross-fit fold must contain held-out and training rows")
        fitted = _train_mean_model(
            model_factory,
            train_x[training],
            train_y[training],
            train_x[training],
            train_y[training],
            seed=seed,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=resolved_device,
        )
        with torch.no_grad():
            prediction = fitted.model(train_x[held_out].to(resolved_device))
        out_of_fold[held_out] = prediction.to(torch.float64).cpu()
    cross_fit_scale = residual_scale(
        train_y.to(torch.float64) - out_of_fold,
        floor=floor,
        ceiling=scale_ceiling(
            train_y,
            multiplier=ceiling_multiplier,
            absolute_ceiling=absolute_ceiling,
        ),
    )
    final = _train_mean_model(
        model_factory,
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=final_seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=resolved_device,
    )
    checkpoint_sha256 = dlinear_state_sha256(final.model)
    scale_sha256 = _tensor_sha256(cross_fit_scale)
    predictor = DLinearGaussian(
        mean_model=final.model,
        scale=cross_fit_scale,
        target_names=target_names,
        checkpoint_sha256=checkpoint_sha256,
    )
    return DLinearTrainingResult(
        predictor=predictor,
        checkpoint_sha256=checkpoint_sha256,
        cross_fit_scale_sha256=scale_sha256,
        best_epoch=final.best_epoch,
        best_validation_mse=final.best_loss,
    )
