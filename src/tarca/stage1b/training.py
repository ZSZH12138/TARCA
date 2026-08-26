from __future__ import annotations

import hashlib
import math
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tarca.contracts import canonical_json_hash
from tarca.stage1b.modeling import OfficialOperablePredictor
from tarca.stage1b.neural import OperableNeuralPredictor

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


def _tensor_digest(name: str, tensor: Tensor, digest: Any) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())


def _training_data_hash(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        _tensor_digest(name, tensor, digest)
    return digest.hexdigest()


def _policy_hash(policy: TrainingPolicy) -> str:
    return canonical_json_hash(
        {
            "device": policy.device,
            "precision": policy.precision,
            "batch_size": policy.batch_size,
            "max_epochs": policy.max_epochs,
            "patience": policy.patience,
            "learning_rate": policy.learning_rate,
            "dataloader_workers": policy.dataloader_workers,
            "checkpoint_every_epochs": policy.checkpoint_every_epochs,
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(destination)


def _checkpoint_path(policy: TrainingPolicy, task_sha256: str) -> Path:
    return policy.checkpoint_root / f"training-{task_sha256}.pt"


def _resolve_resume_path(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("resume checkpoint must be inside checkpoint_root") from error
    if not resolved.is_file():
        raise ValueError("resume checkpoint must be a regular file")
    return resolved


def _load_checkpoint(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or loaded.get("schema_version") != "1.0.0":
        raise ValueError("training checkpoint has an unsupported schema")
    return cast(dict[str, Any], loaded)


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


def _checkpoint_payload(
    *,
    task_sha256: str,
    data_sha256: str,
    config_sha256: str,
    precision_sha256: str,
    seed: int,
    epoch: int,
    model: Stage1BNeuralPredictor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    shuffle_generator: torch.Generator,
    best_loss: float,
    best_epoch: int,
    best_state: Mapping[str, Tensor],
    stale_epochs: int,
    status: Literal["IN_PROGRESS", "COMPLETE"],
) -> dict[str, Any]:
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "schema_version": "1.0.0",
        "task_sha256": task_sha256,
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "precision_sha256": precision_sha256,
        "seed": seed,
        "epoch": epoch,
        "status": status,
        "model_state": _cpu_snapshot(model.state_dict()),
        "optimizer_state": _cpu_snapshot(optimizer.state_dict()),
        "scaler_state": _cpu_snapshot(scaler.state_dict()),
        "shuffle_generator_state": shuffle_generator.get_state().clone(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": _cpu_snapshot(cuda_rng),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_state": _cpu_snapshot(dict(best_state)),
        "stale_epochs": stale_epochs,
    }


def _restore_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_task_sha256: str,
    expected_data_sha256: str,
    expected_config_sha256: str,
    expected_precision_sha256: str,
    model: Stage1BNeuralPredictor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    shuffle_generator: torch.Generator,
    use_cuda: bool,
) -> tuple[int, float, int, dict[str, Tensor], int]:
    if payload.get("data_sha256") != expected_data_sha256:
        raise ValueError("resume checkpoint does not match the training data")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("resume checkpoint does not match the training configuration")
    if payload.get("precision_sha256") != expected_precision_sha256:
        raise ValueError("resume checkpoint does not match device and precision")
    if payload.get("task_sha256") != expected_task_sha256:
        raise ValueError("resume checkpoint does not match the training task")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    scaler_state = payload.get("scaler_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("resume checkpoint is missing model or optimizer state")
    if not isinstance(scaler_state, dict):
        raise ValueError("resume checkpoint is missing scaler state")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scaler.load_state_dict(scaler_state)
    shuffle_state = payload.get("shuffle_generator_state")
    cpu_rng_state = payload.get("torch_cpu_rng_state")
    if not isinstance(shuffle_state, Tensor) or not isinstance(cpu_rng_state, Tensor):
        raise ValueError("resume checkpoint is missing deterministic RNG state")
    shuffle_generator.set_state(shuffle_state)
    torch.set_rng_state(cpu_rng_state)
    cuda_states = payload.get("torch_cuda_rng_states")
    if use_cuda:
        if not isinstance(cuda_states, list) or not all(
            isinstance(state, Tensor) for state in cuda_states
        ):
            raise ValueError("resume checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_states)
    best_state_value = payload.get("best_state")
    if not isinstance(best_state_value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, Tensor)
        for name, tensor in best_state_value.items()
    ):
        raise ValueError("resume checkpoint is missing the best model state")
    epoch = payload.get("epoch")
    best_loss = payload.get("best_loss")
    best_epoch = payload.get("best_epoch")
    stale_epochs = payload.get("stale_epochs")
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("resume checkpoint epoch is invalid")
    if not isinstance(best_loss, float) or not math.isfinite(best_loss):
        raise ValueError("resume checkpoint best loss is invalid")
    if not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("resume checkpoint best epoch is invalid")
    if not isinstance(stale_epochs, int) or stale_epochs < 0:
        raise ValueError("resume checkpoint stale epoch count is invalid")
    return epoch, best_loss, best_epoch, cast(dict[str, Tensor], best_state_value), stale_epochs


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
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= resolved_policy.max_epochs:
        raise ValueError("stop_after_epoch must be within the training budget")
    device = torch.device(resolved_policy.device)
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but CUDA is unavailable")
    if resolved_policy.precision == "AMP_FP16" and not use_cuda:
        raise ValueError("AMP_FP16 requires CUDA")
    if not explicit_policy and (resume_from is not None or stop_after_epoch is not None):
        raise ValueError("checkpoint and interruption controls require an explicit policy")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
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
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=resolved_policy.learning_rate)
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
        if resume_from is not None:
            resume_path = _resolve_resume_path(resume_from, resolved_policy.checkpoint_root)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
