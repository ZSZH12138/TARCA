from __future__ import annotations

import hashlib
import math
import os
import random
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor

from tarca.contracts import canonical_json_hash
from tarca.stage1b.evidence_io import sha256_file
from tarca.stage1b.modeling import OfficialOperablePredictor
from tarca.stage1b.neural import OperableNeuralPredictor

Stage1BNeuralPredictor = OperableNeuralPredictor | OfficialOperablePredictor
Precision = Literal["FP32", "AMP_FP16"]


class CheckpointPolicy(Protocol):
    @property
    def device(self) -> str: ...

    @property
    def precision(self) -> Precision: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def max_epochs(self) -> int: ...

    @property
    def patience(self) -> int: ...

    @property
    def learning_rate(self) -> float: ...

    @property
    def dataloader_workers(self) -> int: ...

    @property
    def checkpoint_root(self) -> Path: ...

    @property
    def checkpoint_every_epochs(self) -> int: ...

    @property
    def optimizer(self) -> str: ...

    @property
    def betas(self) -> tuple[float, float]: ...

    @property
    def epsilon(self) -> float: ...

    @property
    def weight_decay(self) -> float: ...

    @property
    def gradient_clip_norm(self) -> float: ...

    @property
    def scheduler(self) -> str: ...

    @property
    def deterministic_algorithms(self) -> bool: ...

    @property
    def cudnn_deterministic(self) -> bool: ...

    @property
    def cudnn_benchmark(self) -> bool: ...


def _tensor_digest(name: str, tensor: Tensor, digest: Any) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes())


def training_data_hash(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        _tensor_digest(name, tensor, digest)
    return digest.hexdigest()


def policy_hash(policy: CheckpointPolicy) -> str:
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
            "optimizer": policy.optimizer,
            "betas": policy.betas,
            "epsilon": policy.epsilon,
            "weight_decay": policy.weight_decay,
            "gradient_clip_norm": policy.gradient_clip_norm,
            "scheduler": policy.scheduler,
            "deterministic_algorithms": policy.deterministic_algorithms,
            "cudnn_deterministic": policy.cudnn_deterministic,
            "cudnn_benchmark": policy.cudnn_benchmark,
        }
    )


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


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def checkpoint_path(policy: CheckpointPolicy, task_sha256: str) -> Path:
    return policy.checkpoint_root / f"training-{task_sha256}.pt"


def resolve_resume_path(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("resume checkpoint must be inside checkpoint_root") from error
    if not resolved.is_file():
        raise ValueError("resume checkpoint must be a regular file")
    return resolved


def load_checkpoint(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or loaded.get("schema_version") != "1.0.0":
        raise ValueError("training checkpoint has an unsupported schema")
    if loaded.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
        raise ValueError("training checkpoint status is invalid")
    return cast(dict[str, Any], loaded)


def checkpoint_payload(
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
        "python_rng_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": _cpu_snapshot(cuda_rng),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "best_state": _cpu_snapshot(dict(best_state)),
        "stale_epochs": stale_epochs,
    }


def restore_checkpoint(
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
    python_rng_state = payload.get("python_rng_state")
    cpu_rng_state = payload.get("torch_cpu_rng_state")
    if (
        not isinstance(shuffle_state, Tensor)
        or not isinstance(python_rng_state, tuple)
        or not isinstance(cpu_rng_state, Tensor)
    ):
        raise ValueError("resume checkpoint is missing deterministic RNG state")
    shuffle_generator.set_state(shuffle_state)
    random.setstate(python_rng_state)
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
