from __future__ import annotations

import math
import multiprocessing
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import torch

from tarca.stage2.config import ModelId, load_stage2_config
from tarca.stage2.jobs import _new_neural
from tarca.stage2.resources import stage2_reset_time_gate

_MODEL_IDS = ("PATCHTST", "ITRANSFORMER")
_PROBE_STEPS = 8
_WARMUP_STEPS = 2
_FIXED_OVERHEAD_SECONDS = 4 * 3600
_SAFETY_MULTIPLIER = 1.35


def estimate_stage2_critical_path_seconds(
    *,
    train_window_count: int,
    initialization_count: int,
    maximum_epochs: Mapping[str, int],
    batches_per_second: Mapping[str, float],
    batch_sizes: Mapping[str, int],
    checkpoint_seconds: Mapping[str, float],
    fixed_overhead_seconds: float = _FIXED_OVERHEAD_SECONDS,
    safety_multiplier: float = _SAFETY_MULTIPLIER,
) -> float:
    """Project the two-GPU critical path without changing the scientific workload."""
    positive = (
        train_window_count,
        initialization_count,
        fixed_overhead_seconds,
        safety_multiplier,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
        raise ValueError("Stage 2 ETA projection inputs must be finite and positive")
    projected: list[float] = []
    for model_id in _MODEL_IDS:
        epochs = maximum_epochs[model_id]
        rate = batches_per_second[model_id]
        batch_size = batch_sizes[model_id]
        checkpoint = checkpoint_seconds[model_id]
        if epochs <= 0 or rate <= 0 or batch_size <= 0 or checkpoint < 0:
            raise ValueError("Stage 2 model probe values are outside their valid range")
        sample_epochs = train_window_count * initialization_count * epochs
        training_seconds = sample_epochs / (rate * batch_size)
        projected.append(training_seconds + initialization_count * checkpoint)
    return max(projected) * safety_multiplier + fixed_overhead_seconds


def _gaussian_loss(model: Any, histories: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    forecast = model.forward_distribution(histories)
    if forecast.scale is None:
        raise RuntimeError("Stage 2 server probe requires a probabilistic predictor")
    scale = forecast.scale.clamp_min(1e-4)
    return cast(
        torch.Tensor,
        (((forecast.mean - targets) / scale).square() + 2.0 * scale.log()).mean(),
    )


def _probe_neural_worker(
    repository_root: str,
    config_path: str,
    runtime_root: str,
    model_id: str,
    gpu_id: int,
) -> dict[str, float | str | int | bool]:
    root = Path(repository_root)
    config = load_stage2_config(Path(config_path))
    definition = config.model(cast(ModelId, model_id))
    batch_size = int(definition.parameter("batch_size"))
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(172657089 + gpu_id)
    model = _new_neural(root, config, model_id, 8).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(definition.parameter("learning_rate")),
        betas=config.training.betas,
        eps=config.training.epsilon,
        weight_decay=config.training.neural_weight_decay,
    )
    histories = torch.randn(batch_size, config.data.history, 8, device=device)
    targets = torch.randn(batch_size, config.data.horizon, 8, device=device)

    for _ in range(_WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        _gaussian_loss(model, histories, targets).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(_PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = _gaussian_loss(model, histories, targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{model_id} server probe produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    checkpoint = Path(runtime_root) / f"probe-{model_id.lower()}-gpu{gpu_id}.pt"
    checkpoint_started = time.perf_counter()
    torch.save(model.state_dict(), checkpoint)
    reloaded = _new_neural(root, config, model_id, 8).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    reloaded.eval()
    with torch.no_grad():
        forecast = reloaded.forward_distribution(histories[:1])
    torch.cuda.synchronize(device)
    checkpoint_elapsed = time.perf_counter() - checkpoint_started
    checkpoint.unlink(missing_ok=True)
    finite = bool(
        torch.isfinite(forecast.mean).all()
        and forecast.scale is not None
        and torch.isfinite(forecast.scale).all()
    )
    if not finite:
        raise RuntimeError(f"{model_id} checkpoint reload probe was not finite")
    return {
        "model_id": model_id,
        "gpu_id": gpu_id,
        "batch_size": batch_size,
        "batches_per_second": _PROBE_STEPS / elapsed,
        "checkpoint_seconds": checkpoint_elapsed,
        "checkpoint_reload_finite": finite,
    }


def _linear_probe_seconds() -> float:
    generator = torch.Generator(device="cpu").manual_seed(172657089)
    design = torch.randn(8192, 64, generator=generator)
    targets = torch.randn(8192, 24, generator=generator)
    started = time.perf_counter()
    solution = torch.linalg.lstsq(design, targets).solution
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError("Stage 2 linear server probe was not finite")
    return elapsed


def run_stage2_server_probe(
    repository_root: Path,
    config_path: Path,
    runtime_root: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    """Run two exact neural probes concurrently and enforce the 24-hour reset gate."""
    root = repository_root.resolve()
    config_file = config_path.resolve()
    config = load_stage2_config(config_file)
    runtime = runtime_root.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = tuple(
            pool.submit(
                _probe_neural_worker,
                str(root),
                str(config_file),
                str(runtime),
                model_id,
                gpu_id,
            )
            for gpu_id, model_id in enumerate(_MODEL_IDS)
        )
        observations = tuple(future.result() for future in futures)
    by_model = {str(item["model_id"]): item for item in observations}
    windows_per_trajectory = (
        config.data.trajectory_length - config.data.history - config.data.horizon + 1
    )
    train_windows = (
        windows_per_trajectory
        * config.data.train_trajectories_per_seed
        * len(config.data.development_seeds)
    )
    estimated = estimate_stage2_critical_path_seconds(
        train_window_count=train_windows,
        initialization_count=len(config.training.initialization_seeds),
        maximum_epochs={
            model_id: int(config.model(cast(ModelId, model_id)).parameter("max_epochs"))
            for model_id in _MODEL_IDS
        },
        batches_per_second={
            model_id: float(by_model[model_id]["batches_per_second"])
            for model_id in _MODEL_IDS
        },
        batch_sizes={
            model_id: int(by_model[model_id]["batch_size"]) for model_id in _MODEL_IDS
        },
        checkpoint_seconds={
            model_id: float(by_model[model_id]["checkpoint_seconds"])
            for model_id in _MODEL_IDS
        },
    )
    stage2_reset_time_gate(
        estimated_remaining_seconds=estimated,
        remaining_rental_hours=remaining_rental_hours,
        margin_hours=config.runtime_profile.reset_margin_hours,
    )
    return {
        "probe_contract": "stage2-v1-two-exact-neural-concurrent-max-epochs",
        "neural_observations": observations,
        "linear_probe_seconds": _linear_probe_seconds(),
        "train_window_count": train_windows,
        "estimated_remaining_seconds": estimated,
        "reset_margin_hours": config.runtime_profile.reset_margin_hours,
        "eta_gate_passed": True,
    }
