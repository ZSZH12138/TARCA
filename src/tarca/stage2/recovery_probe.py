from __future__ import annotations

import multiprocessing
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import torch

from tarca.contracts import ArtifactRef, DatasetWindowPartition, sha256_file
from tarca.stage1b.training_checkpoints import load_checkpoint
from tarca.stage2.config import load_stage2_config
from tarca.stage2.data import stack_partition
from tarca.stage2.jobs import _bundle_from_payload, _load_torch, _new_neural
from tarca.stage2.recovery import load_stage2_recovery_spec
from tarca.stage2.resources import stage2_reset_time_gate
from tarca.stage2.training import forecast_fixed_batch_on_model_device

_RECOVERY_FIXED_OVERHEAD_SECONDS = 4 * 3600
_RECOVERY_SAFETY_MULTIPLIER = 1.35


def _development_data_ref(database: Path) -> ArtifactRef:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT a.artifact_json
            FROM attempts a JOIN job_nodes j USING(task_id)
            WHERE j.phase = 'DEV_DATA' AND a.state = 'COMPLETED'
              AND a.artifact_json IS NOT NULL
            ORDER BY a.attempt_number DESC
            """
        ).fetchall()
    if len(rows) != 1:
        raise RuntimeError("recovery probe requires one completed development dataset")
    return ArtifactRef.model_validate_json(rows[0][0])


def _probe_complete_checkpoint_worker(
    repository_root: str,
    config_path: str,
    runtime_root: str,
    model_id: str,
    gpu_id: int,
) -> dict[str, Any]:
    root = Path(repository_root)
    runtime = Path(runtime_root)
    config = load_stage2_config(Path(config_path))
    spec = load_stage2_recovery_spec(
        root / "configs/stage2/stage2_device_mismatch_recovery_v1.json"
    )
    seed = config.training.initialization_seeds[0]
    matches = tuple(
        task for task in spec.tasks if task.model_id == model_id and task.seed == seed
    )
    if len(matches) != 1:
        raise RuntimeError("recovery probe task identity is ambiguous")
    task = matches[0]
    checkpoint = root / task.checkpoint_relative_path
    before_sha256 = sha256_file(checkpoint)
    payload = load_checkpoint(checkpoint)
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("seed") != task.seed
        or payload.get("task_sha256") != task.checkpoint_task_sha256
    ):
        raise RuntimeError("recovery probe checkpoint identity does not match")
    best_state = payload.get("best_state")
    if not isinstance(best_state, dict):
        raise RuntimeError("recovery probe checkpoint has no best model state")
    data_ref = _development_data_ref(runtime / "execution.sqlite3")
    bundle = _bundle_from_payload(_load_torch(root, data_ref))
    validation_x, validation_y, _ = stack_partition(
        bundle, DatasetWindowPartition.VALIDATION
    )
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    model = _new_neural(root, config, model_id, validation_x.shape[2])
    model.load_state_dict(cast(dict[str, torch.Tensor], best_state), strict=True)
    model.to(device)
    model.freeze()
    forecast = forecast_fixed_batch_on_model_device(model, validation_x[:2])
    torch.cuda.synchronize(device)
    finite = bool(
        forecast.scale is not None
        and forecast.mean.shape == validation_y[:2].shape
        and torch.isfinite(forecast.mean).all()
        and torch.isfinite(forecast.scale).all()
        and (forecast.scale > 0).all()
    )
    if not finite:
        raise RuntimeError("recovery probe forecast is invalid")
    after_sha256 = sha256_file(checkpoint)
    return {
        "model_id": model_id,
        "gpu_id": gpu_id,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_status": payload["status"],
        "optimizer_steps": 0,
        "forecast_finite": finite,
        "checkpoint_hash_unchanged": before_sha256 == after_sha256 == task.checkpoint_sha256,
    }


def run_stage2_recovery_probe(
    repository_root: Path,
    config_path: Path,
    runtime_root: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    """Validate two recovered COMPLETE checkpoints concurrently without training."""
    root = repository_root.resolve()
    config_file = config_path.resolve()
    runtime = runtime_root.resolve()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = tuple(
            pool.submit(
                _probe_complete_checkpoint_worker,
                str(root),
                str(config_file),
                str(runtime),
                model_id,
                gpu_id,
            )
            for gpu_id, model_id in enumerate(("PATCHTST", "ITRANSFORMER"))
        )
        observations = tuple(future.result() for future in futures)
    if any(
        observation.get("checkpoint_status") != "COMPLETE"
        or observation.get("optimizer_steps") != 0
        or observation.get("forecast_finite") is not True
        or observation.get("checkpoint_hash_unchanged") is not True
        for observation in observations
    ):
        raise RuntimeError("Stage 2 recovery checkpoint probe failed")
    estimated = (
        max(float(observation["elapsed_seconds"]) for observation in observations)
        * 3
        * _RECOVERY_SAFETY_MULTIPLIER
        + _RECOVERY_FIXED_OVERHEAD_SECONDS
    )
    config = load_stage2_config(config_file)
    stage2_reset_time_gate(
        estimated_remaining_seconds=estimated,
        remaining_rental_hours=remaining_rental_hours,
        margin_hours=config.runtime_profile.reset_margin_hours,
    )
    return {
        "probe_contract": "stage2-v1-recovery-two-complete-checkpoints-concurrent",
        "recovery_mode": "DEVICE_MISMATCH_V1",
        "neural_observations": observations,
        "complete_checkpoint_fast_path_passed": True,
        "zero_optimizer_steps": True,
        "checkpoint_hashes_unchanged": True,
        "estimated_remaining_seconds": estimated,
        "reset_margin_hours": config.runtime_profile.reset_margin_hours,
        "eta_gate_passed": True,
    }
