from __future__ import annotations

import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psutil
import torch

from tarca.stage1b.config import (
    QualificationPartition,
    TrajectoryPartitionCounts,
    WorldRole,
    load_qualification_config,
    load_world_suite,
)
from tarca.stage1b.dataset import generate_world_split, prepare_dataset, stack_partition
from tarca.stage1b.evidence_io import sha256_file
from tarca.stage1b.hardware import estimate_full_run, inventory_hardware
from tarca.stage1b.training import TrainingPolicy, train_candidate
from tarca.stage1b.worlds import build_world


def run_hardware_probe(
    worlds_path: Path,
    qualification_path: Path,
    runtime_root: Path,
    *,
    authorized_over_24_hours: bool = False,
    device: str | None = None,
    precision: str = "FP32",
    dataloader_workers: int | None = None,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    from tarca.stage1b.runner import (
        _full_work_units,
        _jsonable,
        _new_model,
        _operability_smoke,
        _repeat_first_axis,
        _training_seed,
        _write_json,
    )

    suite = load_world_suite(worlds_path)
    qualification = load_qualification_config(qualification_path)
    inventory = inventory_hardware()
    primary_worlds = (
        tuple(suite.world(world_id) for world_id in qualification.target_world_ids)
        if qualification.target_world_ids
        else tuple(
            world for world in suite.worlds if world.role is WorldRole.PRIMARY_MECHANISTIC
        )
    )
    resolved_device = device or ("cuda" if inventory.gpu_vram_bytes else "cpu")
    resolved_workers = (
        dataloader_workers
        if dataloader_workers is not None
        else (3 if resolved_device.startswith("cuda") else 0)
    )
    if type(resolved_workers) is not int or resolved_workers < 0:
        raise ValueError("hardware probe DataLoader workers must be a non-negative integer")
    if resolved_device == "cuda":
        parallel_work_slots = len(inventory.gpu_vram_bytes)
    elif resolved_device.startswith("cuda:"):
        parallel_work_slots = 1
    else:
        parallel_work_slots = 1
    if parallel_work_slots <= 0:
        raise RuntimeError("hardware probe requested CUDA but no CUDA device is available")
    probe_qualification = qualification.model_copy(
        update={
            "trajectories_per_partition": TrajectoryPartitionCounts(
                QUAL_TRAIN=2,
                QUAL_TUNE=1,
                QUAL_SEEN=1,
                QUAL_UNSEEN=1,
            ),
        }
    )
    process = psutil.Process()
    memory_before = process.memory_info().rss
    peak_memory = memory_before
    windows_per_trajectory = (
        qualification.trajectory_length - qualification.history_length - qualification.horizon + 1
    )
    full_train_samples = (
        qualification.trajectories_per_partition.qual_train * windows_per_trajectory
    )
    full_tune_samples = qualification.trajectories_per_partition.qual_tune * windows_per_trajectory
    probe_trajectory_count = sum(
        (
            probe_qualification.trajectories_per_partition.qual_train,
            probe_qualification.trajectories_per_partition.qual_tune,
            probe_qualification.trajectories_per_partition.qual_seen,
            probe_qualification.trajectories_per_partition.qual_unseen,
        )
    )
    full_trajectory_count = sum(
        (
            qualification.trajectories_per_partition.qual_train,
            qualification.trajectories_per_partition.qual_tune,
            qualification.trajectories_per_partition.qual_seen,
            qualification.trajectories_per_partition.qual_unseen,
        )
    )
    generation_projection_seconds = 0.0
    serial_training_projection_seconds = 0.0
    world_observations: list[dict[str, Any]] = []
    training_observations: list[dict[str, Any]] = []
    raw_probe_seconds = 0.0
    probe_work_units = 0
    runtime_root = runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tarca-stage1b-hardware-probe-") as checkpoint:
        checkpoint_root = Path(checkpoint)
        for primary in primary_worlds:
            generation_started = time.perf_counter()
            split = generate_world_split(
                build_world(primary),
                probe_qualification,
                qualification_seed=qualification.qualification_seeds[0],
                source_commit=suite.source(primary.source_id).commit,
            )
            dataset = prepare_dataset(
                split,
                history_length=qualification.history_length,
                horizon=qualification.horizon,
            )
            generation_seconds = time.perf_counter() - generation_started
            raw_probe_seconds += generation_seconds
            projected_generation = (
                generation_seconds
                * full_trajectory_count
                / probe_trajectory_count
                * len(qualification.qualification_seeds)
            )
            generation_projection_seconds += projected_generation
            world_observations.append(
                {
                    "world_id": primary.world_id,
                    "probe_seconds": generation_seconds,
                    "probe_trajectories": probe_trajectory_count,
                    "projected_trajectories": full_trajectory_count,
                    "projected_seconds_all_seeds": projected_generation,
                }
            )
            train_x, train_y = stack_partition(dataset, QualificationPartition.QUAL_TRAIN)
            tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
            formal_train_x = _repeat_first_axis(train_x, full_train_samples)
            formal_train_y = _repeat_first_axis(train_y, full_train_samples)
            formal_tune_x = _repeat_first_axis(tune_x, full_tune_samples)
            formal_tune_y = _repeat_first_axis(tune_y, full_tune_samples)
            peak_memory = max(peak_memory, process.memory_info().rss)
            for model_config in qualification.models:
                model = _new_model(model_config, primary, qualification)
                policy = TrainingPolicy(
                    device=resolved_device,
                    precision=cast(Any, precision),
                    batch_size=model_config.batch_size,
                    max_epochs=1,
                    patience=0,
                    learning_rate=model_config.learning_rate,
                    dataloader_workers=resolved_workers,
                    checkpoint_root=checkpoint_root,
                )
                training_started = time.perf_counter()
                result = train_candidate(
                    model,
                    formal_train_x,
                    formal_train_y,
                    formal_tune_x,
                    formal_tune_y,
                    seed=_training_seed(
                        primary.world_id,
                        qualification.qualification_seeds[0],
                        model.adapter_name,
                    ),
                    policy=policy,
                )
                training_seconds = time.perf_counter() - training_started
                if result.receipt.device != resolved_device or not result.receipt.completed:
                    raise RuntimeError("representative neural candidate used the wrong policy")
                result.model.to("cpu")
                if not _operability_smoke(result.model, dataset):
                    raise RuntimeError("representative neural candidate failed operability smoke")
                if resolved_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                batch_units = math.ceil(full_train_samples / model_config.batch_size) + math.ceil(
                    full_tune_samples / model_config.batch_size
                )
                raw_probe_seconds += training_seconds
                probe_work_units += batch_units
                projected_training = (
                    training_seconds
                    * model_config.max_epochs
                    * len(qualification.qualification_seeds)
                )
                serial_training_projection_seconds += projected_training
                training_observations.append(
                    {
                        "world_id": primary.world_id,
                        "model_id": model_config.model_id,
                        "device": resolved_device,
                        "precision": precision,
                        "dataloader_workers": resolved_workers,
                        "probe_seconds": training_seconds,
                        "probe_epochs": 1,
                        "probe_work_units": batch_units,
                        "projected_epochs_all_seeds": (
                            model_config.max_epochs * len(qualification.qualification_seeds)
                        ),
                        "projected_serial_seconds": projected_training,
                    }
                )
                peak_memory = max(peak_memory, process.memory_info().rss)
    full_units = _full_work_units(suite, qualification)
    memory_after = process.memory_info().rss
    memory_growth = max(peak_memory - memory_before, 64 * 1024**2)
    trajectory_ratio = full_trajectory_count / probe_trajectory_count
    projected_memory = int(
        max(memory_growth * trajectory_ratio, memory_growth * parallel_work_slots)
    )
    decision = estimate_full_run(
        probe_seconds=serial_training_projection_seconds,
        probe_work_units=1,
        full_work_units=1,
        projected_peak_memory_bytes=projected_memory,
        available_memory_bytes=inventory.available_memory_bytes,
        authorized_over_24_hours=authorized_over_24_hours,
        parallel_work_slots=parallel_work_slots,
        fixed_seconds=generation_projection_seconds,
        safety_factor=safety_factor,
    )
    receipt = {
        "schema_version": "2.0.0",
        "probe_id": "stage1b-hardware-probe-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "world_config_sha256": sha256_file(worlds_path),
        "qualification_config_sha256": sha256_file(qualification_path),
        "source_manifest_sha256": suite.source_manifest_sha256(),
        "source_commits": {source.source_id: source.commit for source in suite.sources},
        "probe_world_ids": [world.world_id for world in primary_worlds],
        "probe_adapters": [model.adapter.value for model in qualification.models],
        "probe_device": resolved_device,
        "probe_precision": precision,
        "parallel_work_slots": parallel_work_slots,
        "safety_factor": safety_factor,
        "probe_seconds": raw_probe_seconds,
        "probe_work_units": probe_work_units,
        "full_work_units": full_units,
        "generation_projection_seconds": generation_projection_seconds,
        "serial_training_projection_seconds": serial_training_projection_seconds,
        "world_observations": world_observations,
        "training_observations": training_observations,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "inventory": _jsonable(inventory),
        "decision": _jsonable(decision),
        "minimum_server": {
            "physical_cpu_cores": 20,
            "ram_gib": 128,
            "gpu": "at least one CUDA GPU with 20 GiB usable VRAM",
            "storage_gib": 100,
            "target_runtime_hours": "at most 120",
        },
        "recommended_server": {
            "policy": "use every safely admitted device from this receipt",
            "physical_cpu_cores": inventory.physical_cpu_count,
            "ram_gib": inventory.available_memory_bytes // 1024**3,
            "gpu_count": len(inventory.gpu_vram_bytes),
            "gpu_vram_gib": [value / 1024**3 for value in inventory.gpu_vram_bytes],
            "estimated_hours": decision.estimated_hours,
        },
    }
    _write_json(runtime_root.resolve() / "hardware_probe_v2.json", receipt)
    return receipt
