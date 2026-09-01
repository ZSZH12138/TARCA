from __future__ import annotations

import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from tarca.contracts import (
    ArtifactRef,
    DatasetWindowPartition,
    ForecastDistribution,
    ForecastPredictor,
    sha256_file,
)
from tarca.e02.bootstrap import paired_stratified_bootstrap
from tarca.e02.config import load_e02_config
from tarca.e02.jobs import _stage2_completed, _verify_stage2_artifacts
from tarca.e02.scoring import TrajectoryLineage, TrajectoryScore, score_trajectory
from tarca.stage1b.config import QualificationPartition, load_world_suite
from tarca.stage1b.worlds import SimulationRequest, build_world
from tarca.stage2.config import load_stage2_config
from tarca.stage2.data import stack_partition
from tarca.stage2.jobs import (
    _baseline_from_payload,
    _bundle_from_payload,
    _load_torch,
    _new_neural,
    _window_batch,
)
from tarca.stage2.manifest import stage2_manifest_from_payload
from tarca.stage2.resources import stage2_reset_time_gate
from tarca.stage2.seeds import derive_namespaced_seed

_SAMPLE_WINDOWS = 2048
_FIXED_OVERHEAD_SECONDS = 30 * 60
_SAFETY_MULTIPLIER = 1.35


def estimate_e02_critical_path_seconds(
    *,
    formal_trajectory_count: int,
    windows_per_trajectory: int,
    generation_trajectories_per_second: float,
    neural_windows_per_second: tuple[float, float, float],
    neural_startup_seconds: tuple[float, float, float],
    linear_windows_per_second: float,
    scoring_windows_per_second: float,
    bootstrap_seconds: float,
    score_parallel_tasks: int,
    fixed_overhead_seconds: float = _FIXED_OVERHEAD_SECONDS,
    safety_multiplier: float = _SAFETY_MULTIPLIER,
) -> float:
    """Project the dependency-aware E02 critical path without opening formal data."""
    scalars = (
        formal_trajectory_count,
        windows_per_trajectory,
        generation_trajectories_per_second,
        linear_windows_per_second,
        scoring_windows_per_second,
        bootstrap_seconds,
        score_parallel_tasks,
        fixed_overhead_seconds,
        safety_multiplier,
        *neural_windows_per_second,
        *neural_startup_seconds,
    )
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("E02 ETA inputs must be finite")
    if (
        formal_trajectory_count <= 0
        or windows_per_trajectory <= 0
        or generation_trajectories_per_second <= 0
        or linear_windows_per_second <= 0
        or scoring_windows_per_second <= 0
        or bootstrap_seconds < 0
        or score_parallel_tasks <= 0
        or fixed_overhead_seconds < 0
        or safety_multiplier < 1.0
        or any(value <= 0 for value in neural_windows_per_second)
        or any(value < 0 for value in neural_startup_seconds)
    ):
        raise ValueError("E02 ETA inputs are outside their valid conservative range")
    formal_windows = formal_trajectory_count * windows_per_trajectory
    neural_seconds = tuple(
        startup + formal_windows / rate
        for startup, rate in zip(
            neural_startup_seconds, neural_windows_per_second, strict=True
        )
    )
    # Init 0/1 are the first dual-GPU wave; init 2 is the second wave.
    neural_critical = max(neural_seconds[:2]) + neural_seconds[2]
    linear_seconds = formal_windows / linear_windows_per_second
    prediction_critical = max(neural_critical, linear_seconds)
    score_waves = math.ceil(4 / score_parallel_tasks)
    scoring_seconds = score_waves * formal_windows / scoring_windows_per_second
    generation_seconds = formal_trajectory_count / generation_trajectories_per_second
    measured_path = generation_seconds + prediction_critical + scoring_seconds + bootstrap_seconds
    return measured_path * safety_multiplier + fixed_overhead_seconds


def _verified_probe_inputs(root: Path) -> dict[str, Any]:
    manifest_payload = json.loads(
        (root / "artifacts/stage2/frozen/v1/stage2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(manifest_payload, dict):
        raise RuntimeError("frozen Stage 2 manifest is invalid")
    manifest = stage2_manifest_from_payload(manifest_payload)
    verified = _verify_stage2_artifacts(root, manifest, _stage2_completed(root))
    return {**verified, "stage2_manifest": manifest.payload()}


def _probe_neural_checkpoint_worker(
    repository_root: str,
    development_data_ref: dict[str, Any],
    checkpoint_ref: dict[str, Any],
    checkpoint_index: int,
    gpu_id: int,
) -> dict[str, object]:
    root = Path(repository_root)
    data_ref = ArtifactRef.model_validate(development_data_ref)
    model_ref = ArtifactRef.model_validate(checkpoint_ref)
    if model_ref.relative_path is None:
        raise RuntimeError("E02 neural probe checkpoint requires a repository-relative path")
    checkpoint_path = root / model_ref.relative_path
    before = sha256_file(checkpoint_path)
    bundle = _bundle_from_payload(_load_torch(root, data_ref))
    validation_x, validation_y, _ = stack_partition(
        bundle, DatasetWindowPartition.VALIDATION
    )
    sample_count = min(_SAMPLE_WINDOWS, validation_x.shape[0])
    if sample_count < 512:
        raise RuntimeError("E02 neural probe requires at least 512 validation windows")
    checkpoint = _load_torch(root, model_ref)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    model = _new_neural(
        root,
        load_stage2_config(root / "configs/stage2/stage2_v1.yaml"),
        "ITRANSFORMER",
        validation_x.shape[2],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if model.model_hash != checkpoint["model_sha256"]:
        raise RuntimeError("E02 probe checkpoint model identity drifted")
    model.to(device).freeze()
    startup_seconds = time.perf_counter() - started
    batch_size = 512
    means: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    with torch.no_grad():
        warmup = validation_x[:batch_size].to(device, non_blocking=True)
        model.forward_distribution(warmup)
        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        for start in range(0, sample_count, batch_size):
            values = validation_x[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=str(checkpoint["precision"]) == "AMP_FP16",
            ):
                forecast = model.forward_distribution(values)
            if forecast.scale is None:
                raise RuntimeError("E02 probe checkpoint has no Gaussian scale")
            means.append(forecast.mean.float().cpu())
            scales.append(forecast.scale.float().cpu())
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - inference_started
    mean, scale = torch.cat(means), torch.cat(scales)
    finite = bool(
        mean.shape == validation_y[:sample_count].shape
        and torch.isfinite(mean).all()
        and torch.isfinite(scale).all()
    )
    positive = bool((scale > 0).all())
    after = sha256_file(checkpoint_path)
    if not finite or not positive or before != after or before != model_ref.content_hash:
        raise RuntimeError("E02 frozen checkpoint probe failed")
    return {
        "checkpoint_index": checkpoint_index,
        "stage2_seed": int(checkpoint["seed"]),
        "gpu_id": gpu_id,
        "sample_window_count": sample_count,
        "windows_per_second": sample_count / elapsed,
        "startup_seconds": startup_seconds,
        "forecast_finite": finite,
        "positive_scales": positive,
        "checkpoint_hash_unchanged": True,
    }


def _score_template(value: float) -> TrajectoryScore:
    return TrajectoryScore(
        trajectory_id="probe",
        formal_seed=1729,
        regime="SEEN",
        origin_count=1,
        horizon_count=24,
        variable_count=8,
        crps=value,
        nll=value,
        mae=value,
        coverage=((0.5, 0.5), (0.8, 0.8), (0.9, 0.9), (0.95, 0.95)),
        horizon_crps=(value,) * 24,
        horizon_nll=(value,) * 24,
        horizon_mae=(value,) * 24,
        horizon_coverage=tuple(
            (level, (level,) * 24) for level in (0.5, 0.8, 0.9, 0.95)
        ),
    )


def _probe_cpu_path(root: Path, inputs: dict[str, Any]) -> dict[str, float]:
    stage2_config = load_stage2_config(root / "configs/stage2/stage2_v1.yaml")
    e02_config = load_e02_config(root / "configs/e02/e02_v1.yaml")
    world = build_world(
        load_world_suite(root / "configs/stage1b/worlds_v2.yaml").world(
            stage2_config.upstream.world_id
        )
    )
    seen = next(regime for regime in world.config.regimes if regime.split_role.value == "SEEN")
    generation_started = time.perf_counter()
    generated = world.simulate(
        SimulationRequest(
            seed=derive_namespaced_seed("tarca/e02_predictor_validity_v1/server-probe"),
            partition=QualificationPartition.QUAL_TUNE,
            regime_id=seen.regime_id,
            length=stage2_config.data.trajectory_length,
            warmup_steps=stage2_config.data.warmup_steps,
        )
    )
    generation_seconds = time.perf_counter() - generation_started
    if not bool(torch.isfinite(generated.values).all()):
        raise RuntimeError("E02 development-only generation probe was not finite")

    data_ref = ArtifactRef.model_validate(inputs["development_data_ref"])
    bundle = _bundle_from_payload(_load_torch(root, data_ref))
    batch = _window_batch(bundle)
    manifest = stage2_manifest_from_payload(inputs["stage2_manifest"])
    baseline_ref = ArtifactRef.model_validate(
        inputs["baseline_refs"][manifest.strongest_linear.model_id]
    )
    baseline_value = _baseline_from_payload(
        root, stage2_config, _load_torch(root, baseline_ref)
    )
    if not isinstance(baseline_value, ForecastPredictor):
        raise RuntimeError("E02 strongest-linear probe is not a forecast predictor")
    baseline = baseline_value
    linear_started = time.perf_counter()
    distribution = baseline.predict_distribution(batch)
    linear_seconds = time.perf_counter() - linear_started
    if distribution.scale is None or batch.y is None:
        raise RuntimeError("E02 linear probe requires Gaussian scale and validation targets")

    score_count = min(425, batch.y.shape[0])
    scored_distribution = ForecastDistribution(
        mean=distribution.mean[:score_count],
        scale=distribution.scale[:score_count],
        quantiles={},
        logits=None,
        samples=None,
        window_id=None,
        target_names=distribution.target_names,
    )
    score_started = time.perf_counter()
    score_trajectory(
        scored_distribution,
        batch.y[:score_count],
        TrajectoryLineage("development-probe", 1, "SEEN", score_count),
    )
    scoring_seconds = time.perf_counter() - score_started

    neural: list[TrajectoryScore] = []
    baseline_scores: list[TrajectoryScore] = []
    for seed in e02_config.formal_seeds:
        for regime in ("SEEN", "UNSEEN"):
            for index in range(12):
                identifier = f"probe-{seed}-{regime}-{index}"
                neural.append(
                    replace(
                        _score_template(0.9),
                        trajectory_id=identifier,
                        formal_seed=seed,
                        regime=cast(Any, regime),
                    )
                )
                baseline_scores.append(
                    replace(
                        _score_template(1.0),
                        trajectory_id=identifier,
                        formal_seed=seed,
                        regime=cast(Any, regime),
                    )
                )
    bootstrap_started = time.perf_counter()
    paired_stratified_bootstrap(tuple(neural), tuple(baseline_scores), e02_config.bootstrap)
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    return {
        "generation_trajectories_per_second": 1.0 / generation_seconds,
        "linear_windows_per_second": batch.x.shape[0] / linear_seconds,
        "scoring_windows_per_second": score_count / scoring_seconds,
        "bootstrap_seconds": bootstrap_seconds,
    }


def run_e02_server_probe(
    repository_root: Path,
    config_path: Path,
    runtime_root: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    root = repository_root.resolve()
    e02_config = load_e02_config(config_path.resolve())
    stage2_config = load_stage2_config(root / "configs/stage2/stage2_v1.yaml")
    runtime = runtime_root.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    inputs = _verified_probe_inputs(root)
    checkpoints = tuple(inputs["itransformer_checkpoints"])
    if len(checkpoints) != 3:
        raise RuntimeError("E02 probe requires exactly three frozen iTransformer checkpoints")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        first_wave = tuple(
            pool.submit(
                _probe_neural_checkpoint_worker,
                str(root),
                inputs["development_data_ref"],
                checkpoints[index]["ref"],
                index,
                gpu_id,
            )
            for index, gpu_id in ((0, 0), (1, 1))
        )
        first_observations = tuple(item.result() for item in first_wave)
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        third = pool.submit(
            _probe_neural_checkpoint_worker,
            str(root),
            inputs["development_data_ref"],
            checkpoints[2]["ref"],
            2,
            0,
        ).result()
    observations = (*first_observations, third)
    if any(
        observation.get("forecast_finite") is not True
        or observation.get("positive_scales") is not True
        or observation.get("checkpoint_hash_unchanged") is not True
        for observation in observations
    ):
        raise RuntimeError("E02 neural checkpoint probe failed")
    neural_rates: list[float] = []
    neural_startups: list[float] = []
    for observation in observations:
        rate = observation.get("windows_per_second")
        startup = observation.get("startup_seconds")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or isinstance(startup, bool)
            or not isinstance(startup, (int, float))
        ):
            raise RuntimeError("E02 neural checkpoint probe timing is invalid")
        rate_value = float(rate)
        startup_value = float(startup)
        if (
            not math.isfinite(rate_value)
            or not math.isfinite(startup_value)
            or rate_value <= 0
            or startup_value < 0
        ):
            raise RuntimeError("E02 neural checkpoint probe timing is invalid")
        neural_rates.append(rate_value)
        neural_startups.append(startup_value)
    cpu = _probe_cpu_path(root, inputs)
    windows_per_trajectory = (
        stage2_config.data.trajectory_length
        - stage2_config.data.history
        - stage2_config.data.horizon
        + 1
    )
    formal_trajectories = e02_config.gate.required_completed_trajectories
    estimated = estimate_e02_critical_path_seconds(
        formal_trajectory_count=formal_trajectories,
        windows_per_trajectory=windows_per_trajectory,
        generation_trajectories_per_second=cpu["generation_trajectories_per_second"],
        neural_windows_per_second=cast(
            tuple[float, float, float],
            tuple(neural_rates),
        ),
        neural_startup_seconds=cast(
            tuple[float, float, float],
            tuple(neural_startups),
        ),
        linear_windows_per_second=cpu["linear_windows_per_second"],
        scoring_windows_per_second=cpu["scoring_windows_per_second"],
        bootstrap_seconds=cpu["bootstrap_seconds"],
        score_parallel_tasks=4,
    )
    stage2_reset_time_gate(
        estimated_remaining_seconds=estimated,
        remaining_rental_hours=remaining_rental_hours,
        margin_hours=e02_config.runtime_profile.reset_margin_hours,
    )
    return {
        "probe_contract": "e02-v1-three-frozen-checkpoints-two-gpu-waves",
        "neural_observations": observations,
        "cpu_observation": cpu,
        "formal_trajectory_count": formal_trajectories,
        "formal_window_count": formal_trajectories * windows_per_trajectory,
        "estimated_remaining_seconds": estimated,
        "reset_margin_hours": e02_config.runtime_profile.reset_margin_hours,
        "eta_gate_passed": True,
        "formal_tasks_executed": 0,
    }
