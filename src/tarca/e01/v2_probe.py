from __future__ import annotations

import concurrent.futures
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from tarca.e01.probe import inspect_e01_server
from tarca.e01.resources import E01ServerInventory, ServerAdmissionError
from tarca.e01.v2_config import E01V2Config
from tarca.e01.v2_resources import (
    E01V2CapacityPlan,
    E01V2ProbeObservation,
    choose_v2_capacity_plan,
    initial_v2_probe_candidates,
)

_GIB = 1024**3
_NONFORMAL_PROBE_SEED = 2_147_000_123
_MAX_PROBE_SECONDS = 300.0


def _cpu_analysis_probe(seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8192, 12)).astype(np.float64)
    operations = 0
    for _ in range(12):
        means = values.mean(axis=0)
        centered = values - means
        values = centered * 0.999 + means
        operations += values.size * 3
    return operations, psutil.Process().memory_info().rss


def _gpu_generation_probe(batch_size: int) -> int:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_NONFORMAL_PROBE_SEED)
    host = torch.randn(batch_size, 12, generator=generator, dtype=torch.float64)
    device = host.to(device="cuda:0")
    for _ in range(5):
        device = device.mul(0.75).add(0.25)
    checksum = float(device.mean().item())
    if not math.isfinite(checksum):
        raise RuntimeError("E01-v2 GPU probe produced a non-finite checksum")
    return batch_size * 12 * 5


def run_v2_capacity_observations(
    inventory: E01ServerInventory,
    *,
    deadline_seconds: float = _MAX_PROBE_SECONDS,
) -> tuple[tuple[E01V2ProbeObservation, ...], float]:
    if not torch.cuda.is_available():
        raise RuntimeError("E01-v2 capacity probe requires CUDA")
    if not 0.0 < deadline_seconds <= _MAX_PROBE_SECONDS:
        raise ValueError("E01-v2 probe deadline must be inside (0, 300]")
    probe_started = time.monotonic()
    observations: list[E01V2ProbeObservation] = []
    for candidate in initial_v2_probe_candidates(inventory):
        if time.monotonic() - probe_started >= deadline_seconds:
            raise ServerAdmissionError("E01-v2 five-minute probe deadline was exceeded")
        before_swap = psutil.swap_memory().used
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        oom = False
        cpu_operations = 0
        child_peak = 0
        gpu_operations = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=candidate.cpu_analysis_workers
        ) as pool:
            futures = tuple(
                pool.submit(_cpu_analysis_probe, _NONFORMAL_PROBE_SEED + worker)
                for worker in range(candidate.cpu_analysis_workers)
            )
            try:
                gpu_operations = _gpu_generation_probe(candidate.gpu_batch_size)
            except torch.cuda.OutOfMemoryError:
                oom = True
                torch.cuda.empty_cache()
            for future in futures:
                operations, rss = future.result()
                cpu_operations += operations
                child_peak += rss
        elapsed = max(time.perf_counter() - started, 1e-9)
        host_used = psutil.virtual_memory().total - psutil.virtual_memory().available
        utilization_reader = getattr(torch.cuda, "utilization", None)
        try:
            utilization = float(utilization_reader(0)) if utilization_reader else 0.0
        except (RuntimeError, ValueError):
            utilization = 0.0
        observations.append(
            E01V2ProbeObservation(
                cpu_analysis_workers=candidate.cpu_analysis_workers,
                gpu_batch_size=candidate.gpu_batch_size,
                combined_throughput=(cpu_operations + gpu_operations) / elapsed,
                peak_ram_gib=max(host_used, child_peak) / _GIB,
                gpu_utilization_percent=utilization,
                peak_gpu_vram_gib=torch.cuda.max_memory_allocated(0) / _GIB,
                oom=oom,
                swap_used_gib=max(0, psutil.swap_memory().used - before_swap) / _GIB,
                throttled=False,
            )
        )
    total_elapsed = time.monotonic() - probe_started
    if total_elapsed > deadline_seconds:
        raise ServerAdmissionError("E01-v2 five-minute probe deadline was exceeded")
    return tuple(observations), total_elapsed


def estimate_e01_v2_runtime_hours(
    config: E01V2Config,
    capacity: E01V2CapacityPlan,
) -> float:
    work_units = (
        len(config.formal_seeds)
        * len(config.conditions)
        * config.sample_sizes[-1]
        * config.horizons[-1]
        * 8
    )
    seconds = work_units / max(capacity.expected_combined_throughput, 1.0) * 1.5
    return max(seconds / 3600.0, 1.0 / 3600.0)


def run_bounded_e01_v2_probe(
    artifact_root: Path,
    config: E01V2Config,
) -> tuple[
    E01ServerInventory,
    tuple[E01V2ProbeObservation, ...],
    float,
    float,
]:
    inventory = inspect_e01_server(artifact_root)
    observations, elapsed = run_v2_capacity_observations(inventory)
    capacity = choose_v2_capacity_plan(inventory, observations)
    return inventory, observations, estimate_e01_v2_runtime_hours(config, capacity), elapsed
