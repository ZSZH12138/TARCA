from __future__ import annotations

import math
from dataclasses import dataclass

from tarca.e01.resources import (
    E01ServerInventory,
    ServerAdmissionError,
    server_admission_check,
)

_MAX_PROBE_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class E01V2CapacityCandidate:
    cpu_analysis_workers: int
    gpu_batch_size: int
    gpu_worker_count: int = 1
    service_cores: int = 2


@dataclass(frozen=True, slots=True)
class E01V2ProbeObservation:
    cpu_analysis_workers: int
    gpu_batch_size: int
    combined_throughput: float
    peak_ram_gib: float
    gpu_utilization_percent: float
    peak_gpu_vram_gib: float
    oom: bool = False
    swap_used_gib: float = 0.0
    throttled: bool = False

    def __post_init__(self) -> None:
        if self.cpu_analysis_workers <= 0 or self.gpu_batch_size <= 0:
            raise ValueError("E01-v2 probe concurrency and batch size must be positive")
        values = (
            self.combined_throughput,
            self.peak_ram_gib,
            self.gpu_utilization_percent,
            self.peak_gpu_vram_gib,
            self.swap_used_gib,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("E01-v2 probe observations must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class E01V2CapacityPlan:
    cpu_analysis_workers: int
    gpu_worker_count: int
    gpu_batch_size: int
    service_cores: int
    expected_combined_throughput: float
    peak_ram_gib: float
    peak_gpu_vram_gib: float


def initial_v2_probe_candidates(
    inventory: E01ServerInventory,
) -> tuple[E01V2CapacityCandidate, ...]:
    if inventory.physical_cpu_cores < 4 or len(inventory.gpu_names) != 1:
        raise ServerAdmissionError("E01-v2 requires CPU capacity and exactly one CUDA GPU")
    service_cores = 2
    maximum_workers = inventory.physical_cpu_cores - service_cores
    workers = tuple(value for value in (8, 10, 12) if value <= maximum_workers)
    if not workers:
        workers = (maximum_workers,)
    return tuple(
        E01V2CapacityCandidate(worker, batch, 1, service_cores)
        for worker in workers
        for batch in (2048, 4096, 8192)
    )


def choose_v2_capacity_plan(
    inventory: E01ServerInventory,
    observations: tuple[E01V2ProbeObservation, ...],
) -> E01V2CapacityPlan:
    if not observations:
        raise ServerAdmissionError("no E01-v2 capacity observations were recorded")
    maximum_workers = inventory.physical_cpu_cores - 2
    ram_ceiling = inventory.available_ram_gib * 0.85
    gpu_ceiling = inventory.gpu_vram_gib[0] * 0.90 if inventory.gpu_vram_gib else 0.0
    safe = tuple(
        item
        for item in observations
        if item.cpu_analysis_workers <= maximum_workers
        and item.peak_ram_gib <= ram_ceiling
        and item.peak_gpu_vram_gib <= gpu_ceiling
        and item.swap_used_gib == 0.0
        and not item.oom
        and not item.throttled
        and item.combined_throughput > 0.0
    )
    if not safe:
        raise ServerAdmissionError("no safe E01-v2 capacity observation remains")
    selected = max(
        safe,
        key=lambda item: (
            item.combined_throughput,
            item.cpu_analysis_workers,
            item.gpu_batch_size,
        ),
    )
    return E01V2CapacityPlan(
        cpu_analysis_workers=selected.cpu_analysis_workers,
        gpu_worker_count=1,
        gpu_batch_size=selected.gpu_batch_size,
        service_cores=2,
        expected_combined_throughput=selected.combined_throughput,
        peak_ram_gib=selected.peak_ram_gib,
        peak_gpu_vram_gib=selected.peak_gpu_vram_gib,
    )


def v2_server_admission_check(
    inventory: E01ServerInventory,
    *,
    estimated_runtime_hours: float,
    remaining_rental_hours: float,
    minimum_storage_gib: float,
    reset_margin_hours: float,
    probe_elapsed_seconds: float,
) -> None:
    if not math.isfinite(probe_elapsed_seconds) or probe_elapsed_seconds < 0.0:
        raise ValueError("probe elapsed time must be finite and non-negative")
    if probe_elapsed_seconds > _MAX_PROBE_SECONDS:
        raise ServerAdmissionError("E01-v2 five-minute capacity probe deadline was exceeded")
    server_admission_check(
        inventory,
        estimated_runtime_hours=estimated_runtime_hours,
        remaining_rental_hours=remaining_rental_hours,
        minimum_storage_gib=minimum_storage_gib,
        reset_margin_hours=reset_margin_hours,
    )
