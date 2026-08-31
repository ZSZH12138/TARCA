from __future__ import annotations

import math
from dataclasses import dataclass

from tarca.execution import (
    GpuPackingDecision,
    HostAdmissionPolicy,
    ResourceCapacity,
    decide_gpu_packing,
)
from tarca.execution.telemetry import GpuSample

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class Stage2ServerInventory:
    logical_cpu_count: int
    physical_cpu_count: int
    ram_bytes: int
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]
    free_storage_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.logical_cpu_count,
                self.physical_cpu_count,
                self.ram_bytes,
                self.free_storage_bytes,
            )
            <= 0
        ):
            raise ValueError("Stage 2 server inventory values must be positive")
        if len(self.gpu_names) != len(self.gpu_vram_bytes):
            raise ValueError("GPU names and memory inventories must align")


@dataclass(frozen=True, slots=True)
class Stage2ProbeObservation:
    cuda_available: bool
    cuda_device_count: int
    source_hashes_verified: bool
    checkpoint_roundtrip_passed: bool
    fp32_finite: bool
    amp_finite: bool


@dataclass(frozen=True, slots=True)
class Stage2CapacityPlan:
    work_cpu_cores: int
    scheduler_monitor_cores: int
    system_io_cores: int
    gpu_worker_count: int
    host_memory_ceiling_gib: int
    storage_floor_gib: int
    dataloader_workers_per_gpu_job: int
    policy: HostAdmissionPolicy


def stage2_server_admission_check(
    inventory: Stage2ServerInventory,
    probes: Stage2ProbeObservation,
) -> None:
    if inventory.physical_cpu_count < 28 or inventory.logical_cpu_count < 28:
        raise RuntimeError("Stage 2 requires at least 28 available CPU cores")
    if inventory.ram_bytes < 224 * _GIB:
        raise RuntimeError("Stage 2 requires at least 224 GiB RAM")
    if (
        len(inventory.gpu_names) != 2
        or len(inventory.gpu_vram_bytes) != 2
        or any("RTX 4090" not in name.upper() for name in inventory.gpu_names)
        or any(memory < 24 * _GIB for memory in inventory.gpu_vram_bytes)
    ):
        raise RuntimeError("Stage 2 requires exactly two RTX 4090 GPUs with 24 GiB each")
    if inventory.free_storage_bytes < 200 * _GIB:
        raise RuntimeError("Stage 2 requires at least 200 GiB free local storage")
    if not probes.cuda_available or probes.cuda_device_count != 2:
        raise RuntimeError("Stage 2 CUDA probe must observe exactly two devices")
    if not probes.source_hashes_verified:
        raise RuntimeError("Stage 2 source hash verification did not pass")
    if not probes.checkpoint_roundtrip_passed:
        raise RuntimeError("Stage 2 checkpoint round-trip did not pass")
    if not probes.fp32_finite:
        raise RuntimeError("Stage 2 FP32 probe must be finite")


def choose_stage2_capacity_plan(
    inventory: Stage2ServerInventory,
    probes: Stage2ProbeObservation,
) -> Stage2CapacityPlan:
    stage2_server_admission_check(inventory, probes)
    policy = HostAdmissionPolicy(
        scheduler_monitor_cores=1,
        system_io_reserved_cores=3,
        maximum_data_cores=24,
        maximum_host_memory_bytes=200 * _GIB,
        minimum_local_storage_free_bytes=200 * _GIB,
        initial_loader_workers_per_gpu_job=3,
    )
    return Stage2CapacityPlan(
        work_cpu_cores=24,
        scheduler_monitor_cores=1,
        system_io_cores=3,
        gpu_worker_count=2,
        host_memory_ceiling_gib=200,
        storage_floor_gib=200,
        dataloader_workers_per_gpu_job=3,
        policy=policy,
    )


def stage2_resource_capacity(inventory: Stage2ServerInventory) -> ResourceCapacity:
    return ResourceCapacity(
        logical_cpu_count=inventory.logical_cpu_count,
        physical_cpu_count=inventory.physical_cpu_count,
        available_memory_bytes=inventory.ram_bytes,
        gpu_memory_bytes=inventory.gpu_vram_bytes,
        local_storage_available=True,
        local_storage_free_bytes=inventory.free_storage_bytes,
    )


@dataclass(frozen=True, slots=True)
class InferenceBundleController:
    """Controls 1-3 inference bundles inside one exclusive GPU worker."""

    def observe(
        self,
        sample: GpuSample,
        *,
        stable_seconds: float,
        current_jobs: int,
        oom: bool = False,
        throttled: bool = False,
        throughput_loss: bool = False,
        data_wait: bool = False,
    ) -> GpuPackingDecision:
        return decide_gpu_packing(
            sample,
            stable_seconds,
            current_jobs,
            oom=oom,
            throttled=throttled,
            throughput_loss=throughput_loss,
            data_wait=data_wait,
        )


def stage2_reset_time_gate(
    *,
    estimated_remaining_seconds: float,
    remaining_rental_hours: float,
    margin_hours: float = 1.0,
) -> None:
    values = (estimated_remaining_seconds, remaining_rental_hours, margin_hours)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Stage 2 timing values must be finite")
    if estimated_remaining_seconds < 0 or remaining_rental_hours <= 0 or margin_hours < 0:
        raise ValueError("Stage 2 timing values are outside their valid range")
    boundary_seconds = remaining_rental_hours * 3600
    if estimated_remaining_seconds + margin_hours * 3600 >= boundary_seconds:
        raise RuntimeError("Stage 2 ETA plus safety margin reaches the rental reset boundary")
