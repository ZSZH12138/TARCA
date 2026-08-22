from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import psutil  # type: ignore[import-untyped]
import torch


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    cpu_model: str
    logical_cpu_count: int
    physical_cpu_count: int
    total_memory_bytes: int
    available_memory_bytes: int
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]
    platform: str


@dataclass(frozen=True, slots=True)
class HardwareGateDecision:
    feasible: bool
    estimated_hours: float
    projected_peak_memory_bytes: int
    available_memory_bytes: int
    failed_checks: tuple[str, ...]
    maximum_hours: float
    memory_safety_fraction: float


def inventory_hardware() -> HardwareInventory:
    virtual_memory = psutil.virtual_memory()
    gpu_names: list[str] = []
    gpu_vram: list[int] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_names.append(properties.name)
            gpu_vram.append(properties.total_memory)
    cpu_model = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"
    return HardwareInventory(
        cpu_model=cpu_model,
        logical_cpu_count=psutil.cpu_count(logical=True) or 1,
        physical_cpu_count=psutil.cpu_count(logical=False) or 1,
        total_memory_bytes=virtual_memory.total,
        available_memory_bytes=virtual_memory.available,
        gpu_names=tuple(gpu_names),
        gpu_vram_bytes=tuple(gpu_vram),
        platform=platform.platform(),
    )


def estimate_full_run(
    *,
    probe_seconds: float,
    probe_work_units: int,
    full_work_units: int,
    projected_peak_memory_bytes: int,
    available_memory_bytes: int,
    maximum_hours: float = 120.0,
    memory_safety_fraction: float = 0.8,
) -> HardwareGateDecision:
    if probe_seconds <= 0 or probe_work_units <= 0 or full_work_units <= 0:
        raise ValueError("hardware estimate work and time inputs must be positive")
    if projected_peak_memory_bytes <= 0 or available_memory_bytes <= 0:
        raise ValueError("hardware estimate memory inputs must be positive")
    if maximum_hours <= 0 or not 0 < memory_safety_fraction < 1:
        raise ValueError("hardware gate limits are invalid")
    estimated_hours = probe_seconds * full_work_units / probe_work_units / 3600.0
    failed: list[str] = []
    if estimated_hours > maximum_hours:
        failed.append("runtime")
    if projected_peak_memory_bytes > available_memory_bytes * memory_safety_fraction:
        failed.append("memory")
    return HardwareGateDecision(
        feasible=not failed,
        estimated_hours=estimated_hours,
        projected_peak_memory_bytes=projected_peak_memory_bytes,
        available_memory_bytes=available_memory_bytes,
        failed_checks=tuple(failed),
        maximum_hours=maximum_hours,
        memory_safety_fraction=memory_safety_fraction,
    )

