from __future__ import annotations

import math
from dataclasses import dataclass


class ServerAdmissionError(RuntimeError):
    """Raised when E01-v2 cannot run without changing the frozen science."""


_MINIMUM_AVAILABLE_RAM_GIB = 96.0


@dataclass(frozen=True, slots=True)
class E01ServerInventory:
    physical_cpu_cores: int
    logical_cpu_count: int
    available_ram_gib: float
    gpu_names: tuple[str, ...]
    gpu_vram_gib: tuple[float, ...]
    free_storage_gib: float

    def __post_init__(self) -> None:
        if min(self.physical_cpu_cores, self.logical_cpu_count) <= 0:
            raise ValueError("CPU inventory must be positive")
        if self.physical_cpu_cores > self.logical_cpu_count:
            raise ValueError("physical CPU cores cannot exceed logical CPUs")
        numeric = (self.available_ram_gib, self.free_storage_gib, *self.gpu_vram_gib)
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("memory and storage inventory must be finite and positive")
        if len(self.gpu_names) != len(self.gpu_vram_gib) or any(
            not name.strip() for name in self.gpu_names
        ):
            raise ValueError("GPU names and VRAM entries must match")


def worker_thread_environment() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
    }


def server_admission_check(
    inventory: E01ServerInventory,
    *,
    estimated_runtime_hours: float,
    remaining_rental_hours: float,
    minimum_storage_gib: float,
    reset_margin_hours: float,
) -> None:
    times = (estimated_runtime_hours, remaining_rental_hours, reset_margin_hours)
    if any(not math.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError("runtime and rental durations must be finite and positive")
    if not math.isfinite(minimum_storage_gib) or minimum_storage_gib <= 0.0:
        raise ValueError("minimum storage must be finite and positive")
    if inventory.free_storage_gib < minimum_storage_gib:
        raise ServerAdmissionError("free storage is below the E01 recovery requirement")
    if inventory.available_ram_gib < _MINIMUM_AVAILABLE_RAM_GIB:
        raise ServerAdmissionError(
            "available RAM is below the 96 GiB E01 execution safety envelope"
        )
    if len(inventory.gpu_names) != 1 or "4090" not in inventory.gpu_names[0].upper():
        raise ServerAdmissionError("the authorized single RTX 4090 was not detected")
    if inventory.gpu_vram_gib[0] < 23.0:
        raise ServerAdmissionError("usable GPU VRAM is below the RTX 4090 envelope")
    if inventory.physical_cpu_cores < 14:
        raise ServerAdmissionError("physical CPU capacity is below the authorized envelope")
    if remaining_rental_hours <= estimated_runtime_hours + reset_margin_hours:
        raise ServerAdmissionError("remaining rental time does not include the reset margin")
