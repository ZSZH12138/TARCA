from __future__ import annotations

import shutil
from pathlib import Path

from tarca.e01.resources import E01ServerInventory
from tarca.stage1b.hardware import inventory_hardware

_GIB = 1024**3


def inspect_e01_server(artifact_root: Path) -> E01ServerInventory:
    """Read the host inventory used by the bounded E01-v2 capacity probe."""

    hardware = inventory_hardware()
    disk = shutil.disk_usage(artifact_root.resolve())
    return E01ServerInventory(
        physical_cpu_cores=hardware.physical_cpu_count,
        logical_cpu_count=hardware.logical_cpu_count,
        available_ram_gib=hardware.available_memory_bytes / _GIB,
        gpu_names=hardware.gpu_names,
        gpu_vram_gib=tuple(value / _GIB for value in hardware.gpu_vram_bytes),
        free_storage_gib=disk.free / _GIB,
    )
