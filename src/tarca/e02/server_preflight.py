from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import torch

from tarca.contracts import canonical_json_bytes
from tarca.e02.server_handoff import (
    load_e02_server_handoff,
    load_server_bundle_verification_receipt,
    load_stage2_restore_receipt,
)
from tarca.e02.server_probe import run_e02_server_probe
from tarca.stage2.freeze import verify_frozen_stage2_suite
from tarca.stage2.server_preflight import _verify_sources

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class E02HardwareInventory:
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    physical_cpu_count: int
    ram_bytes: int
    free_storage_bytes: int
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.physical_cpu_count, self.ram_bytes, self.free_storage_bytes) <= 0:
            raise ValueError("E02 hardware inventory values must be positive")
        if len(self.gpu_names) != len(self.gpu_vram_bytes):
            raise ValueError("E02 GPU names and memory inventories must align")


def e02_server_hardware_check(inventory: E02HardwareInventory) -> None:
    if inventory.python_version != "3.10":
        raise RuntimeError("E02 requires Python 3.10")
    if inventory.torch_version != "2.2.2":
        raise RuntimeError("E02 requires PyTorch 2.2.2")
    if inventory.torch_cuda_version != "12.1" or not inventory.cuda_available:
        raise RuntimeError("E02 requires CUDA 12.1 with CUDA available")
    if inventory.physical_cpu_count < 28:
        raise RuntimeError("E02 requires at least 28 physical CPU cores")
    if inventory.ram_bytes < 224 * _GIB:
        raise RuntimeError("E02 requires at least 224 GiB RAM")
    if inventory.free_storage_bytes < 200 * _GIB:
        raise RuntimeError("E02 requires at least 200 GiB free local storage")
    if len(inventory.gpu_names) != 2 or len(inventory.gpu_vram_bytes) != 2:
        raise RuntimeError("E02 requires exactly two RTX 4090 GPUs")
    if any("4090" not in name.upper() for name in inventory.gpu_names):
        raise RuntimeError("E02 requires exactly two RTX 4090 GPUs")
    if any(memory < 23 * _GIB for memory in inventory.gpu_vram_bytes):
        raise RuntimeError("each E02 GPU requires at least 23 GiB driver-reported VRAM")


def _collect_hardware(artifact_root: Path) -> E02HardwareInventory:
    return E02HardwareInventory(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        torch_version=torch.__version__.split("+")[0],
        torch_cuda_version=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        physical_cpu_count=psutil.cpu_count(logical=False) or 0,
        ram_bytes=psutil.virtual_memory().total,
        free_storage_bytes=shutil.disk_usage(artifact_root.resolve()).free,
        gpu_names=tuple(
            torch.cuda.get_device_properties(index).name
            for index in range(torch.cuda.device_count())
        ),
        gpu_vram_bytes=tuple(
            int(torch.cuda.get_device_properties(index).total_memory)
            for index in range(torch.cuda.device_count())
        ),
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".e02-preflight-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_e02_server_preflight(
    repository_root: Path,
    e02_config_path: Path,
    stage2_config_path: Path,
    artifact_root: Path,
    handoff_path: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    root = repository_root.resolve()
    artifacts = artifact_root.resolve()
    runtime = artifacts / "runtime"
    handoff = load_e02_server_handoff(handoff_path.resolve())
    restore_receipt = load_stage2_restore_receipt(
        runtime / "stage2_restore_receipt.json", handoff
    )
    bundle_receipt = load_server_bundle_verification_receipt(
        runtime / "server_bundle_verification_receipt.json"
    )
    hardware = _collect_hardware(artifacts)
    e02_server_hardware_check(hardware)
    if remaining_rental_hours <= 1.0:
        raise RuntimeError("E02 remaining rental time must exceed the one-hour margin")
    freeze = verify_frozen_stage2_suite(root / "artifacts/stage2")
    if freeze.receipt_sha256 != handoff.stage2_freeze_receipt_sha256:
        raise RuntimeError("E02 frozen Stage 2 identity does not match the handoff")
    _verify_sources(root, stage2_config_path.resolve())
    probe = run_e02_server_probe(
        root,
        e02_config_path.resolve(),
        runtime,
        remaining_rental_hours=remaining_rental_hours,
    )
    if any(
        (
            probe.get("probe_contract")
            != "e02-v1-three-frozen-checkpoints-two-gpu-waves",
            probe.get("eta_gate_passed") is not True,
            probe.get("formal_tasks_executed") != 0,
        )
    ):
        raise RuntimeError("E02 server probe evidence is incomplete")
    observed_at = datetime.now(UTC)
    evidence: dict[str, Any] = {
        "schema_version": "tarca-e02-server-preflight-evidence-v1",
        "status": "PREFLIGHT_PASS",
        "remaining_rental_hours": remaining_rental_hours,
        "observed_at_utc": observed_at.isoformat(),
        "rental_reset_at_utc": (
            observed_at + timedelta(hours=remaining_rental_hours)
        ).isoformat(),
        "complete_archive_sha256": handoff.complete_archive_sha256,
        "stage2_restore_receipt_sha256": restore_receipt["receipt_sha256"],
        "server_bundle_sha256": bundle_receipt["server_bundle_sha256"],
        "server_bundle_verification_receipt_sha256": bundle_receipt["receipt_sha256"],
        "stage2_freeze_receipt_sha256": freeze.receipt_sha256,
        "e02_scientific_config_sha256": handoff.e02_scientific_config_sha256,
        "source_hashes_verified": True,
        "hardware": asdict(hardware),
        "gpu_count": 2,
        "work_cpu_cores": 24,
        "scheduler_monitor_cores": 1,
        "system_io_cores": 3,
        "host_memory_ceiling_gib": 200,
        "storage_floor_gib": 200,
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
        **probe,
    }
    normalized = json.loads(canonical_json_bytes(evidence))
    if not isinstance(normalized, dict):  # pragma: no cover - canonical object above
        raise RuntimeError("E02 preflight evidence serialization failed")
    evidence = normalized
    _atomic_json(runtime / "bootstrap_evidence.json", evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TARCA E02 v1 fresh-server preflight")
    parser.add_argument("--remaining-rental-hours", type=float, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--e02-config", type=Path, default=Path("configs/e02/e02_v1.yaml"))
    parser.add_argument(
        "--stage2-config", type=Path, default=Path("configs/stage2/stage2_v1.yaml")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/e02"))
    parser.add_argument(
        "--handoff", type=Path, default=Path("configs/e02/e02_server_handoff_v1.json")
    )
    arguments = parser.parse_args(argv)
    result = run_e02_server_preflight(
        arguments.repository_root,
        arguments.e02_config,
        arguments.stage2_config,
        arguments.artifact_root,
        arguments.handoff,
        remaining_rental_hours=arguments.remaining_rental_hours,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
