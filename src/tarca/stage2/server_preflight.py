from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

from tarca.stage2.recovery import load_stage2_recovery_input_receipt
from tarca.stage2.recovery_probe import run_stage2_recovery_probe
from tarca.stage2.server_probe import run_stage2_server_probe


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_hardware(artifact_root: Path, remaining_hours: float) -> None:
    _require(sys.version_info[:2] == (3, 10), "Python 3.10 is required")
    _require(torch.__version__.split("+")[0] == "2.2.2", "Torch 2.2.2 is required")
    _require(torch.version.cuda == "12.1", "Torch CUDA 12.1 is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 2, "exactly two GPUs are required")
    _require((psutil.cpu_count(logical=False) or 0) >= 28, "28 physical CPUs are required")
    _require(psutil.virtual_memory().total >= 224 * 1024**3, "224 GiB RAM is required")
    _require(shutil.disk_usage(artifact_root).free >= 200 * 1024**3, "200 GiB disk is required")
    _require(remaining_hours > 1.0, "remaining rental time is insufficient")
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        _require("4090" in properties.name, "each GPU must be an RTX 4090")
        _require(properties.total_memory >= 23 * 1024**3, "each GPU needs 23 GiB VRAM")


def _verify_sources(repository_root: Path, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for source in config["sources"]:
        source_root = (
            repository_root / "third_party/stage2" / source["source_id"] / source["commit"]
        )
        for asset in source["assets"]:
            path = source_root / asset["relative_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            _require(digest == asset["sha256"], "official source hash mismatch")


def _exercise_cuda_and_checkpoint(runtime: Path) -> tuple[bool, bool]:
    values = torch.linspace(-1, 1, 4096, device="cuda:0")
    fp32 = (values.float() * values.float()).sum()
    with torch.autocast("cuda", dtype=torch.float16):
        amp = (values * values).sum()
    _require(bool(torch.isfinite(fp32)), "FP32 probe was not finite")
    _require(bool(torch.isfinite(amp)), "AMP probe was not finite")
    with tempfile.NamedTemporaryFile(dir=runtime, delete=False) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"probe": values[:16].cpu()}, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu")
        _require(torch.equal(loaded["probe"], values[:16].cpu()), "checkpoint roundtrip failed")
    finally:
        checkpoint.unlink(missing_ok=True)
    return bool(torch.isfinite(fp32)), bool(torch.isfinite(amp))


def run_server_preflight(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    root = repository_root.resolve()
    config = config_path.resolve()
    artifacts = artifact_root.resolve()
    runtime = artifacts / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    recovery_input = runtime / "recovery_input_receipt.json"
    recovery_mode = recovery_input.is_file()
    if recovery_mode:
        load_stage2_recovery_input_receipt(recovery_input)
    _verify_hardware(artifacts, remaining_rental_hours)
    _verify_sources(root, config)
    fp32_finite, amp_finite = _exercise_cuda_and_checkpoint(runtime)
    if recovery_mode:
        throughput = run_stage2_recovery_probe(
            root,
            config,
            runtime,
            remaining_rental_hours=remaining_rental_hours,
        )
    else:
        throughput = run_stage2_server_probe(
            root,
            config,
            runtime,
            remaining_rental_hours=remaining_rental_hours,
        )
    evidence: dict[str, Any] = {
        "status": "PREFLIGHT_PASS",
        "remaining_rental_hours": remaining_rental_hours,
        "gpu_count": 2,
        "source_hashes_verified": True,
        "checkpoint_roundtrip_passed": True,
        "fp32_finite": fp32_finite,
        "amp_finite": amp_finite,
        "formal_tasks_executed": 0,
        **throughput,
    }
    (runtime / "bootstrap_evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TARCA Stage 2 spawn-safe server preflight")
    parser.add_argument("--remaining-rental-hours", type=float, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/stage2/stage2_v1.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/stage2"))
    arguments = parser.parse_args(argv)
    run_server_preflight(
        arguments.repository_root,
        arguments.config,
        arguments.artifact_root,
        remaining_rental_hours=arguments.remaining_rental_hours,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
