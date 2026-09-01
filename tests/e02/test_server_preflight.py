from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.contracts import canonical_json_bytes, canonical_json_hash
from tarca.e02.server_preflight import (
    E02HardwareInventory,
    e02_server_hardware_check,
    run_e02_server_preflight,
)

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def _hardware(**changes: object) -> E02HardwareInventory:
    values: dict[str, object] = {
        "python_version": "3.10",
        "torch_version": "2.2.2",
        "torch_cuda_version": "12.1",
        "cuda_available": True,
        "physical_cpu_count": 28,
        "ram_bytes": 224 * GIB,
        "free_storage_bytes": 300 * GIB,
        "gpu_names": ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"),
        "gpu_vram_bytes": (23 * GIB, 23 * GIB),
    }
    values.update(changes)
    return E02HardwareInventory(**values)  # type: ignore[arg-type]


def _restore_receipt(artifact_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "tarca-e02-stage2-restore-v1",
        "status": "RESTORED",
        "complete_archive_filename": "tarca-stage2-v1-complete-20260901T011423Z.tar.gz",
        "complete_archive_sha256": (
            "7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a"
        ),
        "stage2_freeze_receipt_sha256": (
            "37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166"
        ),
        "e02_scientific_config_sha256": (
            "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c"
        ),
        "restored_file_count": 70,
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }
    receipt = {**payload, "receipt_sha256": canonical_json_hash(payload)}
    path = artifact_root / "runtime/stage2_restore_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def _bundle_receipt(artifact_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "tarca-e02-server-bundle-verification-v1",
        "status": "VERIFIED",
        "server_bundle_sha256": "b" * 64,
        "verified_file_count": 100,
        "formal_tasks_executed": 0,
    }
    receipt = {**payload, "receipt_sha256": canonical_json_hash(payload)}
    path = artifact_root / "runtime/server_bundle_verification_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"python_version": "3.11"}, "Python 3.10"),
        ({"torch_version": "2.2.1"}, "PyTorch 2.2.2"),
        ({"torch_cuda_version": "12.0"}, "CUDA 12.1"),
        ({"cuda_available": False}, "CUDA 12.1"),
        ({"physical_cpu_count": 27}, "28 physical"),
        ({"ram_bytes": 223 * GIB}, "224 GiB"),
        ({"free_storage_bytes": 199 * GIB}, "200 GiB"),
        ({"gpu_names": ("RTX 4090",), "gpu_vram_bytes": (23 * GIB,)}, "exactly two"),
        (
            {"gpu_names": ("RTX 4080", "RTX 4090")},
            "exactly two RTX 4090",
        ),
        ({"gpu_vram_bytes": (22 * GIB, 23 * GIB)}, "23 GiB"),
    ),
)
def test_e02_hardware_gate_fails_closed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        e02_server_hardware_check(_hardware(**changes))


def test_e02_server_preflight_binds_restore_hardware_probe_and_zero_formal_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/e02"
    restore = _restore_receipt(artifact_root)
    bundle = _bundle_receipt(artifact_root)
    monkeypatch.setattr(
        "tarca.e02.server_preflight._collect_hardware", lambda root: _hardware()
    )
    monkeypatch.setattr(
        "tarca.e02.server_preflight._verify_sources", lambda *args: None
    )
    monkeypatch.setattr(
        "tarca.e02.server_preflight.verify_frozen_stage2_suite",
        lambda root: SimpleNamespace(receipt_sha256=restore["stage2_freeze_receipt_sha256"]),
    )
    monkeypatch.setattr(
        "tarca.e02.server_preflight.run_e02_server_probe",
        lambda *args, **kwargs: {
            "probe_contract": "e02-v1-three-frozen-checkpoints-two-gpu-waves",
            "estimated_remaining_seconds": 7200.0,
            "reset_margin_hours": 1.0,
            "eta_gate_passed": True,
            "formal_tasks_executed": 0,
            "neural_observations": (),
        },
    )

    evidence = run_e02_server_preflight(
        tmp_path,
        ROOT / "configs/e02/e02_v1.yaml",
        ROOT / "configs/stage2/stage2_v1.yaml",
        artifact_root,
        ROOT / "configs/e02/e02_server_handoff_v1.json",
        remaining_rental_hours=24.0,
    )

    assert evidence["status"] == "PREFLIGHT_PASS"
    assert evidence["stage2_restore_receipt_sha256"] == restore["receipt_sha256"]
    assert evidence["server_bundle_sha256"] == bundle["server_bundle_sha256"]
    assert evidence["server_bundle_verification_receipt_sha256"] == bundle["receipt_sha256"]
    assert evidence["gpu_count"] == 2
    assert evidence["work_cpu_cores"] == 24
    assert evidence["host_memory_ceiling_gib"] == 200
    assert evidence["formal_tasks_executed"] == 0
    assert evidence["scientific_results_visible"] is False
    stored = json.loads(
        (artifact_root / "runtime/bootstrap_evidence.json").read_text(encoding="utf-8")
    )
    assert stored == evidence


def test_e02_server_preflight_requires_restore_receipt_before_hardware_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def hardware(root: Path) -> E02HardwareInventory:
        nonlocal called
        called = True
        return _hardware()

    monkeypatch.setattr("tarca.e02.server_preflight._collect_hardware", hardware)

    with pytest.raises(RuntimeError, match="restore receipt"):
        run_e02_server_preflight(
            tmp_path,
            ROOT / "configs/e02/e02_v1.yaml",
            ROOT / "configs/stage2/stage2_v1.yaml",
            tmp_path / "artifacts/e02",
            ROOT / "configs/e02/e02_server_handoff_v1.json",
            remaining_rental_hours=24.0,
        )

    assert called is False
