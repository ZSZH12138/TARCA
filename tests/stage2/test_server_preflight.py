from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tarca.contracts import canonical_json_hash
from tarca.stage2.recovery import Stage2RecoveryRejected
from tarca.stage2.server_preflight import run_server_preflight

ROOT = Path(__file__).resolve().parents[2]


def test_server_preflight_has_an_importable_spawn_safe_module_entrypoint() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "deploy/stage2/py310"), str(ROOT / "src"))),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "tarca.stage2.server_preflight", "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "remaining-rental-hours" in completed.stdout


def test_server_preflight_routes_restored_run_to_two_gpu_recovery_probe(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    runtime = artifact_root / "runtime"
    runtime.mkdir(parents=True)
    input_payload = {
        "schema_version": "tarca-stage2-recovery-input-v1",
        "status": "RESTORED",
        "source_archive_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "source_database_sha256": "c" * 64,
        "server_bundle_sha256": "d" * 64,
    }
    recovery_input = {
        **input_payload,
        "receipt_sha256": canonical_json_hash(input_payload),
    }
    (runtime / "recovery_input_receipt.json").write_text(
        json.dumps(recovery_input) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "tarca.stage2.server_preflight._verify_hardware", lambda *args: None
    )
    monkeypatch.setattr(
        "tarca.stage2.server_preflight._verify_sources", lambda *args: None
    )
    monkeypatch.setattr(
        "tarca.stage2.server_preflight._exercise_cuda_and_checkpoint",
        lambda *args: (True, True),
    )
    monkeypatch.setattr(
        "tarca.stage2.server_preflight.run_stage2_server_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong probe")),
    )
    monkeypatch.setattr(
        "tarca.stage2.server_preflight.run_stage2_recovery_probe",
        lambda *args, **kwargs: {
            "probe_contract": "stage2-v1-recovery-two-complete-checkpoints-concurrent",
            "estimated_remaining_seconds": 7200.0,
            "reset_margin_hours": 1,
            "eta_gate_passed": True,
            "recovery_mode": "DEVICE_MISMATCH_V1",
            "complete_checkpoint_fast_path_passed": True,
            "zero_optimizer_steps": True,
            "checkpoint_hashes_unchanged": True,
        },
    )

    result = run_server_preflight(
        tmp_path,
        ROOT / "configs/stage2/stage2_v1.yaml",
        artifact_root,
        remaining_rental_hours=24,
    )

    assert result["recovery_mode"] == "DEVICE_MISMATCH_V1"
    assert result["zero_optimizer_steps"] is True


def test_server_preflight_rejects_tampered_recovery_input_before_hardware_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    runtime = artifact_root / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "recovery_input_receipt.json").write_text("{}\n", encoding="utf-8")
    hardware_called = False

    def hardware(*args: object) -> None:
        nonlocal hardware_called
        del args
        hardware_called = True

    monkeypatch.setattr("tarca.stage2.server_preflight._verify_hardware", hardware)

    with pytest.raises(Stage2RecoveryRejected, match="recovery input receipt"):
        run_server_preflight(
            tmp_path,
            ROOT / "configs/stage2/stage2_v1.yaml",
            artifact_root,
            remaining_rental_hours=24,
        )

    assert hardware_called is False
