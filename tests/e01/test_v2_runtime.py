from __future__ import annotations

import json
from pathlib import Path

import pytest

from tarca.e01.resources import E01ServerInventory
from tarca.e01.v2_resources import E01V2ProbeObservation
from tarca.e01.v2_runtime import (
    E01_V2_FORMAL_RUN_ACKNOWLEDGEMENT,
    E01V2RuntimeAuthorizationError,
    dispatch_e01_v2_runtime_command,
    dry_run_e01_v2,
    launch_e01_v2,
    preflight_e01_v2,
    prepare_e01_v2,
    resume_e01_v2,
    status_e01_v2,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e01/e01_v2.yaml"


def _inventory() -> E01ServerInventory:
    return E01ServerInventory(
        physical_cpu_cores=14,
        logical_cpu_count=14,
        available_ram_gib=112.0,
        gpu_names=("NVIDIA GeForce RTX 4090",),
        gpu_vram_gib=(24.0,),
        free_storage_gib=350.0,
    )


def _observations() -> tuple[E01V2ProbeObservation, ...]:
    return (
        E01V2ProbeObservation(8, 2048, 100.0, 24.0, 60.0, 2.0),
        E01V2ProbeObservation(10, 4096, 180.0, 44.0, 80.0, 8.0),
        E01V2ProbeObservation(12, 8192, 250.0, 80.0, 95.0, 18.0),
    )


def _preflight(artifact_root: Path):
    return preflight_e01_v2(
        ROOT,
        CONFIG,
        artifact_root,
        inventory=_inventory(),
        observations=_observations(),
        estimated_runtime_hours=0.5,
        remaining_rental_hours=4.0,
        probe_elapsed_seconds=40.0,
        runtime_identity={
            "python": "3.10",
            "torch": "2.2.2",
            "cuda": "12.1",
            "cudnn_major": 8,
        },
    )


def test_v2_prepare_freezes_101_tasks_without_formal_execution(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    receipt = prepare_e01_v2(ROOT, CONFIG, artifact_root)

    assert receipt["status"] == "PREPARED"
    assert receipt["formal_tasks_executed"] == 0
    assert receipt["graph"] == {
        **receipt["graph"],
        "total_tasks": 101,
        "gpu_generation_tasks": 50,
        "cpu_analysis_tasks": 50,
        "aggregation_tasks": 1,
    }
    assert receipt["e01_b_carry_forward"]["status"] == "PASS"
    assert (artifact_root / "runtime/prepared_receipt_v2.json").is_file()
    assert not (artifact_root / "runtime/execution.sqlite3").exists()


def test_v2_dry_run_is_read_only_and_declares_nine_capacity_candidates(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    prepare_e01_v2(ROOT, CONFIG, artifact_root)

    result = dry_run_e01_v2(ROOT, CONFIG, artifact_root)

    assert result["status"] == "DRY_RUN_OK"
    assert result["initial_capacity_candidates"] == 9
    assert result["formal_tasks_executed"] == 0
    assert result["scientific_results_visible"] is False
    assert not (artifact_root / "runtime/execution.sqlite3").exists()


def test_v2_preflight_binds_config_graph_carry_forward_and_capacity(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    prepared = prepare_e01_v2(ROOT, CONFIG, artifact_root)

    receipt = _preflight(artifact_root)

    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["prepared_receipt_sha256"] == prepared["receipt_sha256"]
    assert (
        receipt["e01_b_carry_forward_receipt_sha256"]
        == prepared["e01_b_carry_forward"]["receipt_sha256"]
    )
    assert receipt["capacity_plan"]["cpu_analysis_workers"] == 12
    assert receipt["capacity_plan"]["gpu_batch_size"] == 8192
    assert receipt["probe_elapsed_seconds"] == pytest.approx(40.0)
    assert (artifact_root / "runtime/preflight_receipt_v2.json").is_file()


def test_v2_launch_requires_separate_exact_authorization(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    prepare_e01_v2(ROOT, CONFIG, artifact_root)
    _preflight(artifact_root)

    with pytest.raises(E01V2RuntimeAuthorizationError, match="formal-run acknowledgement"):
        launch_e01_v2(ROOT, CONFIG, artifact_root, acknowledgement="yes")
    assert not (artifact_root / "runtime/execution.sqlite3").exists()


def test_v2_launch_fails_closed_on_prepared_receipt_drift(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    prepare_e01_v2(ROOT, CONFIG, artifact_root)
    _preflight(artifact_root)
    path = artifact_root / "runtime/prepared_receipt_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["graph"]["total_tasks"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(E01V2RuntimeAuthorizationError, match="prepared receipt hash"):
        launch_e01_v2(
            ROOT,
            CONFIG,
            artifact_root,
            acknowledgement=E01_V2_FORMAL_RUN_ACKNOWLEDGEMENT,
        )


def test_v2_resume_and_status_fail_closed_before_launch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e01-v2"
    assert status_e01_v2(artifact_root, empty_ok=True) == {
        "status": "NOT_STARTED",
        "scientific_results_visible": False,
    }
    with pytest.raises(E01V2RuntimeAuthorizationError, match="execution database"):
        resume_e01_v2(
            ROOT,
            CONFIG,
            artifact_root,
            acknowledgement=E01_V2_FORMAL_RUN_ACKNOWLEDGEMENT,
        )
    with pytest.raises(ValueError, match="allowlisted"):
        dispatch_e01_v2_runtime_command("shell", ROOT, CONFIG, artifact_root)
