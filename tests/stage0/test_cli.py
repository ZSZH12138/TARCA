from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tarca.contracts import GateDecision, GateStatus, StrictContractModel, canonical_json_bytes
from tarca.stage0.checks import freeze_stage0
from tarca.stage0.environment import capture_environment_profile, run_doctor

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_doctor_cli_returns_machine_readable_cpu_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/doctor.py"), "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["checks"]["torch_basic"] == "PASS"
    assert payload["checks"]["pot_sinkhorn"] == "PASS"
    assert payload["checks"]["torch_hook"] == "PASS"
    assert payload["checks"]["python_version"] == "PASS"
    assert payload["checks"]["workspace_disk"] == "PASS"
    assert payload["resources"]["logical_cpu_count"] >= 1
    assert payload["resources"]["disk_free_bytes"] > 0
    assert payload["resources"]["python_version"]
    assert payload["gpu_required"] is False


def test_doctor_function_runs_without_network_or_gpu_requirement() -> None:
    payload = run_doctor(REPO_ROOT)

    assert isinstance(payload, StrictContractModel)
    assert type(payload).__name__ == "DoctorReport"
    assert payload.status == "PASS"
    assert payload.gpu_required is False
    assert payload.checks.model_dump(mode="json") == {
        "pot_sinkhorn": "PASS",
        "python_version": "PASS",
        "pyvene_import": "PASS",
        "torch_basic": "PASS",
        "torch_hook": "PASS",
        "workspace_disk": "PASS",
        "workspace_write": "PASS",
    }


def test_environment_profile_is_a_replaceable_default_not_a_compute_limit() -> None:
    profile = capture_environment_profile(REPO_ROOT)

    assert profile.profile_role == "DEFAULT_EXECUTION_PROFILE"
    assert profile.execution_backend_replaceable is True
    assert profile.compute_boundary_fixed is False
    assert profile.accelerators.cuda_device_count >= 0
    assert profile.accelerators.tested_torch_dtypes == ("float32", "float64")


def test_check_stage0_cli_can_publish_completion_receipt(
    stage0_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = freeze_stage0(stage0_repo)
    decision = GateDecision(
        gate_id="GATE_0_NOVELTY",
        status=GateStatus.PASS,
        rationale="Human-authorized novelty decision.",
        evidence=(manifest.novelty_claims_ref, manifest.related_work_ref),
    )
    decision_path = stage0_repo / "artifacts/stage0/gate0_decision.json"
    decision_path.write_bytes(canonical_json_bytes(decision) + b"\n")

    script_path = REPO_ROOT / "scripts/check_stage0.py"
    spec = importlib.util.spec_from_file_location("tarca_check_stage0_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", stage0_repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_stage0.py", "--complete", "--skip-doctor", "--json"],
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completion_status"] == "COMPLETED"


def test_check_stage0_cli_freeze_stops_at_pending_human_gate(
    stage0_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = REPO_ROOT / "scripts/check_stage0.py"
    spec = importlib.util.spec_from_file_location("tarca_check_stage0_freeze_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", stage0_repo)
    monkeypatch.setattr(sys, "argv", ["check_stage0.py", "--freeze", "--json"])

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "next_action": "Obtain the human GATE_0_NOVELTY decision, then run --complete.",
        "status": "FROZEN_PENDING_GATE_OR_COMPLETION",
    }
