import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e02.runner import E02RunResult
from tarca.e02.runtime import (
    E02_ACKNOWLEDGEMENT,
    E02RuntimeAuthorizationError,
    dispatch_e02_runtime_command,
)
from tarca.execution import ExecutionStateStore, RunTerminalStatus

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e02/e02_v1.yaml"
FREEZE_SHA256 = "37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166"


def _server_evidence(artifact_root: Path) -> Path:
    restore_payload = {
        "schema_version": "tarca-e02-stage2-restore-v1",
        "status": "RESTORED",
        "complete_archive_filename": "tarca-stage2-v1-complete-20260901T011423Z.tar.gz",
        "complete_archive_sha256": (
            "7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a"
        ),
        "stage2_freeze_receipt_sha256": FREEZE_SHA256,
        "e02_scientific_config_sha256": (
            "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c"
        ),
        "restored_file_count": 70,
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }
    restore = {
        **restore_payload,
        "receipt_sha256": canonical_json_hash(restore_payload),
    }
    runtime = artifact_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "stage2_restore_receipt.json").write_bytes(
        canonical_json_bytes(restore) + b"\n"
    )
    bundle_payload = {
        "schema_version": "tarca-e02-server-bundle-verification-v1",
        "status": "VERIFIED",
        "server_bundle_sha256": "b" * 64,
        "verified_file_count": 100,
        "formal_tasks_executed": 0,
    }
    bundle = {
        **bundle_payload,
        "receipt_sha256": canonical_json_hash(bundle_payload),
    }
    (runtime / "server_bundle_verification_receipt.json").write_bytes(
        canonical_json_bytes(bundle) + b"\n"
    )
    evidence = {
        "schema_version": "tarca-e02-server-preflight-evidence-v1",
        "status": "PREFLIGHT_PASS",
        "remaining_rental_hours": 24.0,
        "observed_at_utc": "2026-09-01T00:00:00+00:00",
        "rental_reset_at_utc": "2099-09-02T00:00:00+00:00",
        "complete_archive_sha256": restore_payload["complete_archive_sha256"],
        "stage2_restore_receipt_sha256": restore["receipt_sha256"],
        "server_bundle_sha256": bundle["server_bundle_sha256"],
        "server_bundle_verification_receipt_sha256": bundle["receipt_sha256"],
        "stage2_freeze_receipt_sha256": FREEZE_SHA256,
        "e02_scientific_config_sha256": restore_payload["e02_scientific_config_sha256"],
        "source_hashes_verified": True,
        "hardware": {},
        "gpu_count": 2,
        "work_cpu_cores": 24,
        "scheduler_monitor_cores": 1,
        "system_io_cores": 3,
        "host_memory_ceiling_gib": 200,
        "storage_floor_gib": 200,
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
        "probe_contract": "e02-v1-three-frozen-checkpoints-two-gpu-waves",
        "estimated_remaining_seconds": 7200.0,
        "reset_margin_hours": 1.0,
        "eta_gate_passed": True,
    }
    path = runtime / "bootstrap_evidence.json"
    path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    return path


def test_e02_prepare_does_not_open_formal_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tarca.stage2.data._read_formal_storage",
        lambda: (_ for _ in ()).throw(AssertionError("formal storage opened")),
    )
    receipt = dispatch_e02_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    assert receipt["formal_tasks_executed"] == 0
    assert not (tmp_path / "runtime/sealed_access_grant.json").exists()


def test_e02_launch_wrong_token_creates_no_grant(tmp_path: Path) -> None:
    with pytest.raises(E02RuntimeAuthorizationError):
        dispatch_e02_runtime_command("launch", ROOT, CONFIG, tmp_path, acknowledgement="close")
    assert not (tmp_path / "runtime/sealed_access_grant.json").exists()


def test_e02_full_local_runtime_lifecycle_without_formal_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/e02"
    prepared = dispatch_e02_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    assert dispatch_e02_runtime_command(
        "dry-run", ROOT, CONFIG, artifact_root
    )["prepared_receipt_sha256"] == prepared["receipt_sha256"]
    freeze = SimpleNamespace(receipt_sha256=FREEZE_SHA256)
    monkeypatch.setattr(
        "tarca.e02.runtime.verify_frozen_stage2_suite", lambda root: freeze
    )
    evidence_path = _server_evidence(artifact_root)
    preflight = dispatch_e02_runtime_command(
        "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence_path
    )
    assert preflight["evidence_sha256"] == sha256_file(evidence_path)
    graph = SimpleNamespace(graph_id="e02-graph-" + "b" * 64)
    monkeypatch.setattr("tarca.e02.runtime._compiled_graph", lambda root, config: graph)
    monkeypatch.setattr("tarca.e02.runtime._runtime_capacity", lambda root: object())
    monkeypatch.setattr("tarca.e02.runtime._worker_cpu_ids", lambda: tuple(range(4, 28)))
    monkeypatch.setattr(
        "tarca.e02.runtime.LocalMultiProcessBackend", lambda *args, **kwargs: object()
    )

    def fake_run(*args, **kwargs):
        ExecutionStateStore(kwargs["database_path"])
        frozen = artifact_root / "frozen/v1"
        frozen.mkdir(parents=True, exist_ok=True)
        (frozen / "e02_receipt.json").write_text(
            '{"outcome":"PASS","receipt_sha256":"' + "c" * 64 + '"}\n',
            encoding="utf-8",
        )
        return E02RunResult(
            run_id="run-" + "b" * 64,
            graph_id=graph.graph_id,
            status=RunTerminalStatus.COMPLETED,
            completed=(),
        )

    monkeypatch.setattr("tarca.e02.runtime.run_e02_formal", fake_run)
    launched = dispatch_e02_runtime_command(
        "launch",
        ROOT,
        CONFIG,
        artifact_root,
        acknowledgement=E02_ACKNOWLEDGEMENT,
    )
    assert launched["status"] == "COMPLETED"
    assert (artifact_root / "runtime/sealed_access_grant.json").is_file()
    resumed = dispatch_e02_runtime_command(
        "resume",
        ROOT,
        CONFIG,
        artifact_root,
        acknowledgement=E02_ACKNOWLEDGEMENT,
    )
    assert resumed["status"] == "COMPLETED"
    assert dispatch_e02_runtime_command("status", ROOT, CONFIG, artifact_root)[
        "scientific_results_visible"
    ] is False
    assert dispatch_e02_runtime_command("finalize", ROOT, CONFIG, artifact_root)[
        "outcome"
    ] == "PASS"
    assert dispatch_e02_runtime_command("recover", ROOT, CONFIG, artifact_root)[
        "status"
    ] == "RECOVERED"
    with pytest.raises(ValueError, match="allowlisted"):
        dispatch_e02_runtime_command("unknown", ROOT, CONFIG, artifact_root)


def test_e02_preflight_rejects_missing_server_evidence_before_formal_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/e02"
    dispatch_e02_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    monkeypatch.setattr(
        "tarca.e02.runtime.verify_frozen_stage2_suite",
        lambda root: SimpleNamespace(receipt_sha256=FREEZE_SHA256),
    )

    with pytest.raises(E02RuntimeAuthorizationError, match="server preflight evidence"):
        dispatch_e02_runtime_command(
            "preflight",
            ROOT,
            CONFIG,
            artifact_root,
            evidence_path=artifact_root / "runtime/missing.json",
        )

    assert not (artifact_root / "runtime/preflight_receipt.json").exists()
    assert not (artifact_root / "runtime/sealed_access_grant.json").exists()


def test_e02_launch_rechecks_bound_server_evidence_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/e02"
    dispatch_e02_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    evidence_path = _server_evidence(artifact_root)
    monkeypatch.setattr(
        "tarca.e02.runtime.verify_frozen_stage2_suite",
        lambda root: SimpleNamespace(receipt_sha256=FREEZE_SHA256),
    )
    dispatch_e02_runtime_command(
        "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence_path
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["estimated_remaining_seconds"] = 1.0
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")

    with pytest.raises(E02RuntimeAuthorizationError, match="evidence hash drifted"):
        dispatch_e02_runtime_command(
            "launch",
            ROOT,
            CONFIG,
            artifact_root,
            acknowledgement=E02_ACKNOWLEDGEMENT,
        )

    assert not (artifact_root / "runtime/sealed_access_grant.json").exists()
