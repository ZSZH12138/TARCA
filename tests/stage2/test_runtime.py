import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.contracts import canonical_json_hash
from tarca.execution import ExecutionStateStore, RunTerminalStatus
from tarca.stage2.runner import Stage2RunResult
from tarca.stage2.runtime import (
    STAGE2_ACKNOWLEDGEMENT,
    STAGE2_RECOVERY_ACKNOWLEDGEMENT,
    Stage2RuntimeAuthorizationError,
    dispatch_stage2_runtime_command,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/stage2_v1.yaml"


def test_stage2_launch_requires_exact_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(Stage2RuntimeAuthorizationError):
        dispatch_stage2_runtime_command("launch", ROOT, CONFIG, tmp_path, acknowledgement="close")
    assert not (tmp_path / "runtime").exists()


def test_stage2_repair_requires_exact_acknowledgement_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        del args, kwargs
        called = True
        return {}

    monkeypatch.setattr(
        "tarca.stage2.runtime.authorize_stage2_device_mismatch_recovery", forbidden
    )

    with pytest.raises(Stage2RuntimeAuthorizationError, match="recovery acknowledgement"):
        dispatch_stage2_runtime_command(
            "repair", ROOT, CONFIG, tmp_path, acknowledgement="close"
        )

    assert called is False


def test_stage2_restore_input_dispatches_only_the_frozen_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Path] = {}
    archive = tmp_path / "recovery.tar.gz"
    bundle = tmp_path / "bundle.tar.gz"

    def restore(
        repository_root: Path,
        *,
        recovery_archive: Path,
        server_bundle: Path,
        spec_path: Path,
    ) -> dict[str, object]:
        captured.update(
            repository_root=repository_root,
            recovery_archive=recovery_archive,
            server_bundle=server_bundle,
            spec_path=spec_path,
        )
        return {"status": "RESTORED"}

    monkeypatch.setattr("tarca.stage2.runtime.restore_stage2_recovery_archive", restore)

    result = dispatch_stage2_runtime_command(
        "restore-input",
        ROOT,
        CONFIG,
        ROOT / "artifacts/stage2",
        recovery_archive=archive,
        server_bundle=bundle,
    )

    assert result["status"] == "RESTORED"
    assert captured == {
        "repository_root": ROOT.resolve(),
        "recovery_archive": archive,
        "server_bundle": bundle,
        "spec_path": ROOT / "configs/stage2/stage2_device_mismatch_recovery_v1.json",
    }


def test_stage2_repair_dispatches_the_frozen_recovery_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Path] = {}

    def authorize(
        repository_root: Path,
        artifact_root: Path,
        *,
        spec_path: Path,
        recovery_input_receipt_path: Path,
    ) -> dict[str, object]:
        captured.update(
            repository_root=repository_root,
            artifact_root=artifact_root,
            spec_path=spec_path,
            recovery_input_receipt_path=recovery_input_receipt_path,
        )
        return {"status": "AUTHORIZED"}

    monkeypatch.setattr(
        "tarca.stage2.runtime.authorize_stage2_device_mismatch_recovery", authorize
    )

    result = dispatch_stage2_runtime_command(
        "repair",
        ROOT,
        CONFIG,
        tmp_path,
        acknowledgement=STAGE2_RECOVERY_ACKNOWLEDGEMENT,
    )

    assert result["status"] == "AUTHORIZED"
    assert captured["spec_path"] == (
        ROOT / "configs/stage2/stage2_device_mismatch_recovery_v1.json"
    )
    assert captured["recovery_input_receipt_path"] == (
        tmp_path.resolve() / "runtime/recovery_input_receipt.json"
    )


def test_stage2_resume_activates_complete_checkpoint_recovery_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    runtime = artifact_root / "runtime"
    ExecutionStateStore(runtime / "execution.sqlite3")
    graph_id = "stage2-graph-" + "a" * 64
    run_id = "run-" + "a" * 64

    def write_receipt(name: str, payload: dict[str, object]) -> None:
        sealed = {**payload, "receipt_sha256": canonical_json_hash(payload)}
        path = runtime / name
        path.write_text(
            json.dumps(sealed, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    write_receipt(
        "launch_authorization_receipt.json",
        {
            "schema_version": "tarca-stage2-launch-v1",
            "status": "AUTHORIZED",
            "run_id": run_id,
            "graph_id": graph_id,
        },
    )
    write_receipt(
        "device_mismatch_recovery_receipt.json",
        {
            "schema_version": "tarca-stage2-recovery-authorization-v1",
            "status": "AUTHORIZED",
            "reason": "DEVICE_MISMATCH_V1",
            "run_id": run_id,
            "graph_id": graph_id,
        },
    )
    graph = SimpleNamespace(graph_id=graph_id)
    monkeypatch.setattr("tarca.stage2.runtime._compiled_graph", lambda root, config: graph)
    monkeypatch.setattr("tarca.stage2.runtime._runtime_capacity", lambda root: object())
    environments: list[dict[str, str]] = []

    def backend(*args: object, **kwargs: object) -> object:
        del args
        environments.append(kwargs["environment_overrides"])
        return object()

    monkeypatch.setattr("tarca.stage2.runtime.LocalMultiProcessBackend", backend)
    monkeypatch.setattr(
        "tarca.stage2.runtime.run_stage2",
        lambda *args, **kwargs: Stage2RunResult(
            run_id=run_id,
            graph_id=graph_id,
            status=RunTerminalStatus.COMPLETED,
            completed=(),
        ),
    )

    dispatch_stage2_runtime_command(
        "resume",
        ROOT,
        CONFIG,
        artifact_root,
        acknowledgement=STAGE2_ACKNOWLEDGEMENT,
    )

    assert environments == [
        {
            "TARCA_EXECUTION_KIND": "stage2-v1",
            "TARCA_STAGE2_RECOVERY_MODE": "DEVICE_MISMATCH_V1",
        }
    ]


def test_stage2_prepare_is_idempotent_and_executes_no_formal_tasks(tmp_path: Path) -> None:
    first = dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    second = dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    assert first == second
    assert first["formal_tasks_executed"] == 0
    assert first["status"] == "PREPARED"


def test_stage2_full_local_runtime_lifecycle_without_scientific_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    prepared = dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    assert dispatch_stage2_runtime_command(
        "dry-run", ROOT, CONFIG, artifact_root
    )["prepared_receipt_sha256"] == prepared["receipt_sha256"]
    evidence = tmp_path / "preflight.json"
    evidence.write_text(
        """{"amp_finite":true,"checkpoint_roundtrip_passed":true,"estimated_remaining_seconds":3600,"eta_gate_passed":true,"formal_tasks_executed":0,"fp32_finite":true,"probe_contract":"stage2-v1-two-exact-neural-concurrent-max-epochs","remaining_rental_hours":24,"reset_margin_hours":1,"source_hashes_verified":true,"status":"PREFLIGHT_PASS"}\n""",
        encoding="utf-8",
    )
    dispatch_stage2_runtime_command(
        "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence
    )
    graph = SimpleNamespace(graph_id="stage2-graph-" + "a" * 64)
    monkeypatch.setattr("tarca.stage2.runtime._compiled_graph", lambda root, config: graph)
    monkeypatch.setattr("tarca.stage2.runtime._runtime_capacity", lambda root: object())
    monkeypatch.setattr(
        "tarca.stage2.runtime.LocalMultiProcessBackend", lambda *args, **kwargs: object()
    )

    def fake_run(*args: object, **kwargs: object) -> Stage2RunResult:
        del args, kwargs
        return Stage2RunResult(
            run_id="run-" + "a" * 64,
            graph_id=graph.graph_id,
            status=RunTerminalStatus.COMPLETED,
            completed=(),
        )

    monkeypatch.setattr("tarca.stage2.runtime.run_stage2", fake_run)
    launched = dispatch_stage2_runtime_command(
        "launch",
        ROOT,
        CONFIG,
        artifact_root,
        acknowledgement=STAGE2_ACKNOWLEDGEMENT,
    )
    assert launched["status"] == "COMPLETED"
    database = artifact_root / "runtime/execution.sqlite3"
    ExecutionStateStore(database)
    resumed = dispatch_stage2_runtime_command(
        "resume",
        ROOT,
        CONFIG,
        artifact_root,
        acknowledgement=STAGE2_ACKNOWLEDGEMENT,
    )
    assert resumed["status"] == "COMPLETED"
    assert dispatch_stage2_runtime_command("status", ROOT, CONFIG, artifact_root)[
        "scientific_results_visible"
    ] is False
    recovered = dispatch_stage2_runtime_command("recover", ROOT, CONFIG, artifact_root)
    assert recovered["status"] == "RECOVERED"
    assert dispatch_stage2_runtime_command("freeze", ROOT, CONFIG, artifact_root)[
        "status"
    ] == "FREEZE_REQUIRES_COMPLETED_MANIFEST"
    with pytest.raises(ValueError, match="allowlisted"):
        dispatch_stage2_runtime_command("unknown", ROOT, CONFIG, artifact_root)


def test_stage2_preflight_rejects_status_only_evidence(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    evidence = tmp_path / "preflight.json"
    evidence.write_text('{"status":"PREFLIGHT_PASS"}\n', encoding="utf-8")
    with pytest.raises(Stage2RuntimeAuthorizationError, match="evidence is incomplete"):
        dispatch_stage2_runtime_command(
            "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence
        )


def test_stage2_preflight_rejects_noncanonical_probe_contract(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    evidence = tmp_path / "preflight.json"
    evidence.write_text(
        """{"amp_finite":true,"checkpoint_roundtrip_passed":true,"estimated_remaining_seconds":3600,"eta_gate_passed":true,"formal_tasks_executed":0,"fp32_finite":true,"probe_contract":"ad-hoc-probe","remaining_rental_hours":24,"reset_margin_hours":1,"source_hashes_verified":true,"status":"PREFLIGHT_PASS"}\n""",
        encoding="utf-8",
    )
    with pytest.raises(Stage2RuntimeAuthorizationError, match="failed boundary"):
        dispatch_stage2_runtime_command(
            "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence
        )


def test_stage2_preflight_accepts_only_the_complete_checkpoint_recovery_probe(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, artifact_root)
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
    (artifact_root / "runtime/recovery_input_receipt.json").write_text(
        json.dumps(recovery_input, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "recovery-preflight.json"
    evidence.write_text(
        json.dumps(
            {
                "amp_finite": True,
                "checkpoint_roundtrip_passed": True,
                "checkpoint_hashes_unchanged": True,
                "complete_checkpoint_fast_path_passed": True,
                "estimated_remaining_seconds": 7200,
                "eta_gate_passed": True,
                "formal_tasks_executed": 0,
                "fp32_finite": True,
                "probe_contract": "stage2-v1-recovery-two-complete-checkpoints-concurrent",
                "recovery_mode": "DEVICE_MISMATCH_V1",
                "remaining_rental_hours": 24,
                "reset_margin_hours": 1,
                "source_hashes_verified": True,
                "status": "PREFLIGHT_PASS",
                "zero_optimizer_steps": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = dispatch_stage2_runtime_command(
        "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence
    )

    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["recovery_mode"] == "DEVICE_MISMATCH_V1"


def test_stage2_preflight_rejects_a_tampered_recovery_input_receipt(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts/stage2"
    dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, artifact_root)
    runtime = artifact_root / "runtime"
    (runtime / "recovery_input_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "tarca-stage2-recovery-input-v1",
                "status": "RESTORED",
                "source_archive_sha256": "a" * 64,
                "source_manifest_sha256": "b" * 64,
                "source_database_sha256": "c" * 64,
                "server_bundle_sha256": "d" * 64,
                "receipt_sha256": "e" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "recovery-preflight.json"
    evidence.write_text(
        json.dumps(
            {
                "amp_finite": True,
                "checkpoint_roundtrip_passed": True,
                "checkpoint_hashes_unchanged": True,
                "complete_checkpoint_fast_path_passed": True,
                "estimated_remaining_seconds": 7200,
                "eta_gate_passed": True,
                "formal_tasks_executed": 0,
                "fp32_finite": True,
                "probe_contract": "stage2-v1-recovery-two-complete-checkpoints-concurrent",
                "recovery_mode": "DEVICE_MISMATCH_V1",
                "remaining_rental_hours": 24,
                "reset_margin_hours": 1,
                "source_hashes_verified": True,
                "status": "PREFLIGHT_PASS",
                "zero_optimizer_steps": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Stage2RuntimeAuthorizationError, match="recovery input receipt"):
        dispatch_stage2_runtime_command(
            "preflight", ROOT, CONFIG, artifact_root, evidence_path=evidence
        )
