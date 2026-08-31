from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.execution import ExecutionStateStore, RunTerminalStatus
from tarca.stage2.runner import Stage2RunResult
from tarca.stage2.runtime import (
    STAGE2_ACKNOWLEDGEMENT,
    Stage2RuntimeAuthorizationError,
    dispatch_stage2_runtime_command,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/stage2_v1.yaml"


def test_stage2_launch_requires_exact_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(Stage2RuntimeAuthorizationError):
        dispatch_stage2_runtime_command("launch", ROOT, CONFIG, tmp_path, acknowledgement="close")
    assert not (tmp_path / "runtime").exists()


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
        """{"amp_finite":true,"checkpoint_roundtrip_passed":true,"estimated_remaining_seconds":3600,"formal_tasks_executed":0,"fp32_finite":true,"remaining_rental_hours":24,"source_hashes_verified":true,"status":"PREFLIGHT_PASS"}\n""",
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

    def fake_run(*args, **kwargs):
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
