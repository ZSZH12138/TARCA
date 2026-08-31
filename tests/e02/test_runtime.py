from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.e02.runner import E02RunResult
from tarca.e02.runtime import (
    E02_ACKNOWLEDGEMENT,
    E02RuntimeAuthorizationError,
    dispatch_e02_runtime_command,
)
from tarca.execution import ExecutionStateStore, RunTerminalStatus

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e02/e02_v1.yaml"


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
    freeze = SimpleNamespace(receipt_sha256="f" * 64)
    monkeypatch.setattr(
        "tarca.e02.runtime.verify_frozen_stage2_suite", lambda root: freeze
    )
    dispatch_e02_runtime_command("preflight", ROOT, CONFIG, artifact_root)
    graph = SimpleNamespace(graph_id="e02-graph-" + "b" * 64)
    monkeypatch.setattr("tarca.e02.runtime._compiled_graph", lambda root, config: graph)
    monkeypatch.setattr("tarca.e02.runtime._runtime_capacity", lambda root: object())
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
