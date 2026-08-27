from __future__ import annotations

import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.stage1b.config import load_world_suite
from tarca.stage1b.sources import SourceAcquisitionMode, SourceMaterializationReceipt

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/run_stage1b_runtime.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(SCRIPT), *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_runtime_cli_has_no_formal_experiment_commands() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    assert all(name in completed.stdout for name in ("preflight", "launch", "resume", "status"))
    assert "E01" not in completed.stdout and "E02" not in completed.stdout


def test_empty_status_is_safe_json_and_requires_explicit_empty_permission(tmp_path: Path) -> None:
    blocked = _run("--artifact-root", str(tmp_path), "status")
    allowed = _run("--artifact-root", str(tmp_path), "status", "--empty-ok")

    assert blocked.returncode == 1
    payload = json.loads(allowed.stdout)
    assert payload == {"status": "EMPTY"}
    assert "crps" not in allowed.stdout.lower()
    assert "truth" not in allowed.stdout.lower()


def test_resume_requires_an_existing_execution_database(tmp_path: Path) -> None:
    completed = _run("--artifact-root", str(tmp_path), "resume")

    assert completed.returncode == 1
    assert "execution database" in completed.stdout.lower()


def test_launch_receipts_reject_source_acquisition_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "environment_receipt_v2.json").write_text("{}\n", encoding="utf-8")
    (runtime / "precision_receipt_v2.json").write_text("{}\n", encoding="utf-8")
    (runtime / "hardware_probe_v2.json").write_text(
        json.dumps({"decision": {"feasible": True}}), encoding="utf-8"
    )
    (runtime / "official_sources_receipt_v2.json").write_text(
        json.dumps(
            {
                "source_mode": "offline-capsule",
                "capsule_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "online")
    namespace = runpy.run_path(str(SCRIPT), run_name="stage1b_runtime_receipt_test")

    with pytest.raises(RuntimeError, match="source acquisition mode"):
        namespace["_required_receipts"](tmp_path)


def test_preflight_materializes_sources_from_the_offline_capsule_cache_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml").sources[0]
    checkout = tmp_path / source.source_id / source.commit
    receipt = SourceMaterializationReceipt(
        source_id=source.source_id,
        repository_url=source.repository_url,
        commit=source.commit,
        checkout_root=checkout,
        tree_sha256="a" * 64,
        asset_sha256=tuple((asset.relative_path, asset.sha256) for asset in source.assets),
        authorization_id=source.authorization_id,
        materialized_at_utc=datetime.now(UTC),
    )
    namespace = runpy.run_path(str(SCRIPT), run_name="stage1b_runtime_source_test")
    runtime_globals = namespace["_materialize_sources"].__globals__
    calls: list[SourceAcquisitionMode] = []
    monkeypatch.setitem(
        runtime_globals,
        "repository_v2_inputs",
        lambda _root: SimpleNamespace(world_suite=SimpleNamespace(sources=(source,))),
    )
    monkeypatch.setitem(
        runtime_globals,
        "SubprocessGitRunner",
        SimpleNamespace(discover=lambda: object()),
    )
    monkeypatch.setitem(
        runtime_globals,
        "verify_source_capsule_import",
        lambda _sources, _cache: SimpleNamespace(
            capsule_sha256="b" * 64,
            manifest_sha256="c" * 64,
        ),
    )

    def materialize(
        _source: object, _cache: Path, _runner: object, *, mode: SourceAcquisitionMode
    ) -> SourceMaterializationReceipt:
        calls.append(mode)
        return receipt

    monkeypatch.setitem(runtime_globals, "materialize_source", materialize)
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "offline-capsule")
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", str(tmp_path))

    result = namespace["_materialize_sources"]()

    assert calls == [SourceAcquisitionMode.OFFLINE_CAPSULE]
    assert result["source_mode"] == "offline-capsule"
    assert result["capsule_sha256"] == "b" * 64
    assert result["manifest_sha256"] == "c" * 64
