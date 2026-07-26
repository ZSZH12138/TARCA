"""CLI tests for Stage 1B synthetic dataset generation and engineering smoke."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from scripts import build_synthetic_dataset as build_cli
from scripts import run_synthetic_oracle_smoke as smoke_cli

from tarca.data.synthetic.dataset_builder import load_synthetic_config

ROOT = Path(__file__).parents[3]
EASY_CONFIG = ROOT / "configs" / "synthetic" / "synthetic_easy.yaml"


def _copy_config(project_root: Path, source: Path = EASY_CONFIG) -> Path:
    relative = Path("configs/synthetic") / source.name
    destination = project_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative


@pytest.mark.parametrize(
    "script",
    ("scripts/build_synthetic_dataset.py", "scripts/run_synthetic_oracle_smoke.py"),
)
def test_cli_help_succeeds(script: str) -> None:
    completed = subprocess.run(
        (sys.executable, script, "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(
    "unsafe",
    ("../escape", "nested/../../escape", "C:/absolute", "safe/name:stream"),
)
def test_repository_path_rejects_lexical_escape(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="path"):
        build_cli.resolve_repository_path(unsafe, project_root=tmp_path)


def test_repository_path_rejects_symlink_component(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this Windows host")
    with pytest.raises(ValueError, match=r"symlink|reparse"):
        build_cli.resolve_repository_path("linked/output", project_root=tmp_path)


def test_build_cli_is_reproducible_and_seed_override_changes_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _copy_config(tmp_path)
    hashes: list[str] = []
    for output in ("runs/first", "runs/second"):
        assert (
            build_cli.main(
                ("--config", str(config), "--output", output, "--smoke"),
                project_root=tmp_path,
            )
            == 0
        )
        hashes.append(json.loads(capsys.readouterr().out)["dataset_hash"])
        assert len(tuple((tmp_path / output).iterdir())) == 9
    assert hashes[0] == hashes[1]

    assert (
        build_cli.main(
            ("--config", str(config), "--output", "runs/changed", "--seed", "7"),
            project_root=tmp_path,
        )
        == 0
    )
    changed = json.loads(capsys.readouterr().out)
    assert changed["dataset_hash"] != hashes[0]
    resolved = yaml.safe_load(
        (tmp_path / "runs/changed/config_resolved.yaml").read_text(encoding="utf-8")
    )
    assert resolved["root_seed"] == 7


def test_build_cli_invalid_config_is_nonzero_and_creates_no_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "configs" / "bad.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("name: incomplete\n", encoding="utf-8")
    assert (
        build_cli.main(
            ("--config", "configs/bad.yaml", "--output", "runs/bad"),
            project_root=tmp_path,
        )
        == 2
    )
    assert not (tmp_path / "runs/bad").exists()
    assert json.loads(capsys.readouterr().err)["status"] == "INPUT_ERROR"


def test_build_cli_failed_persisted_validation_cleans_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _copy_config(tmp_path)
    config = load_synthetic_config(tmp_path / relative)
    reports = iter(
        (
            SimpleNamespace(status="VALIDATION_PASS"),
            SimpleNamespace(status="VALIDATION_FAIL"),
        )
    )
    monkeypatch.setattr(build_cli, "load_synthetic_config", lambda _: config)
    monkeypatch.setattr(build_cli, "build_synthetic_dataset", lambda _: object())
    monkeypatch.setattr(build_cli, "validate_synthetic_dataset", lambda *_1, **_2: next(reports))

    def persist(_: object, path: Path) -> object:
        path.mkdir()
        return object()

    monkeypatch.setattr(build_cli, "persist_synthetic_dataset", persist)
    output = tmp_path / "runs" / "failed"
    assert (
        build_cli.main(
            ("--config", str(relative), "--output", "runs/failed"),
            project_root=tmp_path,
        )
        == 1
    )
    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))


def test_smoke_cli_writes_cpu_only_atomic_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _copy_config(tmp_path)
    assert (
        smoke_cli.main(
            ("--config", str(config), "--output", "artifacts/smoke"),
            project_root=tmp_path,
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    output = tmp_path / "artifacts" / "smoke"
    payload = json.loads((output / "e01_engineering_smoke.json").read_text(encoding="utf-8"))
    assert summary["status"] == payload["status"] == "ENGINEERING_SMOKE_PASS"
    assert payload["gpu_used"] is False
    assert payload["pair_count"] <= 16
    assert payload["mc_sample_sizes"] == [32, 64, 128, 256]
    assert (output / "e01_engineering_smoke.md").is_file()
    assert len(tuple((output / "dataset").iterdir())) == 9
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))


def test_smoke_failure_leaves_no_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _copy_config(tmp_path)
    config = load_synthetic_config(tmp_path / config_path)
    monkeypatch.setattr(smoke_cli, "load_synthetic_config", lambda _: config)
    monkeypatch.setattr(smoke_cli, "build_synthetic_dataset", lambda _: object())
    monkeypatch.setattr(smoke_cli, "persist_synthetic_dataset", lambda *_: object())

    def fail(*_: object, **__: object) -> object:
        raise ValueError("forced smoke failure")

    monkeypatch.setattr(smoke_cli, "run_e01_engineering_smoke", fail)
    output = tmp_path / "artifacts" / "failed"
    assert (
        smoke_cli.main(
            ("--config", str(config_path), "--output", "artifacts/failed"),
            project_root=tmp_path,
        )
        == 1
    )
    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))
    assert json.loads(capsys.readouterr().err)["status"] == "SMOKE_ERROR"


def test_smoke_cli_rejects_non_easy_config_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_synthetic_config(EASY_CONFIG)
    medium_named = type(config).model_validate(
        {**config.model_dump(mode="python"), "name": "synthetic_medium"},
        strict=True,
    )
    relative = _copy_config(tmp_path)
    monkeypatch.setattr(smoke_cli, "load_synthetic_config", lambda _: medium_named)
    called = False

    def record(_: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(smoke_cli, "build_synthetic_dataset", record)
    assert (
        smoke_cli.main(
            ("--config", str(relative), "--output", "artifacts/rejected"),
            project_root=tmp_path,
        )
        == 2
    )
    assert called is False
    assert json.loads(capsys.readouterr().err)["status"] == "INPUT_ERROR"


def test_smoke_cli_never_enables_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    relative = _copy_config(tmp_path)
    seen_environment: list[str | None] = []

    def stop_after_cpu_guard(_: object) -> object:
        seen_environment.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        raise ValueError("stop")

    monkeypatch.setattr(smoke_cli, "build_synthetic_dataset", stop_after_cpu_guard)
    assert (
        smoke_cli.main(
            ("--config", str(relative), "--output", "artifacts/cpu"),
            project_root=tmp_path,
        )
        == 1
    )
    assert seen_environment == [""]
