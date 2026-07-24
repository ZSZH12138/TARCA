from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.stage0.executables as executables  # noqa: E402


def test_resolve_external_executable_returns_absolute_path_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    executable = tmp_path / "tools" / "git.exe"
    workspace.mkdir()
    executable.parent.mkdir()
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(executables.shutil, "which", lambda _name: str(executable))

    resolved = executables.resolve_external_executable("git", workspace)

    assert resolved == str(executable.resolve())
    assert Path(resolved).is_absolute()


def test_resolve_external_executable_rejects_workspace_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    executable = workspace / "git.exe"
    workspace.mkdir()
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(executables.shutil, "which", lambda _name: str(executable))

    with pytest.raises(PermissionError, match="workspace"):
        executables.resolve_external_executable("git", workspace)


def test_resolve_external_executable_reports_missing_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executables.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="git"):
        executables.resolve_external_executable("git", tmp_path)
