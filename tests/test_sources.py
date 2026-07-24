from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.stage0.sources as sources_module  # noqa: E402
from tarca.stage0.sources import SourceEntry, resolve_source  # noqa: E402

SHA = "0123456789abcdefABCDEF0123456789abcdefAB"


def make_entry(tmp_path: Path, **changes: str) -> SourceEntry:
    values = {
        "name": "example",
        "paper_title": "Example paper",
        "paper_url": "https://example.test/paper",
        "repository_url": "https://example.test/repository.git",
        "role": "test",
        "license": "MIT",
        "default_branch": "main",
        "verified_commit": "a" * 40,
        "verified_at": "2026-07-23",
        "local_reference_path": ".cache/third_party/example",
        "notes": "Test fixture.",
    }
    return SourceEntry.model_validate({**values, **changes})


def completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_resolve_source_verifies_complete_remote_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed(args, stdout=f"{SHA}\trefs/heads/main\n")

    monkeypatch.setattr(sources_module, "run", fake_run)

    resolution = resolve_source(entry, timeout_seconds=7)

    assert resolution.status == "VERIFIED"
    assert resolution.resolved_commit == SHA
    assert resolution.source == "remote"


def test_resolve_source_verifies_local_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_repository = tmp_path / ".cache" / "third_party" / "example"
    local_repository.mkdir(parents=True)
    (local_repository / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sources_module, "PROJECT_ROOT", tmp_path, raising=False)
    entry = make_entry(tmp_path)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed(args, stdout=f"{SHA}\n")

    monkeypatch.setattr(sources_module, "run", fake_run)

    resolution = resolve_source(entry)

    assert resolution.status == "VERIFIED"
    assert resolution.resolved_commit == SHA
    assert resolution.source == "local"


def test_existing_non_repository_path_uses_remote_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_reference = tmp_path / ".cache" / "third_party" / "example"
    local_reference.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sources_module, "PROJECT_ROOT", tmp_path, raising=False)
    entry = make_entry(tmp_path)
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        return completed(args, stdout=f"{SHA}\trefs/heads/main\n")

    monkeypatch.setattr(sources_module, "run", fake_run)

    resolution = resolve_source(entry)

    assert resolution.status == "VERIFIED"
    assert resolution.source == "remote"
    args = observed["args"]
    assert isinstance(args, list)
    assert Path(args[0]).is_absolute()
    assert Path(args[0]).stem.lower() == "git"
    assert args[1:] == [
        "ls-remote",
        entry.repository_url,
        "refs/heads/main",
    ]


@pytest.mark.parametrize(
    "stdout",
    [
        f"{'g' * 40}\trefs/heads/main\n",
        "abc123\trefs/heads/main\n",
        f"{SHA}\n",
        f"{SHA}\trefs/heads/other\n",
        f"{SHA}\trefs/heads/main\n{SHA}\trefs/heads/other\n",
    ],
    ids=["nonhex", "short", "missing-ref", "wrong-ref", "multiple-lines"],
)
def test_resolve_source_rejects_invalid_remote_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    entry = make_entry(tmp_path)
    monkeypatch.setattr(
        sources_module,
        "run",
        lambda args, **kwargs: completed(args, stdout=stdout),
    )

    resolution = resolve_source(entry)

    assert resolution.status == "INVALID_RESPONSE"
    assert resolution.resolved_commit == ""


def test_resolve_source_reports_missing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path)
    monkeypatch.setattr(
        sources_module,
        "run",
        lambda args, **kwargs: completed(
            args,
            returncode=128,
            stderr="fatal: repository not found",
        ),
    )

    resolution = resolve_source(entry)

    assert resolution.status == "NO_REPOSITORY"
    assert resolution.resolved_commit == ""


def test_resolve_source_reports_generic_network_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path)
    monkeypatch.setattr(
        sources_module,
        "run",
        lambda args, **kwargs: completed(
            args,
            returncode=128,
            stderr="fatal: unable to access remote",
        ),
    )

    resolution = resolve_source(entry)

    assert resolution.status == "NETWORK_ERROR"
    assert resolution.resolved_commit == ""


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["git", "ls-remote"], timeout=3),
        FileNotFoundError("git executable not found"),
    ],
    ids=["timeout", "missing-executable"],
)
def test_resolve_source_maps_subprocess_failures_to_network_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    entry = make_entry(tmp_path)

    def failing_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(sources_module, "run", failing_run)

    resolution = resolve_source(entry, timeout_seconds=3)

    assert resolution.status == "NETWORK_ERROR"
    assert resolution.resolved_commit == ""


def test_resolve_source_uses_non_mutating_argument_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path, default_branch="main;echo-unsafe")
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(args=args, kwargs=kwargs)
        return completed(args, stdout=f"{SHA}\trefs/heads/main;echo-unsafe\n")

    monkeypatch.setattr(sources_module, "run", fake_run)

    resolution = resolve_source(entry, timeout_seconds=11)

    assert resolution.status == "VERIFIED"
    args = observed["args"]
    assert isinstance(args, list)
    assert Path(args[0]).is_absolute()
    assert Path(args[0]).stem.lower() == "git"
    assert args[1:] == [
        "ls-remote",
        entry.repository_url,
        "refs/heads/main;echo-unsafe",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("shell", False) is False
    assert kwargs["timeout"] == 11
    assert "clone" not in args
