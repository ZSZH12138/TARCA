from __future__ import annotations

import hashlib
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tarca.stage1b.config import SourceConfig
from tarca.stage1b.sources import (
    MaterializedSources,
    SourceAcquisitionMode,
    SourceVerificationError,
    materialize_source,
    source_acquisition_mode_from_environment,
    source_cache_root_from_environment,
    verify_materialized_source,
)

COMMIT = "a" * 40
OFFICIAL_BYTES = b"def generate():\n    return 'official'\n"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeGit:
    def __init__(self, files: dict[str, bytes], commit: str = COMMIT) -> None:
        self._files = dict(files)
        self._commit = commit
        self.calls: list[tuple[tuple[str, ...], Path | None]] = []

    def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
        self.calls.append((arguments, cwd))
        if arguments[:2] == ("checkout", "--detach"):
            assert cwd is not None
            for relative_path, content in self._files.items():
                destination = cwd / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        if arguments == ("rev-parse", "HEAD"):
            return self._commit
        return ""


def _source() -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "source_id": "official_world",
            "title": "Official world",
            "repository_url": "https://github.com/example/official-world.git",
            "paper_url": "https://example.org/paper",
            "commit": COMMIT,
            "license_id": "UNDECLARED",
            "code_usage": "DIRECT_OFFICIAL_CODE_AND_DATA",
            "authorization_policy": "USER_AUTHORIZED_NO_LICENSE_BLOCK",
            "authorization_id": "stage1b-v2-user-direct-official-use-2026-08-26",
            "assets": [
                {
                    "asset_id": "generator",
                    "relative_path": "src/generator.py",
                    "sha256": hashlib.sha256(OFFICIAL_BYTES).hexdigest(),
                    "required_for": ["REPRODUCTION", "ORACLE"],
                }
            ],
            "evidence_files": [
                {
                    "url": "https://raw.githubusercontent.com/example/official-world/"
                    f"{COMMIT}/src/generator.py",
                    "sha256": hashlib.sha256(OFFICIAL_BYTES).hexdigest(),
                }
            ],
        }
    )


def test_materializer_checks_out_exact_commit_and_verifies_assets(tmp_path: Path) -> None:
    runner = FakeGit({"src/generator.py": OFFICIAL_BYTES, "README.md": b"official\n"})

    receipt = materialize_source(_source(), tmp_path, runner)

    assert receipt.commit == COMMIT
    assert receipt.authorization_id == "stage1b-v2-user-direct-official-use-2026-08-26"
    assert receipt.checkout_root == tmp_path.resolve() / "official_world" / COMMIT
    assert len(receipt.tree_sha256) == 64
    assert receipt.asset_sha256 == (
        ("src/generator.py", hashlib.sha256(OFFICIAL_BYTES).hexdigest()),
    )
    assert verify_materialized_source(receipt, tmp_path) == receipt.checkout_root
    assert any(call[0] == ("rev-parse", "HEAD") for call in runner.calls)
    assert any(call[0] == ("config", "core.autocrlf", "false") for call in runner.calls)
    assert any(call[0] == ("config", "http.version", "HTTP/1.1") for call in runner.calls)
    assert any(call[0] == ("config", "core.eol", "lf") for call in runner.calls)


def test_materializer_retries_two_transient_fetch_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FlakyFetchGit(FakeGit):
        def __init__(self) -> None:
            super().__init__({"src/generator.py": OFFICIAL_BYTES})
            self.fetch_failures = 0

        def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
            if arguments and arguments[0] == "fetch" and self.fetch_failures < 2:
                self.fetch_failures += 1
                raise SourceVerificationError("git command failed: transient TLS error")
            return super().run(arguments, cwd)

    monkeypatch.setattr("tarca.stage1b.sources.time.sleep", lambda _seconds: None)
    runner = FlakyFetchGit()

    receipt = materialize_source(_source(), tmp_path, runner)

    assert receipt.commit == COMMIT
    assert runner.fetch_failures == 2


def test_offline_capsule_mode_requires_an_existing_verified_checkout_without_git_calls(
    tmp_path: Path,
) -> None:
    runner = FakeGit({"src/generator.py": OFFICIAL_BYTES})

    with pytest.raises(SourceVerificationError, match="offline source cache is missing"):
        materialize_source(
            _source(),
            tmp_path,
            runner,
            mode=SourceAcquisitionMode.OFFLINE_CAPSULE,
        )

    assert runner.calls == []


def test_offline_capsule_mode_revalidates_existing_checkout_without_fetching(
    tmp_path: Path,
) -> None:
    runner = FakeGit({"src/generator.py": OFFICIAL_BYTES})
    expected = materialize_source(_source(), tmp_path, runner)
    runner.calls.clear()

    receipt = materialize_source(
        _source(),
        tmp_path,
        runner,
        mode=SourceAcquisitionMode.OFFLINE_CAPSULE,
    )

    assert receipt.checkout_root == expected.checkout_root
    assert all(call[0][0] not in {"fetch", "remote", "init"} for call in runner.calls)


def test_source_acquisition_mode_defaults_to_online_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TARCA_STAGE1B_SOURCE_MODE", raising=False)
    assert source_acquisition_mode_from_environment() is SourceAcquisitionMode.ONLINE

    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "offline-capsule")
    assert source_acquisition_mode_from_environment() is SourceAcquisitionMode.OFFLINE_CAPSULE

    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "allow-any-source")
    with pytest.raises(ValueError, match="TARCA_STAGE1B_SOURCE_MODE"):
        source_acquisition_mode_from_environment()


def test_source_cache_root_uses_an_explicit_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", raising=False)
    assert source_cache_root_from_environment(tmp_path) == (tmp_path / "third_party/stage1b")

    override = tmp_path / "offline-source-cache"
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", str(override))
    assert source_cache_root_from_environment(tmp_path) == override.resolve()


def test_materializer_rejects_checkout_hash_drift(tmp_path: Path) -> None:
    receipt = materialize_source(
        _source(),
        tmp_path,
        FakeGit({"src/generator.py": OFFICIAL_BYTES}),
    )
    (receipt.checkout_root / "src/generator.py").write_text("changed", encoding="utf-8")

    with pytest.raises(SourceVerificationError, match="hash"):
        verify_materialized_source(receipt, tmp_path)


def test_materializer_rejects_untracked_checkout_drift(tmp_path: Path) -> None:
    source = _source()
    materialize_source(
        source,
        tmp_path,
        FakeGit({"src/generator.py": OFFICIAL_BYTES}),
    )

    class DirtyGit(FakeGit):
        def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
            if arguments == (
                "status",
                "--porcelain",
                "--ignored=matching",
                "--untracked-files=all",
            ):
                return "?? src/__pycache__/generator.pyc"
            return super().run(arguments, cwd)

    with pytest.raises(SourceVerificationError, match="working tree drift"):
        materialize_source(
            source,
            tmp_path,
            DirtyGit({"src/generator.py": OFFICIAL_BYTES}),
        )


def test_materializer_rejects_wrong_checkout_commit(tmp_path: Path) -> None:
    with pytest.raises(SourceVerificationError, match="commit"):
        materialize_source(
            _source(),
            tmp_path,
            FakeGit({"src/generator.py": OFFICIAL_BYTES}, commit="b" * 40),
        )


def test_failed_materialization_removes_read_only_git_objects(tmp_path: Path) -> None:
    class ReadOnlyGit(FakeGit):
        def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
            result = super().run(arguments, cwd)
            if arguments[:2] == ("checkout", "--detach"):
                assert cwd is not None
                read_only = cwd / ".git" / "objects" / "read-only"
                read_only.parent.mkdir(parents=True, exist_ok=True)
                read_only.write_bytes(b"git object")
                read_only.chmod(stat.S_IREAD)
            return result

    with pytest.raises(SourceVerificationError, match="commit"):
        materialize_source(
            _source(),
            tmp_path,
            ReadOnlyGit({"src/generator.py": OFFICIAL_BYTES}, commit="b" * 40),
        )

    assert tuple((tmp_path / "official_world").glob(".official_world-*")) == ()


def test_verifier_rejects_checkout_outside_cache_root(tmp_path: Path) -> None:
    receipt = materialize_source(
        _source(),
        tmp_path,
        FakeGit({"src/generator.py": OFFICIAL_BYTES}),
    )
    escaped = replace(receipt, checkout_root=tmp_path.parent / "escaped")

    with pytest.raises(SourceVerificationError, match="checkout root"):
        verify_materialized_source(escaped, tmp_path)


def test_materialized_sources_requires_one_source_root(tmp_path: Path) -> None:
    receipt = materialize_source(
        _source(),
        tmp_path,
        FakeGit({"src/generator.py": OFFICIAL_BYTES}),
    )
    sources = MaterializedSources(receipts=(receipt,))

    assert sources.root("official_world") == receipt.checkout_root
    with pytest.raises(KeyError, match="missing"):
        sources.root("missing")


def test_materialize_cli_exposes_only_explicit_source_options() -> None:
    completed = subprocess.run(
        (sys.executable, str(REPOSITORY_ROOT / "scripts/materialize_stage1b_sources.py"), "--help"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--cache-root" in completed.stdout
    assert "--source-id" in completed.stdout
    assert "--command" not in completed.stdout


def test_materialize_cli_honors_the_configured_source_acquisition_mode() -> None:
    script = (REPOSITORY_ROOT / "scripts/materialize_stage1b_sources.py").read_text(
        encoding="utf-8"
    )

    assert "source_acquisition_mode_from_environment" in script
    assert "mode=source_acquisition_mode_from_environment()" in script
