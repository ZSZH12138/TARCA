from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tarca.stage1b.config import SourceConfig


class SourceVerificationError(RuntimeError):
    """Raised when official source identity or bytes cannot be verified."""


class GitRunner(Protocol):
    def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str: ...


@dataclass(frozen=True, slots=True)
class SubprocessGitRunner:
    executable: str

    @classmethod
    def discover(cls) -> SubprocessGitRunner:
        executable = shutil.which("git")
        if executable is None:
            raise SourceVerificationError("git executable is unavailable")
        return cls(executable=executable)

    def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
        completed = subprocess.run(
            (self.executable, *arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            shell=False,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-4000:]
            raise SourceVerificationError(
                f"git command failed with exit code {completed.returncode}: {detail}"
            )
        return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class SourceMaterializationReceipt:
    source_id: str
    repository_url: str
    commit: str
    checkout_root: Path
    tree_sha256: str
    asset_sha256: tuple[tuple[str, str], ...]
    authorization_id: str
    materialized_at_utc: datetime


@dataclass(frozen=True, slots=True)
class MaterializedSources:
    receipts: tuple[SourceMaterializationReceipt, ...]

    @classmethod
    def empty(cls) -> MaterializedSources:
        return cls(receipts=())

    def root(self, source_id: str) -> Path:
        matches = tuple(
            receipt.checkout_root for receipt in self.receipts if receipt.source_id == source_id
        )
        if len(matches) != 1:
            raise KeyError(source_id)
        return matches[0]


def _safe_resolve_below(root: Path, candidate: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise SourceVerificationError(f"{label} escapes the source cache root") from error
    return resolved_candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(checkout_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for candidate in checkout_root.rglob("*"):
        relative = candidate.relative_to(checkout_root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if candidate.is_symlink():
            raise SourceVerificationError(
                f"official source contains unsupported symlink: {relative.as_posix()}"
            )
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise SourceVerificationError("official source checkout contains no files")
    return tuple(sorted(files, key=lambda path: path.relative_to(checkout_root).as_posix()))


def _tree_sha256(checkout_root: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_files(checkout_root):
        relative = path.relative_to(checkout_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    candidates = (path, *path.rglob("*"))
    for candidate in candidates:
        if not candidate.is_symlink():
            candidate.chmod(candidate.stat().st_mode | stat.S_IWRITE)
    shutil.rmtree(path)


def _verified_asset_hashes(
    source: SourceConfig,
    checkout_root: Path,
) -> tuple[tuple[str, str], ...]:
    verified: list[tuple[str, str]] = []
    for asset in source.assets:
        asset_path = _safe_resolve_below(
            checkout_root,
            checkout_root / asset.relative_path,
            "asset path",
        )
        if not asset_path.is_file():
            raise SourceVerificationError(
                f"required official asset is missing: {asset.relative_path}"
            )
        actual = _sha256_file(asset_path)
        if actual != asset.sha256:
            raise SourceVerificationError(
                f"official asset hash mismatch for {asset.relative_path}: "
                f"expected {asset.sha256}, got {actual}"
            )
        verified.append((asset.relative_path, actual))
    return tuple(verified)


def _receipt_for_checkout(
    source: SourceConfig,
    checkout_root: Path,
) -> SourceMaterializationReceipt:
    assets = _verified_asset_hashes(source, checkout_root)
    return SourceMaterializationReceipt(
        source_id=source.source_id,
        repository_url=source.repository_url,
        commit=source.commit,
        checkout_root=checkout_root,
        tree_sha256=_tree_sha256(checkout_root),
        asset_sha256=assets,
        authorization_id=source.authorization_id,
        materialized_at_utc=datetime.now(UTC),
    )


def _verify_git_commit(runner: GitRunner, checkout_root: Path, expected_commit: str) -> None:
    actual_commit = runner.run(("rev-parse", "HEAD"), cwd=checkout_root)
    if actual_commit != expected_commit:
        raise SourceVerificationError(
            f"official source commit mismatch: expected {expected_commit}, got {actual_commit}"
        )


def materialize_source(
    source: SourceConfig,
    cache_root: Path,
    runner: GitRunner,
) -> SourceMaterializationReceipt:
    resolved_cache = cache_root.resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    source_root = _safe_resolve_below(
        resolved_cache,
        resolved_cache / source.source_id,
        "source root",
    )
    source_root.mkdir(parents=True, exist_ok=True)
    checkout_root = _safe_resolve_below(
        resolved_cache,
        source_root / source.commit,
        "checkout root",
    )

    if checkout_root.exists():
        _verify_git_commit(runner, checkout_root, source.commit)
        receipt = _receipt_for_checkout(source, checkout_root)
        verify_materialized_source(receipt, resolved_cache)
        return receipt

    temporary = Path(tempfile.mkdtemp(prefix=f".{source.source_id}-", dir=source_root))
    published = False
    try:
        runner.run(("init", "--quiet"), cwd=temporary)
        runner.run(("config", "core.autocrlf", "false"), cwd=temporary)
        runner.run(("config", "core.eol", "lf"), cwd=temporary)
        runner.run(("config", "http.version", "HTTP/1.1"), cwd=temporary)
        runner.run(("remote", "add", "origin", source.repository_url), cwd=temporary)
        runner.run(
            ("fetch", "--depth=1", "--no-tags", "origin", source.commit),
            cwd=temporary,
        )
        runner.run(("checkout", "--detach", "FETCH_HEAD"), cwd=temporary)
        _verify_git_commit(runner, temporary, source.commit)
        _receipt_for_checkout(source, temporary)
        os.replace(temporary, checkout_root)
        published = True
    finally:
        if not published and temporary.exists():
            _remove_tree(temporary)

    receipt = _receipt_for_checkout(source, checkout_root)
    verify_materialized_source(receipt, resolved_cache)
    return receipt


def verify_materialized_source(
    receipt: SourceMaterializationReceipt,
    cache_root: Path,
) -> Path:
    resolved_cache = cache_root.resolve()
    expected_root = _safe_resolve_below(
        resolved_cache,
        resolved_cache / receipt.source_id / receipt.commit,
        "checkout root",
    )
    actual_root = receipt.checkout_root.resolve()
    if actual_root != expected_root:
        raise SourceVerificationError("checkout root does not match source ID and commit")
    if not actual_root.is_dir():
        raise SourceVerificationError("checkout root is missing")
    for relative_path, expected_hash in receipt.asset_sha256:
        asset_path = _safe_resolve_below(
            actual_root,
            actual_root / relative_path,
            "asset path",
        )
        if not asset_path.is_file() or _sha256_file(asset_path) != expected_hash:
            raise SourceVerificationError(
                f"official asset hash mismatch for {relative_path}"
            )
    actual_tree_hash = _tree_sha256(actual_root)
    if actual_tree_hash != receipt.tree_sha256:
        raise SourceVerificationError(
            f"official source tree hash mismatch: expected {receipt.tree_sha256}, "
            f"got {actual_tree_hash}"
        )
    return actual_root
