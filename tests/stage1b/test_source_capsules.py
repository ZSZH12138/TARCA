from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from tarca.stage1b.config import SourceConfig
from tarca.stage1b.source_capsules import (
    SourceCapsuleVerificationError,
    build_source_capsule,
    import_source_capsule,
    read_source_capsule_receipt,
    source_capsule_import_receipt_path,
    source_capsule_receipt_path,
    verify_source_capsule_import,
    write_source_capsule_receipt,
)
from tarca.stage1b.sources import SubprocessGitRunner, verify_materialized_source

OFFICIAL_BYTES = b"def generate():\n    return 'official'\n"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _source(commit: str) -> SourceConfig:
    asset_sha256 = hashlib.sha256(OFFICIAL_BYTES).hexdigest()
    return SourceConfig.model_validate(
        {
            "source_id": "official_world",
            "title": "Official world",
            "repository_url": "https://github.com/example/official-world.git",
            "paper_url": "https://example.org/paper",
            "commit": commit,
            "license_id": "UNDECLARED",
            "code_usage": "DIRECT_OFFICIAL_CODE_AND_DATA",
            "authorization_policy": "USER_AUTHORIZED_NO_LICENSE_BLOCK",
            "authorization_id": "stage1b-v2-user-direct-official-use-2026-08-26",
            "assets": [
                {
                    "asset_id": "generator",
                    "relative_path": "src/generator.py",
                    "sha256": asset_sha256,
                    "required_for": ["REPRODUCTION", "ORACLE"],
                }
            ],
            "evidence_files": [
                {
                    "url": "https://raw.githubusercontent.com/example/official-world/"
                    f"{commit}/src/generator.py",
                    "sha256": asset_sha256,
                }
            ],
        }
    )


def _prepared_source_cache(tmp_path: Path) -> tuple[SourceConfig, Path]:
    if shutil.which("git") is None:
        pytest.skip("git is required for source capsule tests")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet")
    _git(upstream, "config", "user.email", "tarca-tests@example.invalid")
    _git(upstream, "config", "user.name", "TARCA tests")
    (upstream / "README.md").write_text("historical parent\n", encoding="utf-8")
    _git(upstream, "add", "README.md")
    _git(upstream, "commit", "--quiet", "-m", "historical parent")
    source_file = upstream / "src/generator.py"
    source_file.parent.mkdir()
    source_file.write_bytes(OFFICIAL_BYTES)
    _git(upstream, "add", "src/generator.py")
    _git(upstream, "commit", "--quiet", "-m", "official source")
    commit = _git(upstream, "rev-parse", "HEAD")

    source = _source(commit)
    cache_root = tmp_path / "verified-source-cache"
    checkout = cache_root / source.source_id / source.commit
    checkout.parent.mkdir(parents=True)
    _git(
        tmp_path,
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.eol=lf",
        "clone",
        "--quiet",
        "--depth=1",
        "--no-local",
        upstream.as_uri(),
        str(checkout),
    )
    _git(checkout, "config", "core.autocrlf", "false")
    _git(checkout, "config", "core.eol", "lf")
    _git(checkout, "checkout", "--quiet", "--detach", source.commit)
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "true"
    return source, cache_root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tamper_bundle(capsule: Path, temporary_root: Path) -> None:
    unpacked = temporary_root / "unpacked"
    with tarfile.open(capsule, "r:gz") as archive:
        archive.extractall(unpacked)
    bundle = next((unpacked / "bundles").glob("*.bundle"))
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    with tarfile.open(capsule, "w:gz") as archive:
        for path in sorted(unpacked.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(unpacked).as_posix())


def _append_path_traversal_member(capsule: Path, temporary_root: Path) -> None:
    unpacked = temporary_root / "unpacked"
    with tarfile.open(capsule, "r:gz") as archive:
        archive.extractall(unpacked)
    with tarfile.open(capsule, "w:gz") as archive:
        for path in sorted(unpacked.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(unpacked).as_posix())
        escaped = tarfile.TarInfo("../escaped.txt")
        escaped.size = len(b"unsafe")
        archive.addfile(escaped, io.BytesIO(b"unsafe"))


def _replace_manifest_with_oversized_member(capsule: Path) -> None:
    replacement = capsule.with_suffix(".replacement.tar.gz")
    oversized = b" " * (1024 * 1024 + 1)
    with (
        tarfile.open(capsule, "r:gz") as source,
        tarfile.open(replacement, "w:gz") as destination,
    ):
        for member in source.getmembers():
            if member.name == "manifest.json":
                manifest = tarfile.TarInfo("manifest.json")
                manifest.size = len(oversized)
                destination.addfile(manifest, io.BytesIO(oversized))
                continue
            stream = source.extractfile(member)
            assert stream is not None
            with stream:
                destination.addfile(member, stream)
    replacement.replace(capsule)


def test_source_capsule_round_trip_imports_verified_git_checkout(tmp_path: Path) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()

    receipt = build_source_capsule((source,), source_cache, capsule, runner)
    imported = import_source_capsule(
        (source,),
        capsule,
        source_capsule_receipt_path(capsule),
        tmp_path / "server-source-cache",
        runner,
    )

    assert receipt.capsule_sha256 == _sha256_file(capsule)
    assert tuple(item.source_id for item in imported) == (source.source_id,)
    assert verify_materialized_source(imported[0], tmp_path / "server-source-cache") == (
        tmp_path / "server-source-cache" / source.source_id / source.commit
    )
    assert _git(imported[0].checkout_root, "rev-parse", "--is-shallow-repository") == "true"
    assert (
        read_source_capsule_receipt(
            source_capsule_import_receipt_path(tmp_path / "server-source-cache")
        )
        == receipt
    )
    assert verify_source_capsule_import((source,), tmp_path / "server-source-cache") == receipt


def test_source_capsule_retains_an_identical_existing_verified_checkout(tmp_path: Path) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()
    build_source_capsule((source,), source_cache, capsule, runner)
    existing = source_cache / source.source_id / source.commit

    imported = import_source_capsule(
        (source,),
        capsule,
        source_capsule_receipt_path(capsule),
        source_cache,
        runner,
    )

    assert imported[0].checkout_root == existing
    assert verify_materialized_source(imported[0], source_cache) == existing


def test_source_capsule_build_creates_a_missing_output_directory(tmp_path: Path) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "new" / "capsules" / "source-capsule.tar.gz"

    receipt = build_source_capsule((source,), source_cache, capsule, SubprocessGitRunner.discover())

    assert capsule.is_file()
    assert source_capsule_receipt_path(capsule).is_file()
    assert receipt.capsule_sha256 == _sha256_file(capsule)


def test_source_capsule_rejects_a_non_utf8_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "invalid.receipt.json"
    receipt_path.write_bytes(b"\xff\xfe")

    with pytest.raises(SourceCapsuleVerificationError, match="cannot read capsule receipt"):
        read_source_capsule_receipt(receipt_path)


def test_source_capsule_rejects_a_tampered_outer_archive_before_cache_mutation(
    tmp_path: Path,
) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()
    build_source_capsule((source,), source_cache, capsule, runner)
    capsule.write_bytes(capsule.read_bytes() + b"tampered")
    destination = tmp_path / "server-source-cache"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(SourceCapsuleVerificationError, match="capsule SHA-256"):
        import_source_capsule(
            (source,),
            capsule,
            source_capsule_receipt_path(capsule),
            destination,
            runner,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (destination / source.source_id / source.commit).exists()


def test_source_capsule_rejects_a_tampered_bundle_after_outer_hash_is_reauthorized(
    tmp_path: Path,
) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()
    receipt = build_source_capsule((source,), source_cache, capsule, runner)
    _tamper_bundle(capsule, tmp_path / "tamper")
    write_source_capsule_receipt(
        replace(receipt, capsule_sha256=_sha256_file(capsule)),
        source_capsule_receipt_path(capsule),
    )
    destination = tmp_path / "server-source-cache"

    with pytest.raises(SourceCapsuleVerificationError, match="bundle SHA-256"):
        import_source_capsule(
            (source,),
            capsule,
            source_capsule_receipt_path(capsule),
            destination,
            runner,
        )

    assert not (destination / source.source_id / source.commit).exists()


def test_source_capsule_rejects_a_path_traversal_member_without_writing_outside_cache(
    tmp_path: Path,
) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()
    receipt = build_source_capsule((source,), source_cache, capsule, runner)
    _append_path_traversal_member(capsule, tmp_path / "traversal")
    write_source_capsule_receipt(
        replace(receipt, capsule_sha256=_sha256_file(capsule)),
        source_capsule_receipt_path(capsule),
    )
    destination = tmp_path / "server-source-cache"

    with pytest.raises(SourceCapsuleVerificationError, match="unsafe archive member"):
        import_source_capsule(
            (source,),
            capsule,
            source_capsule_receipt_path(capsule),
            destination,
            runner,
        )

    assert not (tmp_path / "escaped.txt").exists()
    assert not (destination / source.source_id / source.commit).exists()


def test_source_capsule_rejects_an_oversized_manifest_before_cache_mutation(
    tmp_path: Path,
) -> None:
    source, source_cache = _prepared_source_cache(tmp_path)
    capsule = tmp_path / "source-capsule.tar.gz"
    runner = SubprocessGitRunner.discover()
    receipt = build_source_capsule((source,), source_cache, capsule, runner)
    _replace_manifest_with_oversized_member(capsule)
    write_source_capsule_receipt(
        replace(receipt, capsule_sha256=_sha256_file(capsule)),
        source_capsule_receipt_path(capsule),
    )
    destination = tmp_path / "server-source-cache"

    with pytest.raises(SourceCapsuleVerificationError, match="manifest exceeds the size limit"):
        import_source_capsule(
            (source,),
            capsule,
            source_capsule_receipt_path(capsule),
            destination,
            runner,
        )

    assert not (destination / source.source_id / source.commit).exists()


@pytest.mark.parametrize(
    ("script_name", "required_options"),
    (
        ("package_stage1b_source_capsule.py", ("--config", "--cache-root", "--output")),
        (
            "import_stage1b_source_capsule.py",
            ("--config", "--cache-root", "--capsule", "--receipt"),
        ),
    ),
)
def test_source_capsule_clis_expose_only_explicit_transfer_options(
    script_name: str,
    required_options: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        (sys.executable, str(REPOSITORY_ROOT / "scripts" / script_name), "--help"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert all(option in completed.stdout for option in required_options)
    assert "--command" not in completed.stdout
