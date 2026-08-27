from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from tarca.contracts import canonical_json_bytes, canonical_json_hash
from tarca.stage1b.config import SourceConfig
from tarca.stage1b.sources import (
    GitRunner,
    SourceAcquisitionMode,
    SourceMaterializationReceipt,
    SourceVerificationError,
    materialize_source,
)

_SCHEMA_VERSION = "2.0.0"
_MANIFEST_NAME = "manifest.json"
_BUNDLE_DIRECTORY = "bundles"
_MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024**3
_MAX_MANIFEST_BYTES = 1024 * 1024


class SourceCapsuleVerificationError(RuntimeError):
    """Raised when an offline official-source capsule cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SourceCapsuleSource:
    source_id: str
    repository_url: str
    commit: str
    authorization_id: str
    tree_sha256: str
    asset_sha256: tuple[tuple[str, str], ...]
    bundle_path: str
    bundle_sha256: str


@dataclass(frozen=True, slots=True)
class SourceCapsuleReceipt:
    schema_version: str
    capsule_sha256: str
    manifest_sha256: str
    sources: tuple[SourceCapsuleSource, ...]


def source_capsule_receipt_path(capsule_path: Path) -> Path:
    """Return the mandatory sidecar receipt location for a capsule archive."""

    return capsule_path.with_name(f"{capsule_path.name}.receipt.json")


def source_capsule_import_receipt_path(cache_root: Path) -> Path:
    """Return the cache-local proof that its checkouts came from an audited capsule."""

    return cache_root / "source-capsule-import-receipt-v2.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_payload(source: SourceCapsuleSource) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "repository_url": source.repository_url,
        "commit": source.commit,
        "authorization_id": source.authorization_id,
        "tree_sha256": source.tree_sha256,
        "asset_sha256": [list(item) for item in source.asset_sha256],
        "bundle_path": source.bundle_path,
        "bundle_sha256": source.bundle_sha256,
    }


def source_capsule_receipt_payload(receipt: SourceCapsuleReceipt) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "capsule_sha256": receipt.capsule_sha256,
        "manifest_sha256": receipt.manifest_sha256,
        "sources": [_source_payload(source) for source in receipt.sources],
    }


def write_source_capsule_receipt(receipt: SourceCapsuleReceipt, path: Path) -> None:
    """Atomically persist the external SHA-256 receipt used by the server importer."""

    _atomic_write_bytes(path, canonical_json_bytes(source_capsule_receipt_payload(receipt)) + b"\n")


def _require_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SourceCapsuleVerificationError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceCapsuleVerificationError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    digest = _require_string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SourceCapsuleVerificationError(f"{label} must be a lowercase SHA-256")
    return digest


def _asset_hashes_from_payload(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise SourceCapsuleVerificationError(f"{label} must be a nonempty array")
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise SourceCapsuleVerificationError(f"{label}[{index}] must contain path and SHA-256")
        path = _require_string(item[0], f"{label}[{index}][0]")
        digest = _require_sha256(item[1], f"{label}[{index}][1]")
        result.append((path, digest))
    if len(result) != len(set(result)):
        raise SourceCapsuleVerificationError(f"{label} contains duplicate entries")
    return tuple(result)


def _source_from_payload(value: object, label: str) -> SourceCapsuleSource:
    payload = _require_object(value, label)
    source_id = _require_string(payload.get("source_id"), f"{label}.source_id")
    bundle_path = _require_string(payload.get("bundle_path"), f"{label}.bundle_path")
    expected_bundle_path = f"{_BUNDLE_DIRECTORY}/{source_id}.bundle"
    if bundle_path != expected_bundle_path:
        raise SourceCapsuleVerificationError(
            f"{label}.bundle_path must equal {expected_bundle_path!r}"
        )
    return SourceCapsuleSource(
        source_id=source_id,
        repository_url=_require_string(payload.get("repository_url"), f"{label}.repository_url"),
        commit=_require_string(payload.get("commit"), f"{label}.commit"),
        authorization_id=_require_string(
            payload.get("authorization_id"), f"{label}.authorization_id"
        ),
        tree_sha256=_require_sha256(payload.get("tree_sha256"), f"{label}.tree_sha256"),
        asset_sha256=_asset_hashes_from_payload(
            payload.get("asset_sha256"), f"{label}.asset_sha256"
        ),
        bundle_path=bundle_path,
        bundle_sha256=_require_sha256(payload.get("bundle_sha256"), f"{label}.bundle_sha256"),
    )


def _sources_from_payload(value: object, label: str) -> tuple[SourceCapsuleSource, ...]:
    if not isinstance(value, list) or not value:
        raise SourceCapsuleVerificationError(f"{label} must be a nonempty array")
    sources = tuple(
        _source_from_payload(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    source_ids = tuple(source.source_id for source in sources)
    if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(set(source_ids)):
        raise SourceCapsuleVerificationError(f"{label} source IDs must be unique and sorted")
    return sources


def read_source_capsule_receipt(path: Path) -> SourceCapsuleReceipt:
    """Load and validate a sidecar receipt without trusting any archive content."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceCapsuleVerificationError(f"cannot read capsule receipt: {path}") from error
    payload = _require_object(raw, "capsule receipt")
    schema_version = _require_string(
        payload.get("schema_version"), "capsule receipt.schema_version"
    )
    if schema_version != _SCHEMA_VERSION:
        raise SourceCapsuleVerificationError("capsule receipt schema version is unsupported")
    return SourceCapsuleReceipt(
        schema_version=schema_version,
        capsule_sha256=_require_sha256(
            payload.get("capsule_sha256"), "capsule receipt.capsule_sha256"
        ),
        manifest_sha256=_require_sha256(
            payload.get("manifest_sha256"), "capsule receipt.manifest_sha256"
        ),
        sources=_sources_from_payload(payload.get("sources"), "capsule receipt.sources"),
    )


def _source_configurations(sources: Iterable[SourceConfig]) -> tuple[SourceConfig, ...]:
    sorted_sources = tuple(sorted(sources, key=lambda source: source.source_id))
    source_ids = tuple(source.source_id for source in sorted_sources)
    if not sorted_sources or len(source_ids) != len(set(source_ids)):
        raise ValueError("source capsule requires one or more uniquely identified sources")
    return sorted_sources


def _source_entry(
    source: SourceConfig,
    receipt: SourceMaterializationReceipt,
    bundle_sha256: str,
) -> SourceCapsuleSource:
    return SourceCapsuleSource(
        source_id=receipt.source_id,
        repository_url=receipt.repository_url,
        commit=receipt.commit,
        authorization_id=receipt.authorization_id,
        tree_sha256=receipt.tree_sha256,
        asset_sha256=receipt.asset_sha256,
        bundle_path=f"{_BUNDLE_DIRECTORY}/{source.source_id}.bundle",
        bundle_sha256=bundle_sha256,
    )


def _manifest_payload(sources: tuple[SourceCapsuleSource, ...]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "sources": [_source_payload(source) for source in sources],
    }


def _add_regular_file(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = archive.gettarinfo(str(path), arcname=arcname)
    if not info.isreg():
        raise SourceCapsuleVerificationError(f"capsule member is not a regular file: {arcname}")
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def _write_capsule_archive(
    output_path: Path,
    manifest_path: Path,
    bundle_paths: tuple[tuple[str, Path], ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as raw_stream,
            gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as gzip_stream,
            tarfile.open(fileobj=gzip_stream, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            _add_regular_file(archive, manifest_path, _MANIFEST_NAME)
            for bundle_name, bundle_path in bundle_paths:
                _add_regular_file(archive, bundle_path, bundle_name)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def build_source_capsule(
    sources: Iterable[SourceConfig],
    cache_root: Path,
    output_path: Path,
    runner: GitRunner,
) -> SourceCapsuleReceipt:
    """Build a locally audited Git-bundle archive and its SHA-256 sidecar receipt."""

    selected = _source_configurations(sources)
    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{resolved_output.name}.build-", dir=resolved_output.parent)
    )
    try:
        bundle_root = temporary_root / _BUNDLE_DIRECTORY
        bundle_root.mkdir()
        entries: list[SourceCapsuleSource] = []
        bundles: list[tuple[str, Path]] = []
        for source in selected:
            source_receipt = materialize_source(
                source,
                cache_root,
                runner,
                mode=SourceAcquisitionMode.ONLINE,
            )
            bundle_name = f"{_BUNDLE_DIRECTORY}/{source.source_id}.bundle"
            bundle_path = temporary_root / bundle_name
            runner.run(
                ("bundle", "create", str(bundle_path), "HEAD"),
                cwd=source_receipt.checkout_root,
            )
            if not bundle_path.is_file():
                raise SourceCapsuleVerificationError(
                    f"git did not create source bundle for {source.source_id}"
                )
            entries.append(_source_entry(source, source_receipt, _sha256_file(bundle_path)))
            bundles.append((bundle_name, bundle_path))
        capsule_sources = tuple(entries)
        manifest = _manifest_payload(capsule_sources)
        manifest_path = temporary_root / _MANIFEST_NAME
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        _write_capsule_archive(resolved_output, manifest_path, tuple(bundles))
        capsule_receipt = SourceCapsuleReceipt(
            schema_version=_SCHEMA_VERSION,
            capsule_sha256=_sha256_file(resolved_output),
            manifest_sha256=canonical_json_hash(manifest),
            sources=capsule_sources,
        )
        write_source_capsule_receipt(
            capsule_receipt,
            source_capsule_receipt_path(resolved_output),
        )
        return capsule_receipt
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _safe_archive_name(raw_name: str) -> str:
    if not raw_name or "\\" in raw_name:
        raise SourceCapsuleVerificationError("capsule contains an unsafe archive member name")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceCapsuleVerificationError("capsule contains an unsafe archive member name")
    normalized = path.as_posix()
    if normalized != raw_name:
        raise SourceCapsuleVerificationError("capsule contains a noncanonical archive member name")
    return normalized


def _archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = _safe_archive_name(member.name)
        if not member.isfile() or member.size < 0 or member.size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise SourceCapsuleVerificationError(f"capsule member is invalid: {name}")
        if name in members:
            raise SourceCapsuleVerificationError(f"capsule contains duplicate member: {name}")
        members[name] = member
    return members


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo, label: str) -> bytes:
    stream = archive.extractfile(member)
    if stream is None:
        raise SourceCapsuleVerificationError(f"cannot read capsule member: {label}")
    with stream:
        return stream.read()


def _manifest_from_archive(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    receipt: SourceCapsuleReceipt,
) -> tuple[SourceCapsuleSource, ...]:
    manifest_member = members.get(_MANIFEST_NAME)
    if manifest_member is None:
        raise SourceCapsuleVerificationError("capsule manifest is missing")
    if manifest_member.size > _MAX_MANIFEST_BYTES:
        raise SourceCapsuleVerificationError("capsule manifest exceeds the size limit")
    try:
        raw_manifest = json.loads(_read_member(archive, manifest_member, _MANIFEST_NAME))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceCapsuleVerificationError("capsule manifest is not valid JSON") from error
    manifest = _require_object(raw_manifest, "capsule manifest")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise SourceCapsuleVerificationError("capsule manifest schema version is unsupported")
    if canonical_json_hash(manifest) != receipt.manifest_sha256:
        raise SourceCapsuleVerificationError("capsule manifest SHA-256 does not match its receipt")
    sources = _sources_from_payload(manifest.get("sources"), "capsule manifest.sources")
    if sources != receipt.sources:
        raise SourceCapsuleVerificationError("capsule manifest sources do not match its receipt")
    expected_members = {_MANIFEST_NAME, *(source.bundle_path for source in sources)}
    if set(members) != expected_members:
        raise SourceCapsuleVerificationError(
            "capsule contains unexpected or missing archive members"
        )
    return sources


def _validate_source_identity(source: SourceConfig, entry: SourceCapsuleSource) -> None:
    expected_assets = tuple((asset.relative_path, asset.sha256) for asset in source.assets)
    expected = (
        source.source_id,
        source.repository_url,
        source.commit,
        source.authorization_id,
        expected_assets,
    )
    actual = (
        entry.source_id,
        entry.repository_url,
        entry.commit,
        entry.authorization_id,
        entry.asset_sha256,
    )
    if actual != expected:
        raise SourceCapsuleVerificationError(
            f"capsule source identity does not match registered source {source.source_id}"
        )


def _validate_receipt_sources(
    sources: tuple[SourceConfig, ...],
    receipt: SourceCapsuleReceipt,
) -> dict[str, SourceConfig]:
    configured_by_id = {source.source_id: source for source in sources}
    if tuple(source.source_id for source in receipt.sources) != tuple(configured_by_id):
        raise SourceCapsuleVerificationError(
            "capsule source set does not match the registered sources"
        )
    for entry in receipt.sources:
        _validate_source_identity(configured_by_id[entry.source_id], entry)
    return configured_by_id


def verify_source_capsule_import(
    sources: Iterable[SourceConfig],
    cache_root: Path,
) -> SourceCapsuleReceipt:
    """Verify that a cache is bound to a capsule matching the registered sources."""

    selected = _source_configurations(sources)
    receipt = read_source_capsule_receipt(source_capsule_import_receipt_path(cache_root.resolve()))
    _validate_receipt_sources(selected, receipt)
    return receipt


def _copy_bundle_from_archive(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = archive.extractfile(member)
    if stream is None:
        raise SourceCapsuleVerificationError(f"cannot read capsule bundle: {member.name}")
    with stream, destination.open("wb") as target:
        shutil.copyfileobj(stream, target, length=1024 * 1024)
    actual_sha256 = _sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise SourceCapsuleVerificationError(
            f"capsule bundle SHA-256 mismatch for {member.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _checkout_bundle(
    source: SourceConfig,
    bundle_path: Path,
    staging_cache: Path,
    runner: GitRunner,
) -> SourceMaterializationReceipt:
    checkout = staging_cache / source.source_id / source.commit
    checkout.mkdir(parents=True)
    try:
        runner.run(("init", "--quiet"), cwd=checkout)
        runner.run(("config", "core.autocrlf", "false"), cwd=checkout)
        runner.run(("config", "core.eol", "lf"), cwd=checkout)
        # Local audited checkouts are intentionally depth-one.  Their bundles contain the
        # pinned commit, tree and blobs but not an unreachable historical parent; declaring
        # the commit as a shallow boundary lets Git import that exact self-contained snapshot.
        (checkout / ".git" / "shallow").write_text(f"{source.commit}\n", encoding="ascii")
        runner.run(("fetch", "--no-tags", str(bundle_path), source.commit), cwd=checkout)
        runner.run(("checkout", "--detach", "FETCH_HEAD"), cwd=checkout)
        receipt = materialize_source(
            source,
            staging_cache,
            runner,
            mode=SourceAcquisitionMode.OFFLINE_CAPSULE,
        )
    except SourceVerificationError as error:
        raise SourceCapsuleVerificationError(
            f"cannot reconstruct verified checkout for {source.source_id} from local bundle"
        ) from error
    return receipt


def _validate_reconstructed_receipt(
    receipt: SourceMaterializationReceipt,
    entry: SourceCapsuleSource,
) -> None:
    if receipt.tree_sha256 != entry.tree_sha256 or receipt.asset_sha256 != entry.asset_sha256:
        raise SourceCapsuleVerificationError(
            f"reconstructed source tree does not match capsule manifest: {receipt.source_id}"
        )


def _publish_checkouts(
    receipts: tuple[SourceMaterializationReceipt, ...],
    staging_cache: Path,
    cache_root: Path,
    runner: GitRunner,
    source_configurations: Mapping[str, SourceConfig],
) -> None:
    for receipt in receipts:
        target = cache_root / receipt.source_id / receipt.commit
        if not target.exists():
            continue
        try:
            existing = materialize_source(
                source_configurations[receipt.source_id],
                cache_root,
                runner,
                mode=SourceAcquisitionMode.OFFLINE_CAPSULE,
            )
        except (SourceVerificationError, ValueError) as error:
            raise SourceCapsuleVerificationError(
                f"existing source cache is not safe to retain: {receipt.source_id}"
            ) from error
        if (
            existing.tree_sha256 != receipt.tree_sha256
            or existing.asset_sha256 != receipt.asset_sha256
        ):
            raise SourceCapsuleVerificationError(
                f"existing source cache identity conflicts with capsule: {receipt.source_id}"
            )
    for receipt in receipts:
        target = cache_root / receipt.source_id / receipt.commit
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = staging_cache / receipt.source_id / receipt.commit
        os.replace(staged, target)


def import_source_capsule(
    sources: Iterable[SourceConfig],
    capsule_path: Path,
    receipt_path: Path,
    cache_root: Path,
    runner: GitRunner,
) -> tuple[SourceMaterializationReceipt, ...]:
    """Import a capsule using only local Git bundles, then publish verified checkouts."""

    selected = _source_configurations(sources)
    receipt = read_source_capsule_receipt(receipt_path)
    resolved_capsule = capsule_path.resolve()
    if not resolved_capsule.is_file():
        raise SourceCapsuleVerificationError(f"capsule archive is missing: {resolved_capsule}")
    actual_capsule_sha256 = _sha256_file(resolved_capsule)
    if actual_capsule_sha256 != receipt.capsule_sha256:
        raise SourceCapsuleVerificationError(
            "capsule SHA-256 does not match its receipt: "
            f"expected {receipt.capsule_sha256}, got {actual_capsule_sha256}"
        )
    configured_by_id = _validate_receipt_sources(selected, receipt)

    resolved_cache = cache_root.resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".i-", dir=resolved_cache))
    try:
        staging_cache = temporary_root
        extracted = temporary_root / "x"
        extracted.mkdir()
        with tarfile.open(resolved_capsule, "r:gz") as archive:
            members = _archive_members(archive)
            entries = _manifest_from_archive(archive, members, receipt)
            staged_receipts: list[SourceMaterializationReceipt] = []
            for entry in entries:
                source = configured_by_id[entry.source_id]
                _validate_source_identity(source, entry)
                bundle_path = extracted / entry.bundle_path
                _copy_bundle_from_archive(
                    archive,
                    members[entry.bundle_path],
                    bundle_path,
                    entry.bundle_sha256,
                )
                staged_receipt = _checkout_bundle(source, bundle_path, staging_cache, runner)
                _validate_reconstructed_receipt(staged_receipt, entry)
                staged_receipts.append(staged_receipt)
        receipts = tuple(staged_receipts)
        _publish_checkouts(
            receipts,
            staging_cache,
            resolved_cache,
            runner,
            configured_by_id,
        )
        write_source_capsule_receipt(
            receipt,
            source_capsule_import_receipt_path(resolved_cache),
        )
        return tuple(
            materialize_source(
                configured_by_id[receipt.source_id],
                resolved_cache,
                runner,
                mode=SourceAcquisitionMode.OFFLINE_CAPSULE,
            )
            for receipt in receipts
        )
    except (OSError, tarfile.TarError) as error:
        raise SourceCapsuleVerificationError(
            "cannot read or import source capsule archive"
        ) from error
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
