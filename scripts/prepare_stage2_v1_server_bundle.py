from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e02.config import load_e02_config
from tarca.stage2.config import load_stage2_config
from tarca.stage2.recovery import load_stage2_recovery_spec

_SOURCE_CAPSULE = "stage2-v1-official-sources.tar.gz"
_FORBIDDEN_BYTES = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN RSA " + b"PRIVATE KEY-----",
    b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
    b"-----BEGIN EC " + b"PRIVATE KEY-----",
    b"C:" + b"\\Users" + b"\\DELL",
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?i)(api[_-]?key|password|secret|access[_-]?token)\s*=\s*[^\s$<{][^\s]{7,}"
)
_FORMAL_ARTIFACT_MARKERS = (
    "formal_prediction",
    "formal-prediction",
    "formal_score",
    "formal-score",
)


def _is_bundle_file(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and "node_modules" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _selected_files(root: Path) -> tuple[Path, ...]:
    directories = (
        "src",
        "configs",
        "deploy/stage2",
        "frontend/stage1b-monitor/dist",
        "tests/stage2",
        "tests/e02",
        "docs/auth",
    )
    files = [
        root / "README.md",
        root / "pyproject.toml",
        root / "scripts/run_stage2_v1.py",
        root / "scripts/run_e02_v1.py",
        root / "scripts/prepare_stage2_v1_server_bundle.py",
        root / "scripts/import_stage2_source_capsule.py",
        root / "scripts/materialize_stage2_sources.py",
        root / "scripts/package_stage2_source_capsule.py",
        root / "docs/superpowers/specs/2026-08-31-stage2-e02-local-runtime-design.md",
        root / "artifacts/stage1b/frozen/v2/manifest.json",
        root / "artifacts/stage1b/frozen/v2/manifest.sha256",
        root / "artifacts/stage1b/frozen/v2/qualification_receipt.json",
        root / "artifacts/e01/frozen/v2/qualification_receipt.json",
        root / f"artifacts/stage2/source-capsules/{_SOURCE_CAPSULE}",
        root / f"artifacts/stage2/source-capsules/{_SOURCE_CAPSULE}.receipt.json",
    ]
    for directory in directories:
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"bundle directory is missing: {directory}")
        files.extend(path for path in base.rglob("*") if _is_bundle_file(path))
    for relative in (
        "docs/research/stage2_e02_local_implementation_report_v1.md",
        "docs/research/stage2_e02_server_handoff_v1.md",
        "docs/research/stage2_device_mismatch_recovery_v1.md",
    ):
        document = root / relative
        if document.is_file():
            files.append(document)
    unique = {path.resolve() for path in files}
    return tuple(sorted(unique, key=lambda path: path.relative_to(root).as_posix()))


def _validate_file(root: Path, path: Path, payload: bytes) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("bundle input escapes repository") from error
    if path.is_symlink():
        raise ValueError(f"bundle rejects symlink: {relative}")
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError("bundle path is not canonical")
    lowered = relative.lower()
    if lowered.startswith("third_party/"):
        raise ValueError("bundle rejects mutable source checkouts")
    if lowered.startswith("artifacts/") and any(
        marker in lowered for marker in _FORMAL_ARTIFACT_MARKERS
    ):
        raise ValueError("bundle rejects formal prediction or score artifacts")
    if any(marker in payload for marker in _FORBIDDEN_BYTES):
        raise ValueError(f"bundle input contains a secret or local absolute path: {relative}")
    if _CREDENTIAL_ASSIGNMENT.search(payload):
        raise ValueError(f"bundle input contains a credential-like assignment: {relative}")
    return relative


def _tar_bytes(files: tuple[tuple[str, bytes, int], ...]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload, mode in files:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, mtime=0, compresslevel=9
    ) as stream:
        stream.write(raw.getvalue())
    return compressed.getvalue()


def _verify_prebuilt_frontend(root: Path) -> None:
    assets = tuple((root / "frontend/stage1b-monitor/dist/assets").glob("*.js"))
    required_markers = (
        b"/api/v1/runtime",
        "服务器预检给出的保守估计".encode(),
        b"eta_source",
    )
    if not assets or any(
        not any(marker in path.read_bytes() for path in assets)
        for marker in required_markers
    ):
        raise ValueError("prebuilt Stage 2 frontend is missing or stale")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".stage2-bundle-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_stage2_server_bundle(
    repository_root: Path,
    output: Path,
    *,
    recovery_archive: Path | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve()
    stage2 = load_stage2_config(root / "configs/stage2/stage2_v1.yaml")
    e02 = load_e02_config(root / "configs/e02/e02_v1.yaml")
    expected_stage2 = "8a0509edfd1487dc36188e8d12ca088d52f0287804f4808215ff0f7c279c069f"
    if stage2.scientific_hash() != expected_stage2:
        raise ValueError("Stage 2 scientific config hash drifted")
    if e02.scientific_hash() != "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c":
        raise ValueError("E02 scientific config hash drifted")
    _verify_prebuilt_frontend(root)
    recovery_spec = load_stage2_recovery_spec(
        root / "configs/stage2/stage2_device_mismatch_recovery_v1.json"
    )
    resolved_recovery: Path | None = None
    if recovery_archive is not None:
        resolved_recovery = recovery_archive.resolve()
        if resolved_recovery.name != recovery_spec.source_archive_filename:
            raise ValueError("recovery archive filename does not match the frozen spec")
        if (
            not resolved_recovery.is_file()
            or sha256_file(resolved_recovery) != recovery_spec.source_archive_sha256
        ):
            raise ValueError("recovery archive SHA-256 does not match the frozen spec")
    selected: list[tuple[str, bytes, int]] = []
    sums: dict[str, str] = {}
    for path in _selected_files(root):
        payload = path.read_bytes()
        relative = _validate_file(root, path, payload)
        mode = 0o755 if relative.endswith(".sh") else 0o644
        selected.append((relative, payload, mode))
        sums[relative] = hashlib.sha256(payload).hexdigest()
    manifest_payload = (
        canonical_json_bytes({"schema_version": "tarca-stage2-bundle-v1", "files": sums}) + b"\n"
    )
    selected.append(("SHA256SUMS.json", manifest_payload, 0o644))
    archive_payload = _tar_bytes(tuple(sorted(selected, key=lambda item: item[0])))
    output_path = output.resolve()
    _atomic_bytes(output_path, archive_payload)
    bundle_sha256 = hashlib.sha256(archive_payload).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": "tarca-stage2-server-bundle-v1",
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": len(archive_payload),
        "file_count": len(selected),
        "stage2_scientific_config_sha256": stage2.scientific_hash(),
        "e02_scientific_config_sha256": e02.scientific_hash(),
        "stage1b_manifest_sha256": stage2.upstream.stage1b_manifest_sha256,
        "e01_receipt_sha256": stage2.upstream.e01_receipt_sha256,
        "source_capsule_sha256": sha256_file(
            root / f"artifacts/stage2/source-capsules/{_SOURCE_CAPSULE}"
        ),
        "formal_tasks_executed": 0,
        "recovery_mode": (
            "DEVICE_MISMATCH_V1" if resolved_recovery is not None else "NONE"
        ),
    }
    if resolved_recovery is not None:
        receipt.update(
            {
                "recovery_archive_filename": resolved_recovery.name,
                "recovery_archive_sha256": recovery_spec.source_archive_sha256,
                "recovery_manifest_sha256": recovery_spec.source_manifest_sha256,
                "recovery_spec_sha256": canonical_json_hash(
                    recovery_spec.model_dump(mode="json")
                ),
            }
        )
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    _atomic_bytes(
        output_path.with_name(output_path.name + ".sha256"),
        f"{bundle_sha256}  {output_path.name}\n".encode(),
    )
    _atomic_bytes(
        output_path.with_name(output_path.name + ".receipt.json"),
        canonical_json_bytes(receipt) + b"\n",
    )
    return receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build deterministic TARCA Stage 2 server bundle")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-archive", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_stage2_server_bundle(
                arguments.repository_root,
                arguments.output,
                recovery_archive=arguments.recovery_archive,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
