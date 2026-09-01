from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e01.v2_carry_forward import verify_e01_b_carry_forward
from tarca.e01.v2_config import load_e01_v2_config

_CONFIG = Path("configs/e01/e01_v2.yaml")
_V1_HISTORY = Path("artifacts/e01/history/e01_v1_history_record.json")
_STAGE1B_ACTIVE = Path("artifacts/stage1b/active.json")
_TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".toml", ".ts", ".tsx"}
_BLOCKED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    "node_modules",
    "__pycache__",
}


def _eligible(path: Path) -> bool:
    return not any(
        part in _BLOCKED_PARTS or part.endswith(".pyc") or part.endswith(".tsbuildinfo")
        for part in path.parts
    )


def _files_below(root: Path, relative: Path) -> Iterable[Path]:
    base = root / relative
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        raise FileNotFoundError(f"required E01-v2 bundle entry is missing: {relative.as_posix()}")
    yield from (path for path in base.rglob("*") if path.is_file() and _eligible(path))


def _bundle_files(repository_root: Path) -> tuple[Path, ...]:
    entries = (
        Path("pyproject.toml"),
        Path("README.md"),
        Path("src"),
        _CONFIG,
        Path("scripts/run_e01_v2.py"),
        Path("deploy/e01/Dockerfile.v2"),
        Path("deploy/e01/compose.e01-v2.yaml"),
        Path("deploy/e01/entrypoint-v2.sh"),
        Path("deploy/e01/server_bootstrap_v2.sh"),
        Path("deploy/e01/server_supervisor_v2.sh"),
        Path("deploy/e01/py310"),
        Path("deploy/stage1b/requirements-server.lock"),
        Path("frontend/stage1b-monitor"),
        _STAGE1B_ACTIVE,
        Path("artifacts/stage1b/frozen/v2"),
        _V1_HISTORY,
        Path("docs/auth/TARCA_项目计划书.md"),
        Path("docs/auth/TARCA_具体实施计划.md"),
        Path("docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"),
        Path("docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md"),
        Path("docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0004.md"),
        Path("docs/research/e01_execution_spec_v2.md"),
        Path("docs/research/e01_server_handoff_v2.md"),
    )
    files = (path for entry in entries for path in _files_below(repository_root, entry))
    return tuple(sorted(set(files), key=lambda path: path.relative_to(repository_root).as_posix()))


def _tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mode = mode
    return info


def _load_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _verify_frozen_inputs(root: Path) -> tuple[str, str, str]:
    config = load_e01_v2_config(root / _CONFIG)
    carry = verify_e01_b_carry_forward(root, config)
    active = _load_mapping(root / _STAGE1B_ACTIVE, "Stage1B active identity")
    if active != {
        "schema_version": "2.0.0",
        "series": "v2",
        "manifest_sha256": config.carry_forward.expected_stage1b_manifest_sha256,
    }:
        raise ValueError("Stage1B active v2 identity drifted")
    return config.scientific_hash(), carry.receipt_sha256, sha256_file(root / _STAGE1B_ACTIVE)


def _verify_prebuilt_frontend(root: Path) -> None:
    assets = tuple((root / "frontend/stage1b-monitor/dist/assets").glob("*.js"))
    if not assets or not any(b"/api/v1/runtime" in path.read_bytes() for path in assets):
        raise ValueError("prebuilt frontend does not load its runtime identity from the API")


def _receipt_path(output: Path) -> Path:
    suffix = ".tar.gz"
    if output.name.endswith(suffix):
        return output.with_name(f"{output.name[: -len(suffix)]}.receipt.json")
    return output.with_name(f"{output.name}.receipt.json")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def build_e01_v2_server_bundle(repository_root: Path, output: Path) -> dict[str, object]:
    root = repository_root.resolve()
    destination = output.resolve()
    scientific_hash, carry_hash, stage1b_active_hash = _verify_frozen_inputs(root)
    _verify_prebuilt_frontend(root)
    files = _bundle_files(root)
    manifest: dict[str, str] = {}
    payloads: list[tuple[str, bytes, int]] = []
    for path in files:
        payload = path.read_bytes()
        if path.suffix.lower() in _TEXT_SUFFIXES:
            if b"-----BEGIN PRIVATE KEY-----" in payload:
                raise ValueError("E01-v2 bundle source contains private key material")
            if b"C:\\Users\\DELL" in payload:
                raise ValueError("E01-v2 bundle source contains an absolute local user path")
        name = (Path("TARCA") / path.relative_to(root)).as_posix()
        manifest[name] = hashlib.sha256(payload).hexdigest()
        payloads.append((name, payload, 0o755 if path.suffix == ".sh" else 0o644))

    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, payload, mode in payloads:
            archive.addfile(_tar_info(name, len(payload), mode), io.BytesIO(payload))
        archive.addfile(
            _tar_info("TARCA/SHA256SUMS.json", len(manifest_payload), 0o644),
            io.BytesIO(manifest_payload),
        )

    bundle_hash = sha256_file(destination)
    Path(f"{destination}.sha256").write_text(f"{bundle_hash}\n", encoding="ascii", newline="\n")
    receipt_payload: dict[str, object] = {
        "schema_version": "tarca-e01-v2-server-bundle-receipt",
        "status": "PASS",
        "bundle_name": destination.name,
        "bundle_sha256": bundle_hash,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "file_count": len(payloads) + 1,
        "scientific_config_sha256": scientific_hash,
        "e01_b_carry_forward_receipt_sha256": carry_hash,
        "stage1b_active_sha256": stage1b_active_hash,
        "formal_tasks_executed": 0,
    }
    receipt = {**receipt_payload, "receipt_sha256": canonical_json_hash(receipt_payload)}
    receipt_path = _receipt_path(destination)
    _write_json(receipt_path, receipt)
    return {**receipt, "bundle": str(destination), "receipt": str(receipt_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic TARCA E01-v2 server bundle"
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/e01/bundle/tarca-e01-server-v2.tar.gz"),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = build_e01_v2_server_bundle(arguments.repository_root, arguments.output)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
