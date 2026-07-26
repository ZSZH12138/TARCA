"""Run the fixed CPU-only Stage 1B E01 engineering smoke."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.data.synthetic.dataset_builder import (  # noqa: E402
    SyntheticConfig,
    build_synthetic_dataset,
    load_synthetic_config,
    persist_synthetic_dataset,
)
from tarca.data.synthetic.validation import run_e01_engineering_smoke  # noqa: E402


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & marker)


def resolve_repository_path(
    value: str | os.PathLike[str],
    *,
    project_root: Path = PROJECT_ROOT,
    require_file: bool = False,
) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TypeError("path: expected text")
    if not raw or "\x00" in raw:
        raise ValueError("path: expected non-empty text without NUL")
    relative = Path(raw)
    if relative.is_absolute() or relative.anchor:
        raise ValueError("path: absolute paths are forbidden")
    if not relative.parts or ".." in relative.parts:
        raise ValueError("path: dot-dot and repository-root paths are forbidden")
    if any(":" in part for part in relative.parts):
        raise ValueError("path: alternate data stream syntax is forbidden")
    root = project_root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current /= part
        if os.path.lexists(current) and _is_reparse(current):
            raise ValueError(f"path: symlink, junction, or reparse component: {current}")
    resolved = current.resolve(strict=require_file)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("path: repository escape is forbidden")
    if require_file and not resolved.is_file():
        raise ValueError("path: expected an existing regular file")
    return resolved


def _ensure_safe_parent(target: Path, project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    relative = target.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if os.path.lexists(current):
            if _is_reparse(current) or not current.is_dir():
                raise ValueError(f"output path: unsafe parent component: {current}")
        else:
            current.mkdir()
    if os.path.lexists(target):
        raise ValueError(f"output path: target already exists: {target}")


def _require_easy_smoke(config: SyntheticConfig) -> None:
    if (
        config.name != "synthetic_easy"
        or config.total_steps > 4096
        or config.mc_samples_smoke > 256
        or config.oracle_pairs_smoke > 16
    ):
        raise ValueError(
            "smoke requires synthetic_easy with total_steps<=4096, "
            "mc_samples_smoke<=256, and oracle_pairs_smoke<=16"
        )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _render_markdown(payload: Mapping[str, object]) -> str:
    convergence = payload["convergence"]
    rows = "\n".join(
        f"| {point['mc_samples']} | {point['total_error']:.8f} | "
        f"{point['estimator_variance']:.8f} |"
        for point in convergence
    )
    return (
        "# TARCA Stage 1B E01 Engineering Smoke\n\n"
        f"- Status: `{payload['status']}`\n"
        "- Research status: `ENGINEERING_SMOKE_ONLY`\n"
        f"- Config hash: `{payload['config_hash']}`\n"
        f"- Dataset hash: `{payload['data_hash']}`\n"
        f"- Root seed: `{payload['root_seed']}`\n"
        f"- Runtime seconds: `{payload['runtime_seconds']:.6f}`\n"
        f"- Additional memory estimate bytes: `{payload['additional_memory_estimate_bytes']}`\n"
        f"- Output size bytes: `{payload['output_size_bytes']}`\n"
        f"- GPU used: `{str(payload['gpu_used']).lower()}`\n\n"
        "## Controls\n\n"
        f"- Correct signature distance: `{payload['correct_signature_distance']:.8f}`\n"
        f"- Wrong-delay signature distance: `{payload['wrong_delay_signature_distance']:.8f}`\n"
        f"- Wrong-scale signature distance: `{payload['wrong_scale_signature_distance']:.8f}`\n"
        f"- Random-concept signature distance: "
        f"`{payload['random_concept_signature_distance']:.8f}`\n"
        f"- True/estimated delay: `{payload['true_delay']}` / "
        f"`{payload['estimated_delay']}`\n\n"
        "## MC convergence\n\n"
        "| Samples | Total error | Estimator variance |\n"
        "|---:|---:|---:|\n"
        f"{rows}\n"
    )


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _cleanup_staging(
    staging: Path,
    output: Path,
    identity: tuple[int, int],
) -> None:
    if not os.path.lexists(staging):
        return
    info = staging.lstat()
    if (
        staging.parent == output.parent
        and staging.name.startswith(f".{output.name}.staging-")
        and not _is_reparse(staging)
        and (info.st_dev, info.st_ino) == identity
    ):
        shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Stage 1B synthetic oracle engineering smoke on CPU."
    )
    parser.add_argument("--config", required=True, help="Repository-relative easy YAML config.")
    parser.add_argument("--output", required=True, help="New repository-relative evidence root.")
    return parser


def _error(status: str, error: BaseException) -> None:
    print(
        json.dumps({"status": status, "reason": str(error)}, ensure_ascii=False),
        file=sys.stderr,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config_path = resolve_repository_path(
            arguments.config,
            project_root=project_root,
            require_file=True,
        )
        output = resolve_repository_path(arguments.output, project_root=project_root)
        config = load_synthetic_config(config_path)
        _require_easy_smoke(config)
        _ensure_safe_parent(output, project_root)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _error("INPUT_ERROR", error)
        return 2

    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        try:
            staging = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
            ).resolve()
            staging_info = staging.lstat()
            staging_identity = (staging_info.st_dev, staging_info.st_ino)
            dataset = build_synthetic_dataset(config)
            persisted = persist_synthetic_dataset(dataset, staging / "dataset")
            report = run_e01_engineering_smoke(dataset, persisted=persisted)
            if report.status != "ENGINEERING_SMOKE_PASS":
                raise RuntimeError("engineering smoke validation failed")
            payload = _json_value(report)
            if not isinstance(payload, dict):
                raise TypeError("engineering smoke report: expected a record")
            payload = {**payload, "research_status": "ENGINEERING_SMOKE_ONLY"}
            _write_atomic(
                staging / "e01_engineering_smoke.json",
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            _write_atomic(
                staging / "e01_engineering_smoke.md",
                _render_markdown(payload),
            )
            if os.path.lexists(output):
                raise ValueError(f"output path: target already exists: {output}")
            staging.rename(output)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _error("SMOKE_ERROR", error)
            return 1
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        if staging is not None and staging_identity is not None:
            _cleanup_staging(staging, output, staging_identity)

    print(
        json.dumps(
            {
                "status": report.status,
                "research_status": "ENGINEERING_SMOKE_ONLY",
                "output": output.relative_to(project_root.resolve()).as_posix(),
                "config_hash": report.config_hash,
                "dataset_hash": report.data_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
