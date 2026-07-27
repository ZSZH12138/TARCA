"""Build and validate one Stage 1B synthetic dataset."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.data.synthetic._path_safety import (  # noqa: E402
    DirectorySnapshot,
    StagingDirectory,
    capture_directory,
    cleanup_staging_directory,
    create_staging_directory,
    publish_staging_child,
)
from tarca.data.synthetic.dataset_builder import (  # noqa: E402
    SyntheticConfig,
    build_synthetic_dataset,
    load_synthetic_config,
    persist_synthetic_dataset,
)
from tarca.data.synthetic.validation import validate_synthetic_dataset  # noqa: E402


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
    """Resolve a lexical repository-relative path without following reparse points."""

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


def _ensure_safe_parent(target: Path, project_root: Path) -> DirectorySnapshot:
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
    return capture_directory(target.parent, trusted_root=root, label="output parent")


def _with_seed(config: SyntheticConfig, seed: int | None) -> SyntheticConfig:
    if seed is None:
        return config
    return SyntheticConfig.model_validate(
        {**config.model_dump(mode="python"), "root_seed": seed},
        strict=True,
    )


def _require_smoke_bounds(config: SyntheticConfig) -> None:
    if (
        config.name != "synthetic_easy"
        or config.total_steps > 4096
        or config.mc_samples_smoke > 256
        or config.oracle_pairs_smoke > 16
    ):
        raise ValueError(
            "--smoke requires synthetic_easy with total_steps<=4096, "
            "mc_samples_smoke<=256, and oracle_pairs_smoke<=16"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one deterministic Stage 1B synthetic dataset on CPU."
    )
    parser.add_argument("--config", required=True, help="Repository-relative YAML config.")
    parser.add_argument("--output", required=True, help="New repository-relative output directory.")
    parser.add_argument("--seed", type=int, help="Optional deterministic root-seed override.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Require the bounded synthetic_easy smoke profile.",
    )
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
        config = _with_seed(load_synthetic_config(config_path), arguments.seed)
        if arguments.smoke:
            _require_smoke_bounds(config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        _error("INPUT_ERROR", error)
        return 2

    staging_guard: StagingDirectory | None = None
    try:
        dataset = build_synthetic_dataset(config)
        validation = validate_synthetic_dataset(dataset)
        if validation.status != "VALIDATION_PASS":
            raise RuntimeError("in-memory synthetic validation failed")
        output_parent = _ensure_safe_parent(output, project_root)
        staging_guard = create_staging_directory(output, output_parent)
        staged_dataset = staging_guard.path / "dataset"
        persisted = persist_synthetic_dataset(dataset, staged_dataset)
        validation = validate_synthetic_dataset(dataset, persisted=persisted)
        if validation.status != "VALIDATION_PASS":
            raise RuntimeError("persisted synthetic validation failed")
        publish_staging_child(staging_guard, "dataset", output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _error("BUILD_ERROR", error)
        return 1
    finally:
        if staging_guard is not None:
            cleanup_staging_directory(staging_guard)

    print(
        json.dumps(
            {
                "status": "BUILD_PASS",
                "output": output.relative_to(project_root.resolve()).as_posix(),
                "root_seed": config.root_seed,
                "config_hash": validation.config_hash,
                "dataset_hash": validation.data_hash,
                "validation_status": validation.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
