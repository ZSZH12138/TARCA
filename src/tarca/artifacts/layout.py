from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath

_RUN_FILENAMES = (
    "config.yaml",
    "data_manifest.json",
    "environment.txt",
    "git_state.txt",
    "intervention_pairs.parquet",
    "metrics.json",
    "metrics_by_regime.parquet",
    "predictions.parquet",
    "stdout.log",
)


def _validate_logical_id(value: str, name: str) -> None:
    if (
        not value.strip()
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError(f"{name} must be a logical identifier")


def required_run_paths(experiment_id: str, run_id: str) -> tuple[str, ...]:
    _validate_logical_id(experiment_id, "experiment_id")
    _validate_logical_id(run_id, "run_id")
    root = PurePosixPath("artifacts", experiment_id, run_id)
    return tuple(sorted((root / filename).as_posix() for filename in _RUN_FILENAMES))


def _validate_canonical_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or re.match(r"^[A-Za-z]:/", value):
        raise ValueError("run path must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(part.endswith(".tmp") for part in path.parts)
    ):
        raise ValueError("run path must be a canonical POSIX relative path")
    return path


def validate_run_layout(
    relative_paths: Iterable[str],
    experiment_id: str,
    run_id: str,
) -> tuple[str, ...]:
    required = set(required_run_paths(experiment_id, run_id))
    values = tuple(relative_paths)
    if len(set(values)) != len(values):
        raise ValueError("run layout paths must be unique")
    root = PurePosixPath("artifacts", experiment_id, run_id)
    plots_root = root / "plots"
    for value in values:
        path = _validate_canonical_path(value)
        if value not in required and (plots_root not in path.parents or path == plots_root):
            raise ValueError(f"run path is outside the fixed result tree: {value}")
    missing = required - set(values)
    if missing:
        raise ValueError(f"run layout is missing required paths: {sorted(missing)}")
    return tuple(sorted(values))
