from __future__ import annotations

from pathlib import Path

_STATIC_FROZEN_PATHS = frozenset(
    {
        "docs/assumption_ledger.md",
        "docs/novelty_claims.md",
        "docs/preregistration_v0.md",
        "docs/related_work_matrix.csv",
        "docs/stage1a_scope.md",
        "docs/terminology.md",
        "pyproject.toml",
        "scripts/check_stage0.py",
        "scripts/check_stage1a.py",
        "scripts/doctor.py",
        "scripts/run_reference_smoke.py",
        "third_party_manifest/sources.yaml",
        "uv.lock",
    }
)

_FROZEN_PYTHON_DIRECTORIES = (
    "src/tarca/artifacts",
    "src/tarca/contracts",
    "src/tarca/data",
    "src/tarca/stage0",
    "tests/stage0",
    "tests/stage1a",
)


def _existing_files(root: Path, relative_directory: str) -> set[str]:
    directory = root / relative_directory
    if not directory.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file()}


def frozen_relative_paths(repo_root: Path) -> tuple[str, ...]:
    """Return existing authority, evidence, and completed-stage boundary paths."""
    root = repo_root.resolve()
    frozen = {
        relative_path for relative_path in _STATIC_FROZEN_PATHS if (root / relative_path).is_file()
    }
    frozen.update(_existing_files(root, "docs/auth"))
    frozen.update(_existing_files(root, "artifacts/stage0"))
    for relative_directory in _FROZEN_PYTHON_DIRECTORIES:
        frozen.update(
            path for path in _existing_files(root, relative_directory) if path.endswith(".py")
        )
    return tuple(sorted(frozen))
