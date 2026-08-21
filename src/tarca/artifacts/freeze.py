from __future__ import annotations

from pathlib import Path

_STATIC_FROZEN_PATHS = frozenset(
    {
        "docs/assumption_ledger.md",
        "docs/novelty_claims.md",
        "docs/preregistration_v0.md",
        "docs/related_work_matrix.csv",
        "docs/terminology.md",
        "pyproject.toml",
        "third_party_manifest/sources.yaml",
        "uv.lock",
    }
)


def _existing_files(root: Path, relative_directory: str) -> set[str]:
    directory = root / relative_directory
    if not directory.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file()}


def frozen_relative_paths(repo_root: Path) -> tuple[str, ...]:
    """Return existing authority, research-input, and Stage 0 artifact paths."""
    root = repo_root.resolve()
    frozen = {
        relative_path for relative_path in _STATIC_FROZEN_PATHS if (root / relative_path).is_file()
    }
    frozen.update(_existing_files(root, "docs/auth"))
    frozen.update(_existing_files(root, "artifacts/stage0"))
    return tuple(sorted(frozen))
