"""Resolve external tools without trusting executables stored in the workspace."""

from __future__ import annotations

import shutil
from pathlib import Path


def resolve_external_executable(name: str, workspace_root: Path) -> str:
    """Return an absolute executable path and reject workspace-local candidates."""
    discovered = shutil.which(name)
    if discovered is None:
        raise FileNotFoundError(f"{name} was not found on PATH")

    executable = Path(discovered).resolve(strict=True)
    workspace = workspace_root.resolve(strict=True)
    if executable.is_relative_to(workspace):
        raise PermissionError(f"Refusing workspace-local {name} executable: {executable}")
    if not executable.is_file():
        raise FileNotFoundError(f"{name} did not resolve to a regular file")
    return str(executable)
