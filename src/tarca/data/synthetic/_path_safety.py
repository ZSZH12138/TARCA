"""Identity-bound directory guards for synthetic artifact publication."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

DirectoryIdentity = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    """A materialized directory and the identity observed during validation."""

    path: Path
    resolved: Path
    identity: DirectoryIdentity
    trusted_root: Path | None
    trusted_root_resolved: Path | None
    trusted_root_identity: DirectoryIdentity | None
    label: str


@dataclass(frozen=True, slots=True)
class StagingDirectory:
    """A newly created staging directory bound to its validated parent."""

    path: Path
    resolved: Path
    identity: DirectoryIdentity
    parent: DirectorySnapshot


def capture_directory(
    path: Path,
    *,
    trusted_root: Path | None = None,
    label: str = "output parent",
) -> DirectorySnapshot:
    """Validate and snapshot one existing non-reparse directory."""

    lexical = _absolute_lexical(path)
    root = _absolute_lexical(trusted_root) if trusted_root is not None else None
    try:
        _reject_reparse_components(lexical, label)
        resolved = lexical.resolve(strict=True)
        identity = _directory_identity(lexical, label)
        if root is None:
            return DirectorySnapshot(lexical, resolved, identity, None, None, None, label)

        _reject_reparse_components(root, f"{label} trusted root")
        root_resolved = root.resolve(strict=True)
        root_identity = _directory_identity(root, f"{label} trusted root")
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(f"{label}: directory escaped the trusted root")
        return DirectorySnapshot(
            lexical,
            resolved,
            identity,
            root,
            root_resolved,
            root_identity,
            label,
        )
    except OSError as error:
        raise ValueError(f"{label}: expected an existing safe directory") from error


def verify_directory(snapshot: DirectorySnapshot) -> None:
    """Fail if a snapshotted directory or trusted root changed identity."""

    try:
        _reject_reparse_components(snapshot.path, snapshot.label)
        current_resolved = snapshot.path.resolve(strict=True)
        current_identity = _directory_identity(snapshot.path, snapshot.label)
        if current_resolved != snapshot.resolved or current_identity != snapshot.identity:
            raise ValueError(f"{snapshot.label}: parent directory identity changed or was replaced")

        if snapshot.trusted_root is not None:
            _reject_reparse_components(
                snapshot.trusted_root,
                f"{snapshot.label} trusted root",
            )
            root_resolved = snapshot.trusted_root.resolve(strict=True)
            root_identity = _directory_identity(
                snapshot.trusted_root,
                f"{snapshot.label} trusted root",
            )
            if (
                root_resolved != snapshot.trusted_root_resolved
                or root_identity != snapshot.trusted_root_identity
                or not current_resolved.is_relative_to(root_resolved)
            ):
                raise ValueError(f"{snapshot.label}: trusted root identity changed or was replaced")
    except OSError as error:
        raise ValueError(
            f"{snapshot.label}: parent directory identity changed or was replaced"
        ) from error


def create_staging_directory(
    target: Path,
    parent: DirectorySnapshot,
) -> StagingDirectory:
    """Create a sibling staging directory and reject parent swaps around creation."""

    target = _absolute_lexical(target)
    if target.parent != parent.path:
        raise ValueError(f"{parent.label}: target parent does not match the validated directory")
    verify_directory(parent)
    raw_staging = tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent.path)
    staging = _absolute_lexical(Path(raw_staging))
    try:
        verify_directory(parent)
        _reject_reparse_components(staging, "staging directory")
        resolved = staging.resolve(strict=True)
        identity = _directory_identity(staging, "staging directory")
        if staging.parent != parent.path or resolved.parent != parent.resolved:
            raise ValueError("staging directory: escaped the validated output parent")
        return StagingDirectory(staging, resolved, identity, parent)
    except OSError as error:
        raise ValueError("staging directory: unsafe directory created") from error


def verify_staging_directory(staging: StagingDirectory) -> None:
    """Fail if the staging directory or any validated parent changed."""

    verify_directory(staging.parent)
    try:
        _reject_reparse_components(staging.path, "staging directory")
        if (
            staging.path.resolve(strict=True) != staging.resolved
            or _directory_identity(staging.path, "staging directory") != staging.identity
        ):
            raise ValueError("staging directory: identity changed or was replaced")
    except OSError as error:
        raise ValueError("staging directory: identity changed or was replaced") from error


def publish_staging_directory(staging: StagingDirectory, target: Path) -> Path:
    """Atomically publish a guarded staging directory into its validated parent."""

    target = _absolute_lexical(target)
    _require_publish_target(staging, target)
    staging.path.rename(target)
    _verify_published_identity(target, staging.identity, staging.parent)
    return target.resolve(strict=True)


def publish_staging_child(
    staging: StagingDirectory,
    child_name: str,
    target: Path,
) -> Path:
    """Atomically publish one guarded staging child into the validated parent."""

    if not child_name or Path(child_name).name != child_name:
        raise ValueError("staging child: expected one non-empty path component")
    target = _absolute_lexical(target)
    _require_publish_target(staging, target)
    source = staging.path / child_name
    source_identity = _directory_identity(source, "staging child")
    _reject_reparse_components(source, "staging child")
    source.rename(target)
    _verify_published_identity(target, source_identity, staging.parent)
    return target.resolve(strict=True)


def cleanup_staging_directory(staging: StagingDirectory) -> bool:
    """Remove only the exact staging directory under the unchanged parent."""

    try:
        verify_staging_directory(staging)
    except (OSError, ValueError):
        return False
    shutil.rmtree(staging.path)
    return True


def _require_publish_target(staging: StagingDirectory, target: Path) -> None:
    verify_staging_directory(staging)
    if target.parent != staging.parent.path:
        raise ValueError("output path: target parent does not match the validated directory")
    if os.path.lexists(target):
        raise ValueError(f"output path: target already exists: {target}")
    verify_directory(staging.parent)


def _verify_published_identity(
    target: Path,
    expected_identity: DirectoryIdentity,
    parent: DirectorySnapshot,
) -> None:
    verify_directory(parent)
    _reject_reparse_components(target, "published directory")
    if _directory_identity(target, "published directory") != expected_identity:
        raise ValueError("published directory: identity changed during publication")


def _absolute_lexical(path: Path | None) -> Path:
    if path is None:
        raise TypeError("path: expected a filesystem path")
    return Path(os.path.abspath(os.fspath(path)))


def _directory_identity(path: Path, label: str) -> DirectoryIdentity:
    info = path.lstat()
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & marker)
    ):
        raise ValueError(f"{label}: expected a non-reparse directory")
    return info.st_dev, info.st_ino


def _reject_reparse_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current):
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & marker):
                raise ValueError(f"{label}: symlink, junction, or reparse component: {current}")
