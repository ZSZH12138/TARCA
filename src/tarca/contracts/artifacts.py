"""Deterministic artifact paths with fail-closed filesystem resolution."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import ClassVar

from pydantic import field_validator

from .manifests import NonEmptyString, StrictContractModel

_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class ArtifactLayout(StrictContractModel):
    """Immutable logical layout for one run's required artifacts."""

    experiment_id: NonEmptyString
    run_id: NonEmptyString

    _REQUIRED_NAMES: ClassVar[tuple[str, ...]] = (
        "config.yaml",
        "metrics.json",
        "metrics_by_regime.parquet",
        "predictions.parquet",
        "intervention_pairs.parquet",
        "data_manifest.json",
        "environment.txt",
        "git_state.txt",
        "stdout.log",
        "plots",
    )

    @field_validator("experiment_id", "run_id")
    @classmethod
    def _require_safe_identifier(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("must be a safe single path segment")
        if "/" in value or "\\" in value:
            raise ValueError("must not contain a path separator")
        if "\x00" in value or ":" in value or PureWindowsPath(value).drive:
            raise ValueError("must be a safe single path segment")
        return value

    @property
    def relative_run_root(self) -> PurePosixPath:
        """Return the deterministic POSIX-style logical run root."""

        return PurePosixPath("artifacts", self.experiment_id, self.run_id)

    @property
    def required_relative_paths(self) -> tuple[PurePosixPath, ...]:
        """Return every required artifact path, relative to a filesystem root."""

        run_root = self.relative_run_root
        return tuple(run_root / name for name in self._REQUIRED_NAMES)

    def validate_relative_path(self, relative_path: str) -> PurePosixPath:
        """Validate an unmodified relative path as a descendant of this run."""

        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be an unmodified string")
        if not relative_path:
            raise ValueError("relative path must not be empty")
        if "\\" in relative_path:
            raise ValueError("relative path must use POSIX separators")
        if PurePosixPath(relative_path).is_absolute() or PureWindowsPath(relative_path).drive:
            raise ValueError("relative path must not be absolute or drive-qualified")

        segments = relative_path.split("/")
        if any(not segment for segment in segments):
            raise ValueError("relative path must not contain empty segments")
        if any(segment in {".", ".."} for segment in segments):
            raise ValueError("relative path must not contain dot segments")
        if any("\x00" in segment or ":" in segment for segment in segments):
            raise ValueError("relative path contains an unsafe segment")

        candidate = PurePosixPath(*segments)
        run_root = self.relative_run_root
        if candidate.parts[: len(run_root.parts)] != run_root.parts:
            raise ValueError("relative path must belong to this artifact layout")
        return candidate

    def resolve_path(
        self,
        filesystem_root: str | os.PathLike[str],
        relative_path: str,
    ) -> Path:
        """Resolve a validated path below ``filesystem_root`` without creating it."""

        validated_path = self.validate_relative_path(relative_path)
        root = _coerce_filesystem_root(filesystem_root)
        _reject_link_components(root, "filesystem_root")
        if root.exists() and not root.is_dir():
            raise ValueError("filesystem_root must be a directory")

        root_resolved = _resolve_without_creation(root, "filesystem_root")
        candidate = root.joinpath(*validated_path.parts)
        current = root
        for segment in validated_path.parts:
            current = current / segment
            if _is_link_or_reparse_point(current):
                raise ValueError(
                    "resolved path contains an existing symlink, junction, or reparse point"
                )

        resolved = _resolve_without_creation(candidate, "relative path")
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError("resolved path escapes filesystem_root") from error
        return resolved


def _coerce_filesystem_root(filesystem_root: str | os.PathLike[str]) -> Path:
    if isinstance(filesystem_root, str) and not filesystem_root:
        raise ValueError("filesystem_root must not be empty")
    if isinstance(filesystem_root, bytes) or not isinstance(
        filesystem_root,
        (str, os.PathLike),
    ):
        raise TypeError("filesystem_root must be a string or path-like object")
    try:
        return Path(filesystem_root)
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("filesystem_root is not a valid filesystem path") from error


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError(f"unable to inspect path component {path!s}") from error
    file_attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(file_attributes & _REPARSE_POINT_ATTRIBUTE)


def _reject_link_components(path: Path, field_name: str) -> None:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute_path.anchor)
    for segment in absolute_path.parts[1:]:
        current = current / segment
        if _is_link_or_reparse_point(current):
            raise ValueError(
                f"{field_name} contains an existing symlink, junction, or reparse point"
            )


def _resolve_without_creation(path: Path, field_name: str) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{field_name} could not be resolved safely") from error
