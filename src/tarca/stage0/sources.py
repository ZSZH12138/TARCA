from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired, run
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from tarca.stage0.executables import resolve_external_executable

ResolutionStatus = Literal["VERIFIED", "NETWORK_ERROR", "NO_REPOSITORY", "INVALID_RESPONSE"]
SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class SourceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    paper_title: str
    paper_url: str
    repository_url: str
    role: str
    license: str
    default_branch: str
    verified_commit: str
    verified_at: str
    local_reference_path: str
    notes: str

    @field_validator("verified_commit")
    @classmethod
    def validate_verified_commit(cls, value: str) -> str:
        if SHA_PATTERN.fullmatch(value) is None:
            raise ValueError("verified_commit must be a 40-character hexadecimal SHA")
        return value

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "repository_url must be a credential-free HTTPS URL without "
                "query parameters or fragments"
            )
        return value

    @field_validator("local_reference_path")
    @classmethod
    def validate_local_reference_path(cls, value: str, info: ValidationInfo) -> str:
        path = Path(value)
        name = info.data.get("name")
        expected_parts = (".cache", "third_party", name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != len(expected_parts)
            or path.parts != expected_parts
        ):
            raise ValueError(
                "local_reference_path must be the relative path .cache/third_party/<name>"
            )
        return value


@dataclass(frozen=True)
class CommitResolution:
    name: str
    repository_url: str
    default_branch: str
    status: ResolutionStatus
    resolved_commit: str
    source: str
    detail: str


def load_sources(path: Path) -> list[SourceEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Manifest must be a YAML list.")
    return [SourceEntry.model_validate(item) for item in raw]


def _failure(
    entry: SourceEntry,
    status: ResolutionStatus,
    source: str,
    detail: str,
) -> CommitResolution:
    return CommitResolution(
        name=entry.name,
        repository_url=entry.repository_url,
        default_branch=entry.default_branch,
        status=status,
        resolved_commit="",
        source=source,
        detail=detail,
    )


def _run_git(
    entry: SourceEntry,
    args: list[str],
    source: str,
    timeout_seconds: int,
) -> CommitResolution | CompletedProcess[str]:
    try:
        command = [
            resolve_external_executable("git", PROJECT_ROOT),
            *args[1:],
        ]
        return run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except TimeoutExpired:
        return _failure(
            entry,
            "NETWORK_ERROR",
            source,
            f"git command timed out after {timeout_seconds} seconds.",
        )
    except OSError as error:
        return _failure(
            entry,
            "NETWORK_ERROR",
            source,
            f"trusted git command could not start: {error}.",
        )


def _local_repository_path(entry: SourceEntry) -> Path | None:
    project_root = PROJECT_ROOT.resolve()
    local_path = (project_root / entry.local_reference_path).resolve(strict=False)
    expected_path = (project_root / ".cache" / "third_party" / entry.name).resolve(strict=False)
    if local_path != expected_path or not local_path.is_relative_to(project_root):
        return None

    git_marker = local_path / ".git"
    if local_path.is_dir() and (git_marker.is_dir() or git_marker.is_file()):
        return local_path
    return None


def resolve_source(entry: SourceEntry, timeout_seconds: int = 20) -> CommitResolution:
    local_path = _local_repository_path(entry)
    if local_path is not None:
        result = _run_git(
            entry,
            ["git", "-C", str(local_path), "rev-parse", "HEAD"],
            "local",
            timeout_seconds,
        )
        if isinstance(result, CommitResolution):
            return result
        if result.returncode == 0:
            commit = result.stdout.strip()
            if SHA_PATTERN.fullmatch(commit) is not None:
                return CommitResolution(
                    name=entry.name,
                    repository_url=entry.repository_url,
                    default_branch=entry.default_branch,
                    status="VERIFIED",
                    resolved_commit=commit,
                    source="local",
                    detail="Resolved from local clone.",
                )
            return _failure(
                entry,
                "INVALID_RESPONSE",
                "local",
                "Local git rev-parse did not return one 40-character hexadecimal SHA.",
            )
        return _failure(
            entry,
            "NETWORK_ERROR",
            "local",
            result.stderr.strip() or "Local git rev-parse failed.",
        )

    result = _run_git(
        entry,
        ["git", "ls-remote", entry.repository_url, f"refs/heads/{entry.default_branch}"],
        "remote",
        timeout_seconds,
    )
    if isinstance(result, CommitResolution):
        return result
    if result.returncode != 0:
        detail = result.stderr.strip() or "git ls-remote failed."
        status: ResolutionStatus = "NETWORK_ERROR"
        if "Repository not found" in detail or "not found" in detail.lower():
            status = "NO_REPOSITORY"
        return _failure(entry, status, "remote", detail)

    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        return _failure(
            entry,
            "INVALID_RESPONSE",
            "remote",
            "git ls-remote returned an unexpected number of lines.",
        )
    fields = lines[0].split()
    expected_ref = f"refs/heads/{entry.default_branch}"
    if len(fields) != 2 or SHA_PATTERN.fullmatch(fields[0]) is None or fields[1] != expected_ref:
        return _failure(
            entry,
            "INVALID_RESPONSE",
            "remote",
            "git ls-remote output did not contain the expected branch and SHA.",
        )
    return CommitResolution(
        name=entry.name,
        repository_url=entry.repository_url,
        default_branch=entry.default_branch,
        status="VERIFIED",
        resolved_commit=fields[0],
        source="remote",
        detail="Resolved via git ls-remote.",
    )
