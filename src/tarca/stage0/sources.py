from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import field_validator, model_validator

from tarca.contracts import GitCommit, StrictContractModel

AllowedAction = Literal["DEPENDENCY", "REFERENCE_ONLY", "STATIC_ONLY"]


class ThirdPartySource(StrictContractModel):
    source_id: str
    repository_url: str
    paper_url: str | None
    role: str
    license_status: str
    license_file: str
    allowed_action: AllowedAction
    default_branch: str
    commit: GitCommit
    package_name: str | None = None
    package_version: str | None = None
    release_tag: str | None = None
    release_commit: GitCommit | None = None
    verification_date: date
    local_reference_path: str | None

    @field_validator("repository_url")
    @classmethod
    def _official_github_url(cls, value: str) -> str:
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise ValueError("repository_url must be a canonical GitHub repository URL")
        return value

    @field_validator("local_reference_path")
    @classmethod
    def _safe_local_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("local_reference_path must stay inside the repository")
        return value

    @field_validator("package_name", "package_version", "release_tag")
    @classmethod
    def _optional_release_value_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("release mapping values must not be blank")
        return value

    @model_validator(mode="after")
    def _unknown_license_is_reference_only(self) -> ThirdPartySource:
        if self.license_status == "UNKNOWN" and self.allowed_action == "DEPENDENCY":
            raise ValueError("UNKNOWN license cannot be used as a DEPENDENCY")
        if self.license_status != "UNKNOWN" and self.license_file == "NONE_FOUND":
            raise ValueError("known license_status requires a license_file")
        release_values = (
            self.package_name,
            self.package_version,
            self.release_tag,
            self.release_commit,
        )
        if self.allowed_action == "DEPENDENCY" and any(value is None for value in release_values):
            raise ValueError("DEPENDENCY source requires a complete package release mapping")
        if self.allowed_action != "DEPENDENCY" and any(
            value is not None for value in release_values
        ):
            raise ValueError("package release mapping is only valid for DEPENDENCY sources")
        return self


class ThirdPartySourceManifest(StrictContractModel):
    schema_version: str
    verification_date: date
    sources: tuple[ThirdPartySource, ...]

    @model_validator(mode="after")
    def _source_ids_are_unique(self) -> ThirdPartySourceManifest:
        source_ids = [item.source_id for item in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        package_names = [
            item.package_name for item in self.sources if item.package_name is not None
        ]
        if len(package_names) != len(set(package_names)):
            raise ValueError("package_name values must be unique")
        return self


def load_sources_manifest(path: Path) -> ThirdPartySourceManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("sources manifest must be a mapping with a sources list")
    normalized_sources = tuple(
        {
            **item,
            "verification_date": date.fromisoformat(item["verification_date"]),
        }
        for item in payload["sources"]
    )
    normalized = {
        **payload,
        "verification_date": date.fromisoformat(payload["verification_date"]),
        "sources": normalized_sources,
    }
    return ThirdPartySourceManifest.model_validate(normalized)


def dependency_release_map(manifest: ThirdPartySourceManifest) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for source in manifest.sources:
        if source.allowed_action != "DEPENDENCY":
            continue
        if (
            source.package_name is None
            or source.package_version is None
            or source.release_tag is None
            or source.release_commit is None
        ):  # pragma: no cover - enforced by ThirdPartySource validation
            raise ValueError(f"incomplete dependency release mapping: {source.source_id}")
        mapping[source.package_name] = {
            "package_version": source.package_version,
            "release_tag": source.release_tag,
            "release_commit": source.release_commit,
            "source_id": source.source_id,
        }
    return mapping


def _normalized_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _exact_requirement_versions(requirements: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for requirement in requirements:
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^;\s]+)", requirement)
        if match:
            versions[_normalized_package_name(match.group(1))] = match.group(2)
    return versions


def audit_dependency_bindings(
    repo_root: Path,
    manifest: ThirdPartySourceManifest,
) -> dict[str, int]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    research_requirements = (
        pyproject.get("project", {}).get("optional-dependencies", {}).get("research", [])
    )
    if not isinstance(research_requirements, list) or not all(
        isinstance(item, str) for item in research_requirements
    ):
        raise ValueError("project.optional-dependencies.research must be a string list")
    exact_versions = _exact_requirement_versions(research_requirements)

    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    locked_versions: dict[str, str] = {}
    for package in lock.get("package", []):
        if isinstance(package, dict) and isinstance(package.get("name"), str):
            version = package.get("version")
            if isinstance(version, str):
                locked_versions[_normalized_package_name(package["name"])] = version

    release_mapping = dependency_release_map(manifest)
    for package_name, release in release_mapping.items():
        normalized = _normalized_package_name(package_name)
        expected = release["package_version"]
        if exact_versions.get(normalized) != expected:
            raise ValueError(
                f"pyproject research dependency is not exactly bound to {package_name}=={expected}"
            )
        if locked_versions.get(normalized) != expected:
            raise ValueError(f"uv.lock is not bound to {package_name}=={expected}")

    return {
        "dependency_release_count": len(release_mapping),
        "locked_dependency_count": len(release_mapping),
    }


def verify_remote_release_bindings(
    manifest: ThirdPartySourceManifest,
    resolve_tag_commit: Callable[[ThirdPartySource], str],
) -> int:
    verified = 0
    for source in manifest.sources:
        if source.allowed_action != "DEPENDENCY":
            continue
        if source.release_commit is None:  # pragma: no cover - schema validation enforces this
            raise ValueError(f"dependency has no release commit: {source.source_id}")
        resolved = resolve_tag_commit(source)
        if resolved != source.release_commit:
            raise ValueError(
                f"release tag for {source.source_id} does not resolve to declared commit"
            )
        verified += 1
    return verified
