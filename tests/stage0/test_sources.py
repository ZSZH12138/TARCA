from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarca.stage0.sources import (
    audit_dependency_bindings,
    dependency_release_map,
    load_sources_manifest,
    verify_remote_release_bindings,
)


def test_load_sources_manifest(stage0_repo: Path) -> None:
    manifest = load_sources_manifest(stage0_repo / "third_party_manifest/sources.yaml")

    assert manifest.schema_version == "1.0.0"
    assert manifest.sources[0].commit == "a" * 40
    assert manifest.sources[0].allowed_action == "REFERENCE_ONLY"


def test_unknown_license_cannot_be_dependency(stage0_repo: Path) -> None:
    path = stage0_repo / "third_party_manifest/sources.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("REFERENCE_ONLY", "DEPENDENCY"))

    with pytest.raises(ValidationError, match="UNKNOWN license"):
        load_sources_manifest(path)


def test_source_manifest_rejects_floating_commit(stage0_repo: Path) -> None:
    path = stage0_repo / "third_party_manifest/sources.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("a" * 40, "main"))

    with pytest.raises(ValidationError):
        load_sources_manifest(path)


def test_dependency_requires_package_release_mapping(stage0_repo: Path) -> None:
    path = stage0_repo / "third_party_manifest/sources.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "license_status: UNKNOWN\n    license_file: NONE_FOUND\n    allowed_action: REFERENCE_ONLY",
        "license_status: MIT\n    license_file: LICENSE\n    allowed_action: DEPENDENCY",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValidationError, match="release mapping"):
        load_sources_manifest(path)


def test_dependency_release_map_is_machine_readable(stage0_repo: Path) -> None:
    path = stage0_repo / "third_party_manifest/sources.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "license_status: UNKNOWN\n    license_file: NONE_FOUND\n    allowed_action: REFERENCE_ONLY",
        "license_status: MIT\n    license_file: LICENSE\n    allowed_action: DEPENDENCY\n"
        "    package_name: plot-package\n    package_version: 1.2.3\n"
        "    release_tag: v1.2.3\n"
        "    release_commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    path.write_text(text, encoding="utf-8")

    manifest = load_sources_manifest(path)
    assert dependency_release_map(manifest) == {
        "plot-package": {
            "package_version": "1.2.3",
            "release_tag": "v1.2.3",
            "release_commit": "b" * 40,
            "source_id": "plot",
        }
    }


def test_dependency_release_is_bound_to_pyproject_and_lock(stage0_repo: Path) -> None:
    source_path = stage0_repo / "third_party_manifest/sources.yaml"
    source_text = source_path.read_text(encoding="utf-8").replace(
        "license_status: UNKNOWN\n    license_file: NONE_FOUND\n    allowed_action: REFERENCE_ONLY",
        "license_status: MIT\n    license_file: LICENSE\n    allowed_action: DEPENDENCY\n"
        "    package_name: plot-package\n    package_version: 1.2.3\n"
        "    release_tag: v1.2.3\n"
        "    release_commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    source_path.write_text(source_text, encoding="utf-8")
    (stage0_repo / "pyproject.toml").write_text(
        '[project]\nname = "tarca-test"\n'
        '[project.optional-dependencies]\nresearch = ["plot-package==1.2.3"]\n',
        encoding="utf-8",
    )
    (stage0_repo / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "plot-package"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )

    manifest = load_sources_manifest(source_path)
    summary = audit_dependency_bindings(stage0_repo, manifest)

    assert summary["dependency_release_count"] == 1
    assert summary["locked_dependency_count"] == 1


def test_remote_release_binding_rejects_tag_commit_mismatch(stage0_repo: Path) -> None:
    source_path = stage0_repo / "third_party_manifest/sources.yaml"
    source_text = source_path.read_text(encoding="utf-8").replace(
        "license_status: UNKNOWN\n    license_file: NONE_FOUND\n    allowed_action: REFERENCE_ONLY",
        "license_status: MIT\n    license_file: LICENSE\n    allowed_action: DEPENDENCY\n"
        "    package_name: plot-package\n    package_version: 1.2.3\n"
        "    release_tag: v1.2.3\n"
        "    release_commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    source_path.write_text(source_text, encoding="utf-8")
    manifest = load_sources_manifest(source_path)

    with pytest.raises(ValueError, match="does not resolve"):
        verify_remote_release_bindings(manifest, lambda _source: "c" * 40)
