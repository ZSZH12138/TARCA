from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tarca.stage0.contract import freeze_research_contract, verify_research_contract


def test_freeze_and_verify_research_contract(stage0_repo: Path) -> None:
    manifest = freeze_research_contract(
        stage0_repo,
        created_at=datetime(2026, 8, 20, 1, 2, 3, tzinfo=UTC),
    )

    assert manifest.status == "FROZEN"
    assert manifest.related_work_ref.artifact_type == "RELATED_WORK_BUNDLE"
    assert manifest.environment_lock_ref.artifact_type == "ENVIRONMENT_BUNDLE"
    assert (stage0_repo / "artifacts/stage0/related_work_bundle.json").is_file()
    environment_bundle = json.loads(
        (stage0_repo / "artifacts/stage0/environment_bundle.json").read_text(encoding="utf-8")
    )
    assert environment_bundle["pyproject_ref"]["relative_path"] == "pyproject.toml"
    assert environment_bundle["lock_ref"]["relative_path"] == "uv.lock"
    assert environment_bundle["profile_ref"]["relative_path"].endswith("environment_profile.json")
    assert verify_research_contract(manifest, stage0_repo) is None

    serialized = manifest.model_dump_json(indent=2)
    restored = type(manifest).model_validate_json(serialized)
    assert restored == manifest


def test_verify_detects_tampered_artifact(stage0_repo: Path) -> None:
    manifest = freeze_research_contract(stage0_repo)
    path = stage0_repo / "docs/preregistration_v0.md"
    path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_research_contract(manifest, stage0_repo)


def test_verify_detects_missing_artifact(stage0_repo: Path) -> None:
    manifest = freeze_research_contract(stage0_repo)
    (stage0_repo / "docs/terminology.md").unlink()

    with pytest.raises(FileNotFoundError):
        verify_research_contract(manifest, stage0_repo)


def test_related_work_bundle_binds_matrix_and_sources(stage0_repo: Path) -> None:
    manifest = freeze_research_contract(stage0_repo)
    bundle_path = stage0_repo / manifest.related_work_ref.relative_path  # type: ignore[arg-type]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["matrix_ref"]["artifact_type"] == "RELATED_WORK_MATRIX"
    assert bundle["third_party_versions_ref"]["artifact_type"] == "THIRD_PARTY_VERSIONS"


def test_research_contract_public_function_cannot_overwrite(stage0_repo: Path) -> None:
    freeze_research_contract(stage0_repo)

    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_research_contract(stage0_repo)
