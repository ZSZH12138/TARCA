from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tarca.artifacts.freeze import frozen_relative_paths
from tarca.artifacts.layout import required_run_paths, validate_run_layout
from tarca.contracts import ArtifactManifest, ArtifactRef, RunManifest

HASH_A = "a" * 64
HASH_B = "b" * 64
GIT_COMMIT = "c" * 40


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="predictions",
        artifact_type="PREDICTIONS",
        content_hash=HASH_A,
        schema_version="1.0.0",
        relative_path="artifacts/experiment/run/predictions.parquet",
    )


def test_run_and_artifact_manifests_are_strict_and_frozen() -> None:
    run = RunManifest(
        experiment_id="experiment",
        run_id="run",
        config_hash=HASH_A,
        data_hash=HASH_B,
        git_commit=GIT_COMMIT,
        schema_version="1.0.0",
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        status="COMPLETED",
    )
    artifact = ArtifactManifest(
        artifact=_artifact_ref(),
        media_type="application/vnd.apache.parquet",
        serializer_id="pyarrow-parquet-25",
        producer_stage="STAGE_1A",
        producer_task_id="schema-roundtrip",
        scientific_identity_hash=HASH_B,
        dependencies=(),
        size_bytes=128,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert run.git_commit == GIT_COMMIT
    assert artifact.artifact.content_hash == HASH_A
    with pytest.raises(ValidationError):
        RunManifest.model_validate({**run.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError, match="git_commit"):
        RunManifest(**{**run.model_dump(), "git_commit": "not-a-commit"})
    with pytest.raises(ValidationError, match="schema_version"):
        RunManifest(**{**run.model_dump(), "schema_version": "2.0.0"})


def test_run_layout_requires_exact_safe_result_tree() -> None:
    required = required_run_paths("experiment", "run")
    paths = (*required, "artifacts/experiment/run/plots/forecast.png")

    assert validate_run_layout(paths, "experiment", "run") == tuple(sorted(paths))
    with pytest.raises(ValueError, match="missing required"):
        validate_run_layout(required[:-1], "experiment", "run")
    with pytest.raises(ValueError, match="canonical POSIX"):
        validate_run_layout((*required, "artifacts/experiment/run/../secret"), "experiment", "run")


def test_frozen_path_catalog_includes_authority_and_research_inputs(tmp_path: Path) -> None:
    paths = (
        "docs/auth/TARCA_项目计划书.md",
        "docs/preregistration_v0.md",
        "docs/stage1a_scope.md",
        "docs/superpowers/plans/2026-08-21-stage1a-completion.md",
        "pyproject.toml",
        "scripts/check_stage1a.py",
        "src/tarca/contracts/data.py",
        "src/tarca/data/repository.py",
        "tests/stage1a/test_window_batch.py",
        "uv.lock",
        "artifacts/stage0/research_contract_manifest.json",
    )
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    unrelated = tmp_path / "src/tarca/example.py"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("editable", encoding="utf-8")
    cache = tmp_path / "src/tarca/contracts/__pycache__/data.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")

    frozen = frozen_relative_paths(tmp_path)

    assert set(paths) <= set(frozen)
    assert "src/tarca/example.py" not in frozen
    assert "src/tarca/contracts/__pycache__/data.pyc" not in frozen
