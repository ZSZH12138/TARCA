from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarca.contracts import ArtifactRef
from tarca.contracts.stage0 import ArtifactIndex
from tarca.stage0.artifact_store import LocalArtifactStore


def test_local_artifact_store_publishes_and_reloads_typed_contract(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    index = ArtifactIndex(schema_version="1.0.0", artifacts=())

    ref = store.publish_contract(
        index,
        artifact_id="stage0-index",
        artifact_type="ARTIFACT_INDEX",
        relative_path="artifacts/stage0/artifact_index.json",
    )

    assert ref.relative_path == "artifacts/stage0/artifact_index.json"
    assert store.load_contract(ref, ArtifactIndex) == index
    assert not list((tmp_path / "artifacts/stage0").glob("*.tmp"))


def test_local_artifact_store_refuses_unapproved_replacement(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    index = ArtifactIndex(schema_version="1.0.0", artifacts=())
    publish = {
        "artifact_id": "stage0-index",
        "artifact_type": "ARTIFACT_INDEX",
        "relative_path": "artifacts/stage0/artifact_index.json",
    }
    store.publish_contract(index, **publish)

    with pytest.raises(FileExistsError):
        store.publish_contract(index, **publish)


def test_artifact_index_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ArtifactIndex.model_validate(
            {"schema_version": "1.0.0", "artifacts": [], "unexpected": True}
        )


def test_store_rejects_artifact_reference_outside_repository(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    unsafe = ArtifactRef(
        artifact_id="external",
        artifact_type="TEST",
        content_hash="a" * 64,
        schema_version="1.0.0",
        relative_path=None,
    )

    with pytest.raises(ValueError, match="repository path"):
        store.verify_artifact(unsafe)
