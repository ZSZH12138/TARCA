from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

import tarca.artifacts.store as store_module
from tarca.artifacts.store import LocalArtifactStore
from tarca.contracts import (
    ArtifactManifest,
    StrictContractModel,
    canonical_json_bytes,
    sha256_file,
)
from tarca.contracts.arrow_schemas import PREDICTIONS_SCHEMA


class ExampleContract(StrictContractModel):
    name: str


def _store(repo_root: Path) -> LocalArtifactStore:
    return LocalArtifactStore(
        repo_root,
        producer_stage="STAGE1A",
        producer_task_id="stage1a-contract-test",
        scientific_identity_hash="a" * 64,
    )


def _prediction_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=PREDICTIONS_SCHEMA)


def test_store_publishes_and_reloads_typed_artifacts_with_manifest(tmp_path: Path) -> None:
    store = _store(tmp_path)

    contract_ref = store.publish_contract(ExampleContract(name="alpha"), "EXAMPLE")
    arrow_ref = store.publish_arrow(_prediction_table(), PREDICTIONS_SCHEMA, "PREDICTIONS")
    bytes_ref = store.publish_bytes(b"payload", "BINARY", "application/octet-stream", "1.0.0")
    text_ref = store.publish_text("报告\n", "REPORT", "text/plain", "1.0.0")

    assert store.load_contract(contract_ref, ExampleContract) == ExampleContract(name="alpha")
    assert store.load_arrow(arrow_ref, PREDICTIONS_SCHEMA).schema == PREDICTIONS_SCHEMA
    assert store.load_bytes(bytes_ref) == b"payload"
    assert store.load_bytes(text_ref).decode("utf-8") == "报告\n"
    for ref in (contract_ref, arrow_ref, bytes_ref, text_ref):
        assert store.verify_artifact(ref)
        assert ref.relative_path is not None
        artifact_path = tmp_path / ref.relative_path
        assert sha256_file(artifact_path) == ref.content_hash
        manifest_path = artifact_path.with_name(f"{artifact_path.name}.manifest.json")
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
        assert manifest.artifact == ref
        assert manifest.size_bytes == artifact_path.stat().st_size


def test_schema_failure_does_not_publish_or_return_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    drifted_schema = PREDICTIONS_SCHEMA.set(7, pa.field("scale", pa.float32()))
    drifted_table = pa.Table.from_pylist([], schema=drifted_schema)

    with pytest.raises(ValueError, match="schema mismatch"):
        store.publish_arrow(drifted_table, PREDICTIONS_SCHEMA, "PREDICTIONS")

    assert not list((tmp_path / "artifacts").rglob("*"))


def test_validation_failure_before_publish_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)

    def fail_reload(*_args: object, **_kwargs: object) -> None:
        raise ValueError("injected reload failure")

    monkeypatch.setattr(store, "_validate_contract_file", fail_reload)
    with pytest.raises(ValueError, match="injected reload failure"):
        store.publish_contract(ExampleContract(name="alpha"), "EXAMPLE")

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_store_rejects_path_escape_and_existing_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    value = ExampleContract(name="alpha")

    with pytest.raises(ValueError, match="artifact_type"):
        store.publish_contract(value, "../ESCAPE")

    first_ref = store.publish_contract(value, "EXAMPLE")
    with pytest.raises(FileExistsError):
        store.publish_contract(value, "EXAMPLE")
    assert store.load_contract(first_ref, ExampleContract) == value


def test_verify_rejects_content_corruption(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = store.publish_bytes(b"original", "BINARY", "application/octet-stream", "1.0.0")
    assert ref.relative_path is not None

    (tmp_path / ref.relative_path).write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="content hash mismatch"):
        store.verify_artifact(ref)


def test_verify_rejects_manifest_semantic_tampering(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = store.publish_bytes(b"original", "BINARY", "application/octet-stream", "1.0.0")
    assert ref.relative_path is not None
    artifact_path = tmp_path / ref.relative_path
    manifest_path = artifact_path.with_name(f"{artifact_path.name}.manifest.json")
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
    tampered = manifest.model_copy(update={"producer_task_id": "tampered-task"})
    manifest_path.write_bytes(canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        store.verify_artifact(ref)


def test_load_bytes_consumes_the_exact_payload_that_was_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    ref = store.publish_bytes(b"AAAA", "BINARY", "application/octet-stream", "1.0.0")
    assert ref.relative_path is not None
    artifact_path = tmp_path / ref.relative_path
    swapped = False
    real_sha256_file = store_module.sha256_file

    def hash_then_swap(path: Path) -> str:
        nonlocal swapped
        digest = real_sha256_file(path)
        if path == artifact_path and not swapped:
            artifact_path.write_bytes(b"BBBB")
            swapped = True
        return digest

    monkeypatch.setattr(store_module, "sha256_file", hash_then_swap)

    assert store.load_bytes(ref) == b"AAAA"
