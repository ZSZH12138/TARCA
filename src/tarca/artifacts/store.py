from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

import pyarrow as pa  # type: ignore[import-untyped]  # PyArrow 25 ships no py.typed marker.

from tarca.contracts.arrow_schemas import validate_table
from tarca.contracts.artifacts import ArtifactManifest, ArtifactRef
from tarca.contracts.base import (
    CONTRACT_SCHEMA_VERSION,
    PROTOCOL_ID,
    StrictContractModel,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

TContract = TypeVar("TContract", bound=StrictContractModel)
FileValidator = Callable[[Path], None]

_ARTIFACT_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_LOGICAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ArtifactStore(Protocol):
    def publish_contract(
        self, value: StrictContractModel, artifact_type: str
    ) -> ArtifactRef: ...

    def publish_arrow(
        self, table: pa.Table, expected_schema: pa.Schema, artifact_type: str
    ) -> ArtifactRef: ...

    def publish_bytes(
        self, value: bytes, artifact_type: str, media_type: str, schema_version: str
    ) -> ArtifactRef: ...

    def publish_text(
        self, value: str, artifact_type: str, media_type: str, schema_version: str
    ) -> ArtifactRef: ...

    def load_contract(
        self, ref: ArtifactRef, expected_type: type[TContract]
    ) -> TContract: ...

    def load_arrow(self, ref: ArtifactRef, expected_schema: pa.Schema) -> pa.Table: ...

    def load_bytes(self, ref: ArtifactRef) -> bytes: ...

    def verify_artifact(self, ref: ArtifactRef) -> bool: ...


class LocalArtifactStore:
    """Typed, content-addressed Stage 1 artifact store with atomic publication."""

    def __init__(
        self,
        repo_root: Path,
        *,
        producer_stage: str,
        producer_task_id: str,
        scientific_identity_hash: str,
        dependencies: tuple[ArtifactRef, ...] = (),
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.store_root = self.repo_root / "artifacts" / "stage1a" / "store"
        self.producer_stage = self._validate_logical_key(producer_stage, "producer_stage")
        self.producer_task_id = self._validate_logical_key(
            producer_task_id, "producer_task_id"
        )
        if re.fullmatch(r"[0-9a-f]{64}", scientific_identity_hash) is None:
            raise ValueError("scientific_identity_hash must be a lowercase SHA-256 hash")
        self.scientific_identity_hash = scientific_identity_hash
        self.dependencies = dependencies

    def publish_contract(
        self, value: StrictContractModel, artifact_type: str
    ) -> ArtifactRef:
        if not isinstance(value, StrictContractModel):
            raise TypeError("value must be a StrictContractModel")
        payload = canonical_json_bytes(value) + b"\n"
        expected_type = type(value)
        return self._publish_payload(
            payload,
            artifact_type=artifact_type,
            media_type="application/json",
            serializer_id="canonical-json",
            schema_version=CONTRACT_SCHEMA_VERSION,
            suffix="json",
            validator=lambda path: self._validate_contract_file(
                path, expected_type, payload
            ),
        )

    def publish_arrow(
        self, table: pa.Table, expected_schema: pa.Schema, artifact_type: str
    ) -> ArtifactRef:
        self._validate_arrow_request(table, expected_schema, artifact_type)
        payload = self._serialize_arrow(table)
        return self._publish_payload(
            payload,
            artifact_type=artifact_type,
            media_type="application/vnd.apache.arrow.file",
            serializer_id="pyarrow-ipc-25",
            schema_version=CONTRACT_SCHEMA_VERSION,
            suffix="arrow",
            validator=lambda path: self._validate_arrow_file(path, expected_schema),
        )

    def publish_bytes(
        self, value: bytes, artifact_type: str, media_type: str, schema_version: str
    ) -> ArtifactRef:
        if type(value) is not bytes:
            raise TypeError("value must be bytes")
        return self._publish_payload(
            value,
            artifact_type=artifact_type,
            media_type=media_type,
            serializer_id="raw-bytes",
            schema_version=schema_version,
            suffix="bin",
            validator=lambda path: self._validate_exact_bytes(path, value),
        )

    def publish_text(
        self, value: str, artifact_type: str, media_type: str, schema_version: str
    ) -> ArtifactRef:
        if type(value) is not str:
            raise TypeError("value must be str")
        payload = value.encode("utf-8")
        return self._publish_payload(
            payload,
            artifact_type=artifact_type,
            media_type=media_type,
            serializer_id="utf-8",
            schema_version=schema_version,
            suffix="txt",
            validator=lambda path: self._validate_utf8_text(path, value),
        )

    def load_contract(
        self, ref: ArtifactRef, expected_type: type[TContract]
    ) -> TContract:
        path = self._verified_path(ref)
        value = expected_type.model_validate_json(path.read_bytes())
        if canonical_json_bytes(value) + b"\n" != path.read_bytes():
            raise ValueError("contract artifact is not canonically serialized")
        return value

    def load_arrow(self, ref: ArtifactRef, expected_schema: pa.Schema) -> pa.Table:
        path = self._verified_path(ref)
        table = self._read_arrow(path)
        return validate_table(table, expected_schema)

    def load_bytes(self, ref: ArtifactRef) -> bytes:
        return self._verified_path(ref).read_bytes()

    def verify_artifact(self, ref: ArtifactRef) -> bool:
        self._verified_path(ref)
        return True

    def _publish_payload(
        self,
        payload: bytes,
        *,
        artifact_type: str,
        media_type: str,
        serializer_id: str,
        schema_version: str,
        suffix: str,
        validator: FileValidator,
    ) -> ArtifactRef:
        self._validate_publication_metadata(artifact_type, media_type, schema_version)
        temporary_path = self._write_temporary(payload)
        artifact_path: Path | None = None
        manifest_path: Path | None = None
        artifact_published = False
        try:
            content_hash = sha256_file(temporary_path)
            if content_hash != sha256_bytes(payload):
                raise ValueError("temporary artifact content hash mismatch")
            validator(temporary_path)
            artifact_path = self._artifact_path(artifact_type, content_hash, suffix)
            self._atomic_link(temporary_path, artifact_path)
            artifact_published = True
            ref = self._build_ref(artifact_path, artifact_type, content_hash, schema_version)
            manifest_path = self._publish_manifest(
                ref, artifact_path, media_type, serializer_id
            )
            self._verified_path(ref)
            return ref
        except Exception:
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)
            if artifact_published and artifact_path is not None:
                artifact_path.unlink(missing_ok=True)
            raise
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_temporary(self, payload: bytes) -> Path:
        self.store_root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".publish-", suffix=".tmp", dir=self.store_root)
        path = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def _publish_manifest(
        self,
        ref: ArtifactRef,
        artifact_path: Path,
        media_type: str,
        serializer_id: str,
    ) -> Path:
        manifest = ArtifactManifest(
            artifact=ref,
            media_type=media_type,
            serializer_id=serializer_id,
            producer_stage=self.producer_stage,
            producer_task_id=self.producer_task_id,
            scientific_identity_hash=self.scientific_identity_hash,
            dependencies=self.dependencies,
            size_bytes=artifact_path.stat().st_size,
            created_at=datetime.now(UTC),
        )
        payload = canonical_json_bytes(manifest) + b"\n"
        manifest_path = self._manifest_path(artifact_path)
        temporary_path = self._write_temporary(payload)
        manifest_published = False
        try:
            self._validate_contract_file(temporary_path, ArtifactManifest, payload)
            self._atomic_link(temporary_path, manifest_path)
            manifest_published = True
            ArtifactManifest.model_validate_json(manifest_path.read_bytes())
            return manifest_path
        except Exception:
            if manifest_published:
                manifest_path.unlink(missing_ok=True)
            raise
        finally:
            temporary_path.unlink(missing_ok=True)

    def _verified_path(self, ref: ArtifactRef) -> Path:
        if ref.relative_path is None:
            raise ValueError(f"artifact has no repository path: {ref.artifact_id}")
        path = self._resolve(ref.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != ref.content_hash:
            raise ValueError(f"content hash mismatch for {ref.artifact_id}")
        manifest_path = self._manifest_path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.artifact != ref or manifest.size_bytes != path.stat().st_size:
            raise ValueError(f"artifact manifest mismatch for {ref.artifact_id}")
        return path

    def _artifact_path(self, artifact_type: str, content_hash: str, suffix: str) -> Path:
        return self.store_root / artifact_type.lower() / f"{content_hash}.{suffix}"

    @staticmethod
    def _manifest_path(artifact_path: Path) -> Path:
        return artifact_path.with_name(f"{artifact_path.name}.manifest.json")

    def _build_ref(
        self,
        path: Path,
        artifact_type: str,
        content_hash: str,
        schema_version: str,
    ) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=f"{artifact_type.lower()}-{content_hash}",
            artifact_type=artifact_type,
            content_hash=content_hash,
            schema_version=schema_version,
            relative_path=path.relative_to(self.repo_root).as_posix(),
        )

    @staticmethod
    def _atomic_link(temporary_path: Path, final_path: Path) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(temporary_path, final_path)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.repo_root / relative_path).resolve()
        if candidate == self.repo_root or self.repo_root not in candidate.parents:
            raise ValueError(f"artifact path escapes repository: {relative_path}")
        if candidate != self.store_root and self.store_root not in candidate.parents:
            raise ValueError(f"artifact path is outside the Stage 1 store: {relative_path}")
        return candidate

    @staticmethod
    def _validate_contract_file(
        path: Path, expected_type: type[StrictContractModel], expected_payload: bytes
    ) -> None:
        reloaded = expected_type.model_validate_json(path.read_bytes())
        if canonical_json_bytes(reloaded) + b"\n" != expected_payload:
            raise ValueError("contract artifact canonical reload mismatch")

    @staticmethod
    def _serialize_arrow(table: pa.Table) -> bytes:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        return bytes(sink.getvalue())

    @staticmethod
    def _read_arrow(path: Path) -> pa.Table:
        with pa.memory_map(str(path), "r") as source:
            return pa.ipc.open_file(source).read_all()

    @classmethod
    def _validate_arrow_file(cls, path: Path, expected_schema: pa.Schema) -> None:
        validate_table(cls._read_arrow(path), expected_schema)

    @staticmethod
    def _validate_exact_bytes(path: Path, expected: bytes) -> None:
        if path.read_bytes() != expected:
            raise ValueError("binary artifact reload mismatch")

    @staticmethod
    def _validate_utf8_text(path: Path, expected: str) -> None:
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError("text artifact reload mismatch")

    @staticmethod
    def _validate_arrow_request(
        table: pa.Table, expected_schema: pa.Schema, artifact_type: str
    ) -> None:
        validate_table(table, expected_schema)
        metadata = expected_schema.metadata or {}
        if metadata.get(b"contract_schema_version") != CONTRACT_SCHEMA_VERSION.encode():
            raise ValueError("Arrow schema contract version mismatch")
        if metadata.get(b"protocol_id") != PROTOCOL_ID.encode():
            raise ValueError("Arrow schema protocol ID mismatch")
        if metadata.get(b"artifact_type") != artifact_type.encode():
            raise ValueError("Arrow schema artifact_type mismatch")

    @staticmethod
    def _validate_publication_metadata(
        artifact_type: str, media_type: str, schema_version: str
    ) -> None:
        if _ARTIFACT_TYPE_PATTERN.fullmatch(artifact_type) is None:
            raise ValueError("artifact_type must be an uppercase logical key")
        if not media_type.strip() or any(character in media_type for character in "\r\n\0"):
            raise ValueError("media_type must be nonblank single-line text")
        if not schema_version.strip() or any(
            character in schema_version for character in "\r\n\0"
        ):
            raise ValueError("schema_version must be nonblank single-line text")

    @staticmethod
    def _validate_logical_key(value: str, field_name: str) -> str:
        if _LOGICAL_KEY_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field_name} must be a logical key")
        return value
