from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TypeVar

from tarca.contracts import ArtifactRef, StrictContractModel, canonical_json_bytes, sha256_file

TContract = TypeVar("TContract", bound=StrictContractModel)


class LocalArtifactStore:
    """Minimal local Stage 0 store with strict reload and atomic publication."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.repo_root / relative_path).resolve()
        if candidate == self.repo_root or self.repo_root not in candidate.parents:
            raise ValueError(f"artifact path escapes repository: {relative_path}")
        return candidate

    def ref_for_file(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        relative_path: str,
        schema_version: str = "1.0.0",
    ) -> ArtifactRef:
        path = self.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash=sha256_file(path),
            schema_version=schema_version,
            relative_path=relative_path,
        )

    def publish_contract(
        self,
        value: TContract,
        *,
        artifact_id: str,
        artifact_type: str,
        relative_path: str,
        schema_version: str = "1.0.0",
        overwrite: bool = False,
    ) -> ArtifactRef:
        path = self.resolve(relative_path)
        payload = canonical_json_bytes(value) + b"\n"
        self._atomic_write_contract(path, payload, type(value), overwrite=overwrite)
        return self.ref_for_file(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            schema_version=schema_version,
        )

    def load_contract(self, ref: ArtifactRef, expected_type: type[TContract]) -> TContract:
        self.verify_artifact(ref)
        if ref.relative_path is None:  # pragma: no cover - enforced by verify_artifact
            raise ValueError("artifact has no repository path")
        return expected_type.model_validate_json(self.resolve(ref.relative_path).read_bytes())

    def verify_artifact(self, ref: ArtifactRef) -> bool:
        if ref.relative_path is None:
            raise ValueError(f"artifact has no repository path: {ref.artifact_id}")
        path = self.resolve(ref.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != ref.content_hash:
            raise ValueError(f"content hash mismatch for {ref.artifact_id}")
        return True

    @staticmethod
    def _atomic_write_contract(
        path: Path,
        payload: bytes,
        expected_type: type[StrictContractModel],
        *,
        overwrite: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            expected_type.model_validate_json(temporary_path.read_bytes())
            os.replace(temporary_path, path)
            expected_type.model_validate_json(path.read_bytes())
        finally:
            temporary_path.unlink(missing_ok=True)
