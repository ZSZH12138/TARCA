from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tarca.contracts.base import Sha256Hash, canonical_json_bytes, sha256_bytes
from tarca.contracts.data import (
    DatasetRegistryEntry,
    DatasetRegistryManifest,
    DatasetSourceKind,
    LeakageAudit,
    WindowBatch,
)
from tarca.contracts.data_access import (
    AccessScope,
    DatasetSpec,
    DatasetWindowPartition,
    SealedAccessGrant,
    validate_sealed_access,
)

from .payload import (
    PersistedDatasetPayloadManifest,
    PersistedPartitionPayload,
    PersistedPayloadFile,
)
from .persisted import LocalPayloadBackend, PayloadBackend, load_window_batch


class PersistedDatasetRepository:
    """Registry-injected reader for already materialized Stage 1 window payloads."""

    def __init__(
        self,
        repo_root: Path,
        registry: DatasetRegistryManifest,
        *,
        backend: PayloadBackend | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry = registry
        self.backend = backend or LocalPayloadBackend()
        self.clock = clock or (lambda: datetime.now(UTC))

    def resolve_dataset(self, dataset: DatasetSpec) -> DatasetRegistryEntry:
        for entry in self.registry.entries:
            if entry.dataset == dataset:
                return entry
        raise KeyError(f"no exact dataset registry entry for {dataset.name}@{dataset.version}")

    def build_windows(
        self,
        dataset: DatasetSpec,
        partition: DatasetWindowPartition,
        access: AccessScope,
        grant: SealedAccessGrant | None = None,
    ) -> WindowBatch:
        batch, _audit = self.build_windows_with_audit(dataset, partition, access, grant)
        return batch

    def build_windows_with_audit(
        self,
        dataset: DatasetSpec,
        partition: DatasetWindowPartition,
        access: AccessScope,
        grant: SealedAccessGrant | None = None,
    ) -> tuple[WindowBatch, LeakageAudit]:
        """Load one physical partition and return the mandatory loader audit sidecar."""
        entry = self.resolve_dataset(dataset)
        self._authorize(entry, (partition,), access, grant)
        self._require_persisted_source(entry)
        if partition not in entry.available_partitions:
            raise KeyError(f"no exact physical partition {partition.value} for dataset")
        manifest, dataset_root = self._load_manifest(entry)
        payload = self._partition_payload(manifest, partition)
        files, verified_payloads = self._verify_partition_files(dataset_root, payload)
        batch = load_window_batch(files, verified_payloads)
        audit = self._audit_loaded_partition(batch, partition)
        if not audit.passed:
            raise ValueError(f"loader leakage audit failed: {'; '.join(audit.findings)}")
        return batch, audit

    def hash_dataset(
        self,
        dataset: DatasetSpec,
        access: AccessScope,
        grant: SealedAccessGrant | None = None,
    ) -> Sha256Hash:
        entry = self.resolve_dataset(dataset)
        self._authorize(entry, entry.available_partitions, access, grant)
        self._require_persisted_source(entry)
        manifest, dataset_root = self._load_manifest(entry)
        for payload in manifest.partitions:
            self._verify_partition_files(dataset_root, payload)
        return entry.expected_dataset_hash

    def _authorize(
        self,
        entry: DatasetRegistryEntry,
        partitions: tuple[DatasetWindowPartition, ...],
        access: AccessScope,
        grant: SealedAccessGrant | None,
    ) -> None:
        effective_access = AccessScope(
            sealed=entry.sealed or access.sealed,
            scope_name=access.scope_name,
        )
        accessed_at = self.clock()
        for partition in partitions:
            validate_sealed_access(entry.dataset, partition, effective_access, grant, accessed_at)

    @staticmethod
    def _require_persisted_source(entry: DatasetRegistryEntry) -> None:
        if entry.source_kind is DatasetSourceKind.STAGE1_SYNTHETIC_CONFIG:
            raise NotImplementedError(
                "STAGE1_SYNTHETIC_CONFIG is reserved for Stage 1B and fails closed in Stage 1A"
            )
        if entry.source_kind is not DatasetSourceKind.PERSISTED_STAGE1:
            raise ValueError(f"unsupported dataset source kind: {entry.source_kind}")

    def _load_manifest(
        self, entry: DatasetRegistryEntry
    ) -> tuple[PersistedDatasetPayloadManifest, Path]:
        dataset_root = self._dataset_root(entry)
        path = dataset_root / "payload_manifest.json"
        payload = self.backend.read_bytes(path)
        if sha256_bytes(payload) != entry.expected_dataset_hash:
            raise ValueError("payload manifest hash does not match the dataset registry")
        manifest = PersistedDatasetPayloadManifest.model_validate_json(payload)
        if canonical_json_bytes(manifest) + b"\n" != payload:
            raise ValueError("payload manifest is not canonically serialized")
        if manifest.dataset != entry.dataset:
            raise ValueError("payload manifest dataset identity mismatch")
        manifest_partitions = tuple(item.partition for item in manifest.partitions)
        if manifest_partitions != entry.available_partitions:
            raise ValueError("payload manifest partitions do not match the registry")
        return manifest, dataset_root

    def _dataset_root(self, entry: DatasetRegistryEntry) -> Path:
        candidate = (self.repo_root / entry.relative_location).resolve()
        if candidate == self.repo_root or self.repo_root not in candidate.parents:
            raise ValueError("dataset location escapes repository root")
        return candidate

    @staticmethod
    def _partition_payload(
        manifest: PersistedDatasetPayloadManifest,
        partition: DatasetWindowPartition,
    ) -> PersistedPartitionPayload:
        for payload in manifest.partitions:
            if payload.partition is partition:
                return payload
        raise KeyError(f"payload manifest has no exact physical partition {partition.value}")

    def _verify_partition_files(
        self,
        dataset_root: Path,
        payload: PersistedPartitionPayload,
    ) -> tuple[dict[str, PersistedPayloadFile], dict[str, bytes]]:
        descriptors: dict[str, PersistedPayloadFile] = {}
        verified_payloads: dict[str, bytes] = {}
        for descriptor in payload.files:
            path = self._resolve_payload_path(dataset_root, descriptor.relative_path)
            content = self.backend.read_bytes(path)
            if len(content) != descriptor.size_bytes:
                raise ValueError(f"payload file size mismatch: {descriptor.relative_path}")
            actual_hash = sha256_bytes(content)
            if actual_hash != descriptor.content_hash:
                raise ValueError(f"payload file hash mismatch: {descriptor.relative_path}")
            descriptors[descriptor.role] = descriptor
            verified_payloads[descriptor.role] = content
        return descriptors, verified_payloads

    @staticmethod
    def _audit_loaded_partition(
        batch: WindowBatch, requested_partition: DatasetWindowPartition
    ) -> LeakageAudit:
        loaded_partition = batch.metadata.get("physical_partition")
        if loaded_partition == requested_partition.value:
            return LeakageAudit(passed=True, findings=())
        return LeakageAudit(
            passed=False,
            findings=(
                f"loaded physical partition {loaded_partition} does not match requested "
                f"{requested_partition.value}",
            ),
        )

    @staticmethod
    def _resolve_payload_path(dataset_root: Path, relative_path: str) -> Path:
        candidate = (dataset_root / relative_path).resolve()
        if candidate == dataset_root or dataset_root not in candidate.parents:
            raise ValueError(f"payload path escapes dataset root: {relative_path}")
        return candidate
