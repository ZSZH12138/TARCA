from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
import torch

from tarca.contracts import (
    AccessScope,
    ArtifactRef,
    DatasetRegistryManifest,
    DatasetSourceKind,
    DatasetSpec,
    DatasetWindowPartition,
    LeakageAudit,
    SealedAccessGrant,
)
from tarca.data.persisted import LocalPayloadBackend
from tarca.data.repository import PersistedDatasetRepository

NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)


class PersistedFixture(Protocol):
    repo_root: Path
    dataset: DatasetSpec
    registry: DatasetRegistryManifest


PersistedFixtureFactory = Callable[..., PersistedFixture]


class CountingBackend(LocalPayloadBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def read_bytes(self, path: Path) -> bytes:
        self.calls.append(("read_bytes", path))
        return super().read_bytes(path)


class ReplacingAfterReadBackend(CountingBackend):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self.target = target
        self.replaced = False

    def read_bytes(self, path: Path) -> bytes:
        content = super().read_bytes(path)
        if path == self.target and not self.replaced:
            np.save(path, np.array([[[999.0], [999.5]]], dtype=np.float64))
            self.replaced = True
        return content


def _repository(fixture: PersistedFixture, backend: CountingBackend) -> PersistedDatasetRepository:
    return PersistedDatasetRepository(
        fixture.repo_root, fixture.registry, backend=backend, clock=lambda: NOW
    )


def _grant(
    dataset: DatasetSpec,
    partitions: tuple[DatasetWindowPartition, ...],
    *,
    scope_name: str = "evaluation",
) -> SealedAccessGrant:
    return SealedAccessGrant(
        grant_id="grant-1",
        dataset=dataset,
        scope_name=scope_name,
        allowed_partitions=partitions,
        authorization_ref=ArtifactRef(
            artifact_id="sealed-access-1",
            artifact_type="SEALED_ACCESS_AUTHORIZATION",
            content_hash="b" * 64,
            schema_version="1.0.0",
            relative_path=None,
        ),
        issued_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
    )


def test_exact_registry_resolution_and_partition_load_preserve_payload_identity(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory()
    backend = CountingBackend()
    repository = _repository(fixture, backend)

    assert repository.resolve_dataset(fixture.dataset).dataset == fixture.dataset
    with pytest.raises(KeyError, match="exact dataset"):
        repository.resolve_dataset(DatasetSpec(name="fixture", version="latest"))
    batch = repository.build_windows(
        fixture.dataset,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="development"),
    )

    assert batch.x.dtype is torch.float64
    assert batch.x.tolist() == [[[1.0], [1.5]]]
    assert batch.window_id == ("train-0",)
    assert batch.metadata["physical_partition"] == "TRAIN"
    with pytest.raises(KeyError, match="physical partition"):
        repository.build_windows(
            fixture.dataset,
            DatasetWindowPartition.TEST,
            AccessScope(sealed=False, scope_name="development"),
        )


def test_partition_load_uses_the_exact_bytes_that_passed_hash_verification(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(partitions=(DatasetWindowPartition.TRAIN,))
    x_path = fixture.repo_root / "fixture_dataset" / "windows" / "train" / "x.npy"
    backend = ReplacingAfterReadBackend(x_path)

    batch = _repository(fixture, backend).build_windows(
        fixture.dataset,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="development"),
    )

    assert backend.replaced
    assert batch.x.tolist() == [[[1.0], [1.5]]]


def test_new_loader_returns_a_passing_leakage_audit_for_the_requested_partition(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(partitions=(DatasetWindowPartition.TRAIN,))
    repository = _repository(fixture, CountingBackend())

    batch, audit = repository.build_windows_with_audit(
        fixture.dataset,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="development"),
    )

    assert batch.metadata["physical_partition"] == "TRAIN"
    assert audit == LeakageAudit(passed=True, findings=())


def test_standard_loader_fails_closed_when_partition_leakage_audit_fails(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(
        partitions=(DatasetWindowPartition.TRAIN,),
        physical_partition_override=DatasetWindowPartition.TEST_UNSEEN_REGIME,
    )
    repository = _repository(fixture, CountingBackend())

    with pytest.raises(ValueError, match="loader leakage audit failed"):
        repository.build_windows_with_audit(
            fixture.dataset,
            DatasetWindowPartition.TRAIN,
            AccessScope(sealed=False, scope_name="development"),
        )

    with pytest.raises(ValueError, match="loader leakage audit failed"):
        repository.build_windows(
            fixture.dataset,
            DatasetWindowPartition.TRAIN,
            AccessScope(sealed=False, scope_name="development"),
        )


def test_registry_sealed_cannot_be_downgraded_and_denial_precedes_io(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(sealed=True)
    backend = CountingBackend()
    repository = _repository(fixture, backend)

    with pytest.raises(PermissionError, match="requires a grant"):
        repository.build_windows(
            fixture.dataset,
            DatasetWindowPartition.TRAIN,
            AccessScope(sealed=False, scope_name="evaluation"),
        )
    assert backend.calls == []

    batch = repository.build_windows(
        fixture.dataset,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="evaluation"),
        _grant(fixture.dataset, (DatasetWindowPartition.TRAIN,)),
    )
    assert batch.window_id == ("train-0",)


def test_full_hash_requires_grant_for_every_partition_before_io(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(sealed=True)
    backend = CountingBackend()
    repository = _repository(fixture, backend)

    with pytest.raises(PermissionError, match="partition mismatch"):
        repository.hash_dataset(
            fixture.dataset,
            AccessScope(sealed=True, scope_name="evaluation"),
            _grant(fixture.dataset, (DatasetWindowPartition.TRAIN,)),
        )
    assert backend.calls == []

    partitions = repository.resolve_dataset(fixture.dataset).available_partitions
    assert (
        repository.hash_dataset(
            fixture.dataset,
            AccessScope(sealed=True, scope_name="evaluation"),
            _grant(fixture.dataset, partitions),
        )
        == repository.resolve_dataset(fixture.dataset).expected_dataset_hash
    )


def test_payload_hash_mismatch_and_object_array_fail_closed(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory()
    repository = _repository(fixture, CountingBackend())
    x_path = fixture.repo_root / "fixture_dataset" / "windows" / "train" / "x.npy"
    original = x_path.read_bytes()
    x_path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="payload file hash mismatch"):
        repository.build_windows(
            fixture.dataset,
            DatasetWindowPartition.TRAIN,
            AccessScope(sealed=False, scope_name="development"),
        )

    object_fixture = persisted_fixture_factory(
        partitions=(DatasetWindowPartition.TEST_UNSEEN_REGIME,), object_x=True
    )
    with pytest.raises(ValueError, match="allow_pickle=False"):
        _repository(object_fixture, CountingBackend()).build_windows(
            object_fixture.dataset,
            DatasetWindowPartition.TEST_UNSEEN_REGIME,
            AccessScope(sealed=False, scope_name="development"),
        )


def test_stage1_synthetic_source_is_reserved_for_stage1b(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(source_kind=DatasetSourceKind.STAGE1_SYNTHETIC_CONFIG)
    backend = CountingBackend()

    with pytest.raises(NotImplementedError, match="Stage 1B"):
        _repository(fixture, backend).build_windows(
            fixture.dataset,
            DatasetWindowPartition.TRAIN,
            AccessScope(sealed=False, scope_name="development"),
        )
    assert backend.calls == []
