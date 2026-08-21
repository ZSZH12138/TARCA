from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tarca.contracts import (
    DatasetRegistryEntry,
    DatasetRegistryManifest,
    DatasetSourceKind,
    DatasetSpec,
    DatasetWindowPartition,
    canonical_json_bytes,
    sha256_file,
)
from tarca.data.payload import (
    PersistedDatasetPayloadManifest,
    PersistedPartitionPayload,
    PersistedPayloadFile,
    PersistedWindowMetadata,
)


@dataclass(frozen=True)
class PersistedFixture:
    repo_root: Path
    dataset: DatasetSpec
    registry: DatasetRegistryManifest


PersistedFixtureFactory = Callable[..., PersistedFixture]


def _window_metadata(partition: DatasetWindowPartition) -> PersistedWindowMetadata:
    return PersistedWindowMetadata(
        window_id=(f"{partition.value.lower()}-0",),
        input_feature_names=("load",),
        target_names=("load_target",),
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=(datetime(2026, 1, 1, tzinfo=UTC),),
        feature_end=(datetime(2026, 1, 1, 1, tzinfo=UTC),),
        prediction_start=(datetime(2026, 1, 1, 2, tzinfo=UTC),),
        label_end=(datetime(2026, 1, 1, 2, tzinfo=UTC),),
        forecast_time=((datetime(2026, 1, 1, 2, tzinfo=UTC),),),
        metadata={"physical_partition": partition.value},
    )


def _write_partition(
    dataset_root: Path,
    partition: DatasetWindowPartition,
    value: float,
    *,
    object_x: bool,
) -> PersistedPartitionPayload:
    relative_root = Path("windows") / partition.value.lower()
    partition_root = dataset_root / relative_root
    partition_root.mkdir(parents=True)
    x = (
        np.array([[['unsafe']]], dtype=object)
        if object_x
        else np.array([[[value], [value + 0.5]]], dtype=np.float64)
    )
    arrays = {
        "x": x,
        "y": np.array([[[value + 1.0]]], dtype=np.float64),
        "x_observed_mask": np.ones(x.shape, dtype=np.bool_),
        "y_observed_mask": np.ones((1, 1, 1), dtype=np.bool_),
        "regime": np.array([int(value)], dtype=np.int64),
    }
    files: list[PersistedPayloadFile] = []
    for role, array in arrays.items():
        path = partition_root / f"{role}.npy"
        np.save(path, array, allow_pickle=object_x and role == "x")
        files.append(
            PersistedPayloadFile(
                role=role,
                relative_path=path.relative_to(dataset_root).as_posix(),
                content_hash=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    metadata_path = partition_root / "metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(_window_metadata(partition)) + b"\n")
    files.append(
        PersistedPayloadFile(
            role="metadata",
            relative_path=metadata_path.relative_to(dataset_root).as_posix(),
            content_hash=sha256_file(metadata_path),
            size_bytes=metadata_path.stat().st_size,
        )
    )
    return PersistedPartitionPayload(partition=partition, files=tuple(files))


@pytest.fixture
def persisted_fixture_factory(tmp_path: Path) -> PersistedFixtureFactory:
    fixture_count = 0

    def build(
        *,
        partitions: tuple[DatasetWindowPartition, ...] = (
            DatasetWindowPartition.TRAIN,
            DatasetWindowPartition.TEST_SEEN_REGIME,
        ),
        sealed: bool = False,
        source_kind: DatasetSourceKind = DatasetSourceKind.PERSISTED_STAGE1,
        object_x: bool = False,
    ) -> PersistedFixture:
        nonlocal fixture_count
        fixture_count += 1
        suffix = "" if fixture_count == 1 else f"-{fixture_count}"
        relative_location = f"fixture_dataset{suffix}"
        dataset = DatasetSpec(name=f"fixture{suffix}", version="1.0")
        dataset_root = tmp_path / relative_location
        dataset_root.mkdir()
        payloads = tuple(
            _write_partition(dataset_root, partition, float(index + 1), object_x=object_x)
            for index, partition in enumerate(partitions)
        )
        manifest = PersistedDatasetPayloadManifest(
            schema_version="1.0.0", dataset=dataset, partitions=payloads
        )
        manifest_path = dataset_root / "payload_manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        entry = DatasetRegistryEntry(
            dataset=dataset,
            source_kind=source_kind,
            relative_location=relative_location,
            expected_dataset_hash=sha256_file(manifest_path),
            sealed=sealed,
            available_partitions=partitions,
        )
        registry = DatasetRegistryManifest(
            registry_id="test-registry", registry_version="1.0", entries=(entry,)
        )
        return PersistedFixture(repo_root=tmp_path, dataset=dataset, registry=registry)

    return build
