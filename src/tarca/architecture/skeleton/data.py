"""Data-plane skeleton preserved outside the grandfathered Stage 1 package surface."""

from __future__ import annotations

from typing import Protocol

from tarca.contracts.common import ArtifactRef
from tarca.contracts.data import WindowBatch
from tarca.contracts.errors import UnimplementedCapabilityError
from tarca.contracts.future import AccessScope, DatasetSpec, SplitSpec


class DatasetBuilder(Protocol):
    def build_windows(
        self, dataset: DatasetSpec, split: SplitSpec, access: AccessScope
    ) -> WindowBatch: ...


class DataTransformer(Protocol):
    def fit_transform_train_only(self, dataset: DatasetSpec) -> ArtifactRef: ...

    def transform(self, dataset: DatasetSpec, artifact: ArtifactRef) -> WindowBatch: ...


def build_windows(
    dataset: DatasetSpec | None, split: SplitSpec | None = None, access: AccessScope | None = None
) -> WindowBatch:
    raise UnimplementedCapabilityError("data.build_windows")


def temporal_split(dataset: DatasetSpec, split: SplitSpec) -> ArtifactRef:
    raise UnimplementedCapabilityError("data.temporal_split")


def fit_transform_train_only(dataset: DatasetSpec) -> ArtifactRef:
    raise UnimplementedCapabilityError("data.fit_transform_train_only")


def transform(dataset: DatasetSpec, artifact: ArtifactRef) -> WindowBatch:
    raise UnimplementedCapabilityError("data.transform")


def hash_dataset(dataset: DatasetSpec) -> str:
    raise UnimplementedCapabilityError("data.hash_dataset")


def validate_data_scope(access: AccessScope) -> AccessScope:
    if not isinstance(access, AccessScope):
        raise TypeError("access must be AccessScope")
    return access
