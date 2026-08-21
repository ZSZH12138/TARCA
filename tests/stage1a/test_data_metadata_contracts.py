from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tarca.contracts import (
    CONTRACT_SCHEMA_VERSION,
    DataManifest,
    DatasetRegistryEntry,
    DatasetRegistryManifest,
    DatasetSourceKind,
    DatasetSpec,
    DatasetWindowPartition,
    DataSplitSummary,
    LeakageAudit,
    SplitPartition,
    WindowContractSummary,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _registry_entry(**updates: object) -> DatasetRegistryEntry:
    values: dict[str, object] = {
        "dataset": DatasetSpec(name="synthetic-small", version="1.0.0"),
        "source_kind": DatasetSourceKind.PERSISTED_STAGE1,
        "relative_location": "fixtures/synthetic-small",
        "expected_dataset_hash": HASH_A,
        "sealed": False,
        "available_partitions": (
            DatasetWindowPartition.TRAIN,
            DatasetWindowPartition.VALIDATION,
        ),
    }
    values.update(updates)
    return DatasetRegistryEntry.model_validate(values)


def _window_summary() -> WindowContractSummary:
    return WindowContractSummary(
        history_length=24,
        horizon=6,
        input_feature_names=("load", "temperature"),
        target_names=("load",),
        observed_covariate_names=("temperature",),
        known_future_covariate_names=("holiday",),
        timezone="UTC",
        missingness_protocol="bool-mask-no-nan",
    )


def test_registry_rejects_duplicate_dataset_identity() -> None:
    entry = _registry_entry()

    with pytest.raises(ValidationError, match="dataset identities must be unique"):
        DatasetRegistryManifest(
            registry_id="stage1a-test",
            registry_version="1.0.0",
            entries=(entry, entry),
        )


@pytest.mark.parametrize(
    "location",
    ("", "../outside", "/absolute", "C:/absolute", "folder\\file"),
)
def test_registry_rejects_unsafe_relative_location(location: str) -> None:
    with pytest.raises(ValidationError, match="canonical POSIX relative path"):
        _registry_entry(relative_location=location)


def test_registry_requires_unique_nonempty_physical_partitions() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _registry_entry(available_partitions=())

    with pytest.raises(ValidationError, match="must be unique"):
        _registry_entry(
            available_partitions=(
                DatasetWindowPartition.TRAIN,
                DatasetWindowPartition.TRAIN,
            )
        )


def test_data_manifest_binds_frozen_schema_and_unique_splits() -> None:
    train = DataSplitSummary(partition=SplitPartition.TRAIN, split_hash=HASH_A, count=12)
    manifest = DataManifest(
        schema_version=CONTRACT_SCHEMA_VERSION,
        dataset_name="synthetic-small",
        dataset_version="1.0.0",
        dataset_hash=HASH_B,
        splits=(train,),
        window_contract=_window_summary(),
        source_description="Small persisted Stage 1 fixture.",
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest.dataset_hash == HASH_B
    with pytest.raises(ValidationError, match="schema_version"):
        DataManifest(**{**manifest.model_dump(), "schema_version": "2.0.0"})
    with pytest.raises(ValidationError, match="split partitions must be unique"):
        DataManifest(**{**manifest.model_dump(), "splits": (train, train)})


def test_window_summary_rejects_overlap_and_nonpositive_lengths() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        WindowContractSummary(
            **{
                **_window_summary().model_dump(),
                "known_future_covariate_names": ("load",),
            }
        )

    with pytest.raises(ValidationError):
        WindowContractSummary(**{**_window_summary().model_dump(), "horizon": 0})


def test_leakage_audit_is_frozen_and_cannot_pass_with_findings() -> None:
    audit = LeakageAudit(passed=True, findings=())

    with pytest.raises(FrozenInstanceError):
        audit.passed = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="cannot pass with findings"):
        LeakageAudit(passed=True, findings=("test partition was opened",))
