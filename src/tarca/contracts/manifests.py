"""Strict persistent manifests and in-memory partition leakage checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import RegimeRelation, RunStatus, SplitPartition
from .version import CONTRACT_SCHEMA_VERSION

_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_EVIDENCE_LIMIT = 5
_PARTITIONS = tuple(SplitPartition)


def _require_non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be timezone-aware with UTC offset zero")
    return value


NonEmptyString = Annotated[str, AfterValidator(_require_non_empty)]
Sha256Hash = Annotated[str, Field(pattern=_HASH_PATTERN)]
GitCommit = Annotated[str, Field(pattern=_GIT_COMMIT_PATTERN)]
UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class StrictContractModel(BaseModel):
    """Shared immutable, strict base for persistent TARCA contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[CONTRACT_SCHEMA_VERSION] = CONTRACT_SCHEMA_VERSION


class DataSplitSummary(StrictContractModel):
    """Hash and row count for one dataset partition."""

    partition: SplitPartition
    split_hash: Sha256Hash
    count: NonNegativeInt


class WindowContractSummary(StrictContractModel):
    """Typed summary of the window construction contract."""

    history_length: PositiveInt
    horizon: PositiveInt
    input_feature_names: tuple[NonEmptyString, ...]
    target_names: tuple[NonEmptyString, ...]
    observed_covariate_names: tuple[NonEmptyString, ...]
    known_future_covariate_names: tuple[NonEmptyString, ...]
    timezone: Literal["UTC"]
    missingness_protocol: NonEmptyString

    @field_validator(
        "input_feature_names",
        "target_names",
        "observed_covariate_names",
        "known_future_covariate_names",
    )
    @classmethod
    def _require_unique_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if len(names) != len(set(names)):
            raise ValueError("names must be unique")
        return names


class InterventionPair(StrictContractModel):
    """One deterministic intervention pairing with allocation metadata."""

    pair_id: Sha256Hash
    partition: SplitPartition
    base_window_id: NonEmptyString
    source_window_id: NonEmptyString
    concept_name: NonEmptyString
    regime_relation: RegimeRelation
    matching_distance: NonNegativeFiniteFloat
    concept_delta: FiniteFloat

    @classmethod
    def build(
        cls,
        *,
        partition: SplitPartition,
        base_window_id: str,
        source_window_id: str,
        concept_name: str,
        regime_relation: RegimeRelation,
        matching_distance: float,
        concept_delta: float,
    ) -> Self:
        """Build a pair with its canonical, allocation-independent identity."""

        return cls(
            pair_id=_compute_pair_id(
                base_window_id=base_window_id,
                source_window_id=source_window_id,
                concept_name=concept_name,
                regime_relation=regime_relation,
            ),
            partition=partition,
            base_window_id=base_window_id,
            source_window_id=source_window_id,
            concept_name=concept_name,
            regime_relation=regime_relation,
            matching_distance=matching_distance,
            concept_delta=concept_delta,
        )

    @model_validator(mode="after")
    def _validate_pair_identity(self) -> Self:
        if self.base_window_id == self.source_window_id:
            raise ValueError("base_window_id and source_window_id must differ")
        expected_pair_id = _compute_pair_id(
            base_window_id=self.base_window_id,
            source_window_id=self.source_window_id,
            concept_name=self.concept_name,
            regime_relation=self.regime_relation,
        )
        if self.pair_id != expected_pair_id:
            raise ValueError("pair_id does not match the canonical pair identity")
        return self


class DataManifest(StrictContractModel):
    """Typed manifest for a versioned, partitioned dataset."""

    dataset_name: NonEmptyString
    dataset_version: NonEmptyString
    dataset_hash: Sha256Hash
    splits: tuple[DataSplitSummary, ...]
    window_contract: WindowContractSummary
    source_description: NonEmptyString
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _require_complete_splits(self) -> Self:
        partitions = tuple(summary.partition for summary in self.splits)
        if len(partitions) != len(_PARTITIONS) or set(partitions) != set(_PARTITIONS):
            expected = ", ".join(partition.value for partition in _PARTITIONS)
            raise ValueError(f"splits must contain exactly one summary for each of: {expected}")
        return self


class RunManifest(StrictContractModel):
    """Immutable provenance and lifecycle state for one experiment run."""

    experiment_id: NonEmptyString
    run_id: NonEmptyString
    config_hash: Sha256Hash
    data_hash: Sha256Hash
    git_commit: GitCommit
    created_at: UtcDatetime
    status: RunStatus


class MetricRecord(StrictContractModel):
    """One long-format metric observation."""

    experiment_id: NonEmptyString
    run_id: NonEmptyString
    split: SplitPartition
    metric: NonEmptyString
    value: FiniteFloat
    regime: NonEmptyString | None
    horizon: PositiveInt | None
    concept: NonEmptyString | None


def validate_disjoint_window_partitions(
    partitions: Mapping[SplitPartition, Iterable[str]],
) -> None:
    """Reject a window ID allocated to more than one dataset partition."""

    if not isinstance(partitions, Mapping):
        raise TypeError("partitions must be a mapping")
    supplied_keys = tuple(partitions)
    invalid_keys = tuple(key for key in supplied_keys if not isinstance(key, SplitPartition))
    if invalid_keys:
        raise TypeError("partitions must use SplitPartition keys")
    missing = tuple(partition for partition in _PARTITIONS if partition not in partitions)
    if missing:
        names = ", ".join(partition.value for partition in missing)
        raise ValueError(f"partitions missing required partitions: {names}")

    normalized = {
        partition: _normalize_ids(partitions[partition], f"partitions[{partition.value}]")
        for partition in _PARTITIONS
    }
    records = (
        (window_id, partition) for partition in _PARTITIONS for window_id in normalized[partition]
    )
    conflicts = _find_partition_conflicts(records)
    if conflicts:
        raise ValueError(_format_conflicts("window IDs", conflicts))


def validate_intervention_pair_partitions(
    pairs: Iterable[InterventionPair],
) -> None:
    """Reject window or pair identities allocated across pair partitions."""

    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be an iterable of InterventionPair objects")
    try:
        normalized_pairs = tuple(pairs)
    except TypeError as error:
        raise TypeError("pairs must be an iterable of InterventionPair objects") from error
    for index, pair in enumerate(normalized_pairs):
        if not isinstance(pair, InterventionPair):
            raise TypeError(f"pairs[{index}] must be an InterventionPair")

    window_records = (
        (window_id, pair.partition)
        for pair in normalized_pairs
        for window_id in (pair.base_window_id, pair.source_window_id)
    )
    pair_records = ((pair.pair_id, pair.partition) for pair in normalized_pairs)
    window_conflicts = _find_partition_conflicts(window_records)
    pair_conflicts = _find_partition_conflicts(pair_records)
    evidence: list[str] = []
    if window_conflicts:
        evidence.append(_format_conflicts("window IDs", window_conflicts))
    if pair_conflicts:
        evidence.append(_format_conflicts("pair_id values", pair_conflicts))
    if evidence:
        raise ValueError("; ".join(evidence))


def _compute_pair_id(
    *,
    base_window_id: str,
    source_window_id: str,
    concept_name: str,
    regime_relation: RegimeRelation,
) -> str:
    payload = {
        "base_window_id": base_window_id,
        "concept_name": concept_name,
        "regime_relation": str(regime_relation),
        "source_window_id": source_window_id,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_ids(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of IDs, not a bare string")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{field_name} must be an iterable of IDs") from error
    for index, window_id in enumerate(normalized):
        if not isinstance(window_id, str) or not window_id.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
    return normalized


def _find_partition_conflicts(
    records: Iterable[tuple[str, SplitPartition]],
) -> tuple[tuple[str, tuple[SplitPartition, ...]], ...]:
    partitions_by_id: dict[str, set[SplitPartition]] = {}
    for identifier, partition in records:
        partitions_by_id.setdefault(identifier, set()).add(partition)
    return tuple(
        (
            identifier,
            tuple(
                partition for partition in _PARTITIONS if partition in partitions_by_id[identifier]
            ),
        )
        for identifier in sorted(partitions_by_id)
        if len(partitions_by_id[identifier]) > 1
    )


def _format_conflicts(
    label: str,
    conflicts: tuple[tuple[str, tuple[SplitPartition, ...]], ...],
) -> str:
    evidence = ", ".join(
        f"{identifier!r} [{', '.join(partition.value for partition in partitions)}]"
        for identifier, partitions in conflicts[:_EVIDENCE_LIMIT]
    )
    return f"{label} cross partitions (first {_EVIDENCE_LIMIT} sorted IDs): {evidence}"
