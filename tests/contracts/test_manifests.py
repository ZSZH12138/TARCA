from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tarca.contracts.manifests import (
    DataManifest,
    DataSplitSummary,
    InterventionPair,
    MetricRecord,
    RunManifest,
    StrictContractModel,
    WindowContractSummary,
)
from tarca.contracts.types import RegimeRelation, RunStatus, SplitPartition
from tarca.contracts.version import CONTRACT_SCHEMA_VERSION

SHA_A = f"sha256:{'a' * 64}"
SHA_B = f"sha256:{'b' * 64}"
SHA_C = f"sha256:{'c' * 64}"
SHA_D = f"sha256:{'d' * 64}"
CREATED_AT = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
EXPECTED_PAIR_ID = "sha256:47dc7e968632bf617b5a0421b590bf1b2172e7369426ef7e0854a7bda6edc849"


def _split_summary(
    partition: SplitPartition = SplitPartition.TRAIN,
    *,
    split_hash: str = SHA_A,
    count: int = 3,
) -> DataSplitSummary:
    return DataSplitSummary(partition=partition, split_hash=split_hash, count=count)


def _window_contract(**overrides: object) -> WindowContractSummary:
    values: dict[str, object] = {
        "history_length": 24,
        "horizon": 6,
        "input_feature_names": ("load", "temperature"),
        "target_names": ("load",),
        "observed_covariate_names": ("temperature",),
        "known_future_covariate_names": ("hour",),
        "timezone": "UTC",
        "missingness_protocol": "boolean masks; missing values filled with zero",
    }
    return WindowContractSummary(**{**values, **overrides})


def _manifest_splits() -> tuple[DataSplitSummary, ...]:
    return (
        _split_summary(SplitPartition.TRAIN, split_hash=SHA_A, count=10),
        _split_summary(SplitPartition.VALIDATION, split_hash=SHA_B, count=4),
        _split_summary(SplitPartition.TEST, split_hash=SHA_C, count=5),
    )


def _data_manifest(**overrides: object) -> DataManifest:
    values: dict[str, object] = {
        "dataset_name": "grid-load",
        "dataset_version": "2026.01",
        "dataset_hash": SHA_D,
        "splits": _manifest_splits(),
        "window_contract": _window_contract(),
        "source_description": "Synthetic fixture metadata only",
        "created_at": CREATED_AT,
    }
    return DataManifest(**{**values, **overrides})


def _pair(**overrides: object) -> InterventionPair:
    values: dict[str, object] = {
        "partition": SplitPartition.TRAIN,
        "base_window_id": "base-window",
        "source_window_id": "source-window",
        "concept_name": "temperature",
        "regime_relation": RegimeRelation.CROSS,
        "matching_distance": 0.25,
        "concept_delta": -1.5,
    }
    return InterventionPair.build(**{**values, **overrides})


def _run_manifest(**overrides: object) -> RunManifest:
    values: dict[str, object] = {
        "experiment_id": "experiment-1",
        "run_id": "run-1",
        "config_hash": SHA_A,
        "data_hash": SHA_D,
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "created_at": CREATED_AT,
        "status": RunStatus.PENDING,
    }
    return RunManifest(**{**values, **overrides})


def _metric(**overrides: object) -> MetricRecord:
    values: dict[str, object] = {
        "experiment_id": "experiment-1",
        "run_id": "run-1",
        "split": SplitPartition.TEST,
        "metric": "mae",
        "value": 1.25,
        "regime": None,
        "horizon": None,
        "concept": None,
    }
    return MetricRecord(**{**values, **overrides})


def _persistent_models() -> tuple[StrictContractModel, ...]:
    return (
        _split_summary(),
        _window_contract(),
        _pair(),
        _data_manifest(),
        _run_manifest(),
        _metric(),
    )


def test_models_publish_the_stable_field_surfaces() -> None:
    assert tuple(DataSplitSummary.model_fields) == (
        "schema_version",
        "partition",
        "split_hash",
        "count",
    )
    assert tuple(WindowContractSummary.model_fields) == (
        "schema_version",
        "history_length",
        "horizon",
        "input_feature_names",
        "target_names",
        "observed_covariate_names",
        "known_future_covariate_names",
        "timezone",
        "missingness_protocol",
    )
    assert tuple(InterventionPair.model_fields) == (
        "schema_version",
        "pair_id",
        "partition",
        "base_window_id",
        "source_window_id",
        "concept_name",
        "regime_relation",
        "matching_distance",
        "concept_delta",
    )
    assert tuple(DataManifest.model_fields) == (
        "schema_version",
        "dataset_name",
        "dataset_version",
        "dataset_hash",
        "splits",
        "window_contract",
        "source_description",
        "created_at",
    )
    assert tuple(RunManifest.model_fields) == (
        "schema_version",
        "experiment_id",
        "run_id",
        "config_hash",
        "data_hash",
        "git_commit",
        "created_at",
        "status",
    )
    assert tuple(MetricRecord.model_fields) == (
        "schema_version",
        "experiment_id",
        "run_id",
        "split",
        "metric",
        "value",
        "regime",
        "horizon",
        "concept",
    )


def test_every_persistent_model_uses_the_shared_strict_base_and_version_schema() -> None:
    for model in _persistent_models():
        assert isinstance(model, StrictContractModel)
        assert model.schema_version == CONTRACT_SCHEMA_VERSION
        version_schema = type(model).model_json_schema()["properties"]["schema_version"]
        assert version_schema["const"] == CONTRACT_SCHEMA_VERSION


def test_four_public_models_support_pydantic_json_round_trip() -> None:
    for model in (_pair(), _data_manifest(), _run_manifest(), _metric()):
        assert type(model).model_validate_json(model.model_dump_json()) == model


def test_every_persistent_model_rejects_a_wrong_schema_version() -> None:
    for model in _persistent_models():
        payload = model.model_dump(mode="json")
        payload["schema_version"] = "2.0.0"
        with pytest.raises(ValidationError, match="schema_version"):
            type(model).model_validate_json(json.dumps(payload))


def test_shared_base_rejects_extra_fields_and_field_reassignment() -> None:
    for model in _persistent_models():
        payload = model.model_dump(mode="json")
        payload["unexpected"] = "value"
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(model).model_validate_json(json.dumps(payload))
        with pytest.raises(ValidationError, match="frozen_instance"):
            model.schema_version = CONTRACT_SCHEMA_VERSION


def test_python_mode_does_not_coerce_numbers_enums_or_collections() -> None:
    with pytest.raises(ValidationError, match="partition"):
        _split_summary(partition="train")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="history_length"):
        _window_contract(history_length="24")
    with pytest.raises(ValidationError, match="splits"):
        _data_manifest(splits=list(_manifest_splits()))
    with pytest.raises(ValidationError, match="regime_relation"):
        _pair(regime_relation="cross")
    with pytest.raises(ValidationError, match="status"):
        _run_manifest(status="pending")
    with pytest.raises(ValidationError, match="value"):
        _metric(value="1.25")


@pytest.mark.parametrize(
    "invalid_hash",
    [
        "",
        f"sha256:{'a' * 63}",
        f"sha256:{'a' * 65}",
        f"sha256:{'A' * 64}",
        f"md5:{'a' * 64}",
        f"sha256:{'g' * 64}",
    ],
)
def test_hash_fields_require_the_exact_sha256_format(invalid_hash: str) -> None:
    with pytest.raises(ValidationError, match="split_hash"):
        _split_summary(split_hash=invalid_hash)
    with pytest.raises(ValidationError, match="dataset_hash"):
        _data_manifest(dataset_hash=invalid_hash)
    with pytest.raises(ValidationError, match="config_hash"):
        _run_manifest(config_hash=invalid_hash)


@pytest.mark.parametrize(
    "git_commit",
    [
        "",
        "a" * 39,
        "a" * 41,
        "A" * 40,
        "g" * 40,
        "sha256:" + "a" * 40,
    ],
)
def test_run_manifest_requires_a_lowercase_40_hex_git_commit(git_commit: str) -> None:
    with pytest.raises(ValidationError, match="git_commit"):
        _run_manifest(git_commit=git_commit)


@pytest.mark.parametrize(
    "created_at",
    [
        datetime(2026, 1, 2, 3, 4),
        datetime(2026, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=8))),
    ],
)
def test_manifest_datetimes_must_already_be_utc(created_at: datetime) -> None:
    with pytest.raises(ValidationError, match="created_at"):
        _data_manifest(created_at=created_at)
    with pytest.raises(ValidationError, match="created_at"):
        _run_manifest(created_at=created_at)


@pytest.mark.parametrize("value", ["", "   "])
def test_required_identifiers_and_descriptions_are_non_empty(value: str) -> None:
    for field_name in ("dataset_name", "dataset_version", "source_description"):
        with pytest.raises(ValidationError, match=field_name):
            _data_manifest(**{field_name: value})
    for field_name in ("experiment_id", "run_id"):
        with pytest.raises(ValidationError, match=field_name):
            _run_manifest(**{field_name: value})
        with pytest.raises(ValidationError, match=field_name):
            _metric(**{field_name: value})
    with pytest.raises(ValidationError, match="metric"):
        _metric(metric=value)
    with pytest.raises(ValidationError, match="regime"):
        _metric(regime=value)
    with pytest.raises(ValidationError, match="concept"):
        _metric(concept=value)


@pytest.mark.parametrize("count", [-1, True, 1.5, "1"])
def test_split_counts_are_strict_non_negative_integers(count: object) -> None:
    with pytest.raises(ValidationError, match="count"):
        _split_summary(count=count)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["history_length", "horizon"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_window_lengths_are_strict_positive_integers(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field_name):
        _window_contract(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "input_feature_names",
        "target_names",
        "observed_covariate_names",
        "known_future_covariate_names",
    ],
)
@pytest.mark.parametrize("names", [("",), ("   ",), ("duplicate", "duplicate"), ["name"]])
def test_window_name_tuples_are_strict_non_empty_and_unique(
    field_name: str,
    names: object,
) -> None:
    with pytest.raises(ValidationError, match=field_name):
        _window_contract(**{field_name: names})


def test_window_summary_requires_literal_utc_and_a_missingness_protocol() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _window_contract(timezone="Asia/Shanghai")
    with pytest.raises(ValidationError, match="missingness_protocol"):
        _window_contract(missingness_protocol="   ")


def test_data_manifest_requires_exactly_one_summary_per_partition() -> None:
    missing_test = _manifest_splits()[:2]
    with pytest.raises(ValidationError, match="splits"):
        _data_manifest(splits=missing_test)

    duplicate_train = (
        _split_summary(SplitPartition.TRAIN, split_hash=SHA_A),
        _split_summary(SplitPartition.TRAIN, split_hash=SHA_B),
        _split_summary(SplitPartition.TEST, split_hash=SHA_C),
    )
    with pytest.raises(ValidationError, match="splits"):
        _data_manifest(splits=duplicate_train)


def test_pair_build_uses_the_canonical_identity_payload() -> None:
    pair = _pair()
    assert pair.pair_id == EXPECTED_PAIR_ID


def test_pair_identity_excludes_diagnostics_and_partition() -> None:
    baseline = _pair()
    changed_diagnostics = _pair(matching_distance=99.5, concept_delta=42.0)
    changed_partition = _pair(partition=SplitPartition.TEST)
    assert changed_diagnostics.pair_id == baseline.pair_id
    assert changed_partition.pair_id == baseline.pair_id


def test_pair_direct_construction_recomputes_and_rejects_an_incorrect_id() -> None:
    payload = _pair().model_dump()
    payload["pair_id"] = SHA_A
    with pytest.raises(ValidationError, match="pair_id"):
        InterventionPair(**payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_window_id": ""},
        {"source_window_id": "   "},
        {"base_window_id": "same", "source_window_id": "same"},
        {"concept_name": ""},
    ],
)
def test_pair_rejects_bad_identifiers(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _pair(**overrides)


@pytest.mark.parametrize("matching_distance", [-0.01, math.inf, -math.inf, math.nan])
def test_pair_matching_distance_is_finite_and_non_negative(matching_distance: float) -> None:
    with pytest.raises(ValidationError, match="matching_distance"):
        _pair(matching_distance=matching_distance)


@pytest.mark.parametrize("concept_delta", [math.inf, -math.inf, math.nan])
def test_pair_concept_delta_is_finite(concept_delta: float) -> None:
    with pytest.raises(ValidationError, match="concept_delta"):
        _pair(concept_delta=concept_delta)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_metric_value_is_finite(value: float) -> None:
    with pytest.raises(ValidationError, match="value"):
        _metric(value=value)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "1"])
def test_metric_horizon_is_a_nullable_strict_positive_integer(horizon: object) -> None:
    with pytest.raises(ValidationError, match="horizon"):
        _metric(horizon=horizon)


def test_metric_optional_dimensions_accept_non_empty_values() -> None:
    metric = _metric(regime="winter", horizon=1, concept="temperature")
    assert (metric.regime, metric.horizon, metric.concept) == ("winter", 1, "temperature")
