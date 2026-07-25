from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tarca.contracts.arrow_schemas import (
    intervention_pairs_schema,
    metrics_by_regime_schema,
    predictions_schema,
    validate_arrow_schema,
)

SchemaFactory = Callable[[], pa.Schema]

SCHEMA_CASES: tuple[
    tuple[str, SchemaFactory, tuple[tuple[str, pa.DataType, bool], ...]],
    ...,
] = (
    (
        "metrics_by_regime",
        metrics_by_regime_schema,
        (
            ("experiment_id", pa.string(), False),
            ("run_id", pa.string(), False),
            ("split", pa.string(), False),
            ("metric", pa.string(), False),
            ("value", pa.float64(), False),
            ("regime", pa.string(), True),
            ("horizon", pa.int32(), True),
            ("concept", pa.string(), True),
        ),
    ),
    (
        "predictions",
        predictions_schema,
        (
            ("window_id", pa.string(), False),
            ("split", pa.string(), False),
            ("forecast_time", pa.timestamp("us", tz="UTC"), False),
            ("horizon", pa.int32(), False),
            ("target", pa.string(), False),
            ("y_true", pa.float64(), True),
            ("mean", pa.float64(), False),
            ("scale", pa.float64(), True),
        ),
    ),
    (
        "intervention_pairs",
        intervention_pairs_schema,
        (
            ("schema_version", pa.string(), False),
            ("pair_id", pa.string(), False),
            ("partition", pa.string(), False),
            ("base_window_id", pa.string(), False),
            ("source_window_id", pa.string(), False),
            ("concept_name", pa.string(), False),
            ("regime_relation", pa.string(), False),
            ("matching_distance", pa.float64(), False),
            ("concept_delta", pa.float64(), False),
        ),
    ),
)


def _decoded_metadata(schema: pa.Schema) -> dict[str, str]:
    assert schema.metadata is not None
    return {key.decode("utf-8"): value.decode("utf-8") for key, value in schema.metadata.items()}


def _minimum_table(schema_name: str, schema: pa.Schema) -> pa.Table:
    records: dict[str, list[object]]
    if schema_name == "metrics_by_regime":
        records = {
            "experiment_id": ["experiment-1"],
            "run_id": ["run-1"],
            "split": ["test"],
            "metric": ["mae"],
            "value": [1.25],
            "regime": [None],
            "horizon": [None],
            "concept": [None],
        }
    elif schema_name == "predictions":
        records = {
            "window_id": ["window-1"],
            "split": ["test"],
            "forecast_time": [datetime(2026, 1, 2, 3, 4, tzinfo=UTC)],
            "horizon": [1],
            "target": ["load"],
            "y_true": [None],
            "mean": [1.5],
            "scale": [None],
        }
    else:
        records = {
            "schema_version": ["1.0.0"],
            "pair_id": [f"sha256:{'a' * 64}"],
            "partition": ["train"],
            "base_window_id": ["base-window"],
            "source_window_id": ["source-window"],
            "concept_name": ["temperature"],
            "regime_relation": ["cross"],
            "matching_distance": [0.25],
            "concept_delta": [-1.5],
        }
    return pa.Table.from_pydict(records, schema=schema)


@pytest.mark.parametrize(("schema_name", "factory", "expected_fields"), SCHEMA_CASES)
def test_arrow_schema_has_exact_order_types_nullability_and_metadata(
    schema_name: str,
    factory: SchemaFactory,
    expected_fields: tuple[tuple[str, pa.DataType, bool], ...],
) -> None:
    schema = factory()

    assert isinstance(schema, pa.Schema)
    assert tuple((field.name, field.type, field.nullable) for field in schema) == expected_fields
    assert _decoded_metadata(schema) == {
        "contract_schema_version": "1.0.0",
        "schema_name": schema_name,
    }


def test_predictions_schema_uses_microsecond_utc_forecast_timestamps() -> None:
    forecast_time = predictions_schema().field("forecast_time")

    assert pa.types.is_timestamp(forecast_time.type)
    assert forecast_time.type.unit == "us"
    assert forecast_time.type.tz == "UTC"


@pytest.mark.parametrize(("schema_name", "factory", "_expected_fields"), SCHEMA_CASES)
def test_minimum_legal_table_survives_strict_parquet_round_trip(
    tmp_path: Path,
    schema_name: str,
    factory: SchemaFactory,
    _expected_fields: tuple[tuple[str, pa.DataType, bool], ...],
) -> None:
    expected_schema = factory()
    table = _minimum_table(schema_name, expected_schema)
    parquet_path = tmp_path / f"{schema_name}.parquet"

    pq.write_table(table, parquet_path)
    read_schema = pq.read_schema(parquet_path)
    read_table = pq.read_table(parquet_path)

    assert validate_arrow_schema(read_schema, expected_schema, schema_name=schema_name) is None
    assert (
        validate_arrow_schema(read_table.schema, expected_schema, schema_name=schema_name) is None
    )
    assert read_table.schema.equals(expected_schema, check_metadata=True)
    assert read_table.equals(table)
    assert _decoded_metadata(read_table.schema) == {
        "contract_schema_version": "1.0.0",
        "schema_name": schema_name,
    }

    if schema_name == "predictions":
        timestamp_type = read_table.schema.field("forecast_time").type
        assert timestamp_type == pa.timestamp("us", tz="UTC")
        assert read_table["forecast_time"][0].as_py() == datetime(
            2026,
            1,
            2,
            3,
            4,
            tzinfo=UTC,
        )


def _schema_with_fields(
    schema: pa.Schema,
    fields: tuple[pa.Field, ...],
) -> pa.Schema:
    return pa.schema(fields, metadata=schema.metadata)


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "name",
        "type",
        "nullability",
        "metadata",
    ],
)
def test_validate_arrow_schema_rejects_each_independent_contract_mutation(
    mutation: str,
) -> None:
    expected = predictions_schema()
    fields = tuple(expected)

    if mutation == "order":
        actual = _schema_with_fields(expected, (fields[1], fields[0], *fields[2:]))
        expected_error = r"predictions.*field"
    elif mutation == "name":
        renamed = pa.field(
            "renamed_window_id",
            fields[0].type,
            nullable=fields[0].nullable,
        )
        actual = _schema_with_fields(expected, (renamed, *fields[1:]))
        expected_error = r"predictions.*field"
    elif mutation == "type":
        wrong_type = pa.field(
            fields[0].name,
            pa.int64(),
            nullable=fields[0].nullable,
        )
        actual = _schema_with_fields(expected, (wrong_type, *fields[1:]))
        expected_error = r"predictions.*field"
    elif mutation == "nullability":
        wrong_nullability = pa.field(
            fields[0].name,
            fields[0].type,
            nullable=not fields[0].nullable,
        )
        actual = _schema_with_fields(expected, (wrong_nullability, *fields[1:]))
        expected_error = r"predictions.*field"
    else:
        actual = expected.with_metadata(
            {
                b"contract_schema_version": b"1.0.0",
                b"schema_name": b"wrong-name",
            }
        )
        expected_error = r"predictions.*metadata"

    with pytest.raises(ValueError, match=expected_error):
        validate_arrow_schema(actual, expected, schema_name="predictions")


def test_validate_arrow_schema_rejects_field_count_and_field_metadata_changes() -> None:
    expected = metrics_by_regime_schema()
    fewer_fields = _schema_with_fields(expected, tuple(expected)[:-1])
    first_with_metadata = expected.field(0).with_metadata({b"unit": b"identifier"})
    field_metadata_changed = _schema_with_fields(
        expected,
        (first_with_metadata, *tuple(expected)[1:]),
    )

    with pytest.raises(ValueError, match=r"metrics_by_regime.*field count"):
        validate_arrow_schema(
            fewer_fields,
            expected,
            schema_name="metrics_by_regime",
        )
    with pytest.raises(ValueError, match=r"metrics_by_regime.*field.*metadata"):
        validate_arrow_schema(
            field_metadata_changed,
            expected,
            schema_name="metrics_by_regime",
        )


def test_validate_arrow_schema_rejects_non_schema_inputs_with_schema_context() -> None:
    expected = intervention_pairs_schema()

    with pytest.raises(ValueError, match=r"intervention_pairs.*pyarrow.Schema"):
        validate_arrow_schema(
            ["schema_version", "pair_id"],
            expected,
            schema_name="intervention_pairs",
        )
