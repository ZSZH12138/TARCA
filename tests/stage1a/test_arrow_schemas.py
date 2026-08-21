from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tarca.contracts.arrow_schemas import (
    EFFECTS_SCHEMA,
    INTERVENTION_PAIRS_SCHEMA,
    LOCALIZATION_SCHEMA,
    METRICS_SCHEMA,
    PREDICTIONS_SCHEMA,
    validate_table,
)


def _prediction_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "window_id": "window-0",
                "split": "TRAIN",
                "forecast_time": datetime(2026, 8, 21, tzinfo=UTC),
                "horizon": 0,
                "target": "load",
                "y_true": 1.0,
                "mean": 1.1,
                "scale": 0.2,
            }
        ],
        schema=PREDICTIONS_SCHEMA,
    )


def test_protocol_arrow_schemas_have_exact_common_metadata() -> None:
    schemas = (
        PREDICTIONS_SCHEMA,
        INTERVENTION_PAIRS_SCHEMA,
        EFFECTS_SCHEMA,
        METRICS_SCHEMA,
        LOCALIZATION_SCHEMA,
    )

    for schema in schemas:
        assert schema.metadata is not None
        assert schema.metadata[b"contract_schema_version"] == b"1.0.0"
        assert schema.metadata[b"protocol_id"] == b"TARCA-E2E-STAGE-PROTOCOL-2.0"
        assert schema.metadata[b"artifact_type"]
        validate_table(pa.Table.from_pylist([], schema=schema), schema)

    assert PREDICTIONS_SCHEMA.names == [
        "window_id",
        "split",
        "forecast_time",
        "horizon",
        "target",
        "y_true",
        "mean",
        "scale",
    ]
    assert PREDICTIONS_SCHEMA.field("forecast_time").type == pa.timestamp("ns", tz="UTC")
    assert PREDICTIONS_SCHEMA.field("horizon").type == pa.int32()


def test_predictions_round_trip_with_exact_ipc_and_parquet_schema(tmp_path: Path) -> None:
    table = _prediction_table()
    ipc_path = tmp_path / "predictions.arrow"
    parquet_path = tmp_path / "predictions.parquet"

    with pa.OSFile(str(ipc_path), "wb") as sink, pa.ipc.new_file(
        sink, table.schema
    ) as writer:
        writer.write_table(table)
    pq.write_table(table, parquet_path)

    with pa.memory_map(str(ipc_path), "r") as source:
        ipc_table = pa.ipc.open_file(source).read_all()
    parquet_table = pq.read_table(parquet_path)

    assert validate_table(ipc_table, PREDICTIONS_SCHEMA) is ipc_table
    assert validate_table(parquet_table, PREDICTIONS_SCHEMA) is parquet_table
    assert ipc_table.equals(table)
    assert parquet_table.equals(table)


@pytest.mark.parametrize(
    "schema",
    (
        pa.schema(
            [
                *list(PREDICTIONS_SCHEMA)[:-1],
                pa.field("scale", pa.float32(), nullable=True),
            ],
            metadata=PREDICTIONS_SCHEMA.metadata,
        ),
        PREDICTIONS_SCHEMA.remove_metadata(),
    ),
)
def test_table_validation_rejects_schema_or_metadata_drift(schema: pa.Schema) -> None:
    table = pa.Table.from_pylist([], schema=schema)

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_table(table, PREDICTIONS_SCHEMA)


def test_localization_requires_score_or_cost() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "trace_id": "trace-0",
                "stage": "OT",
                "candidate_id": "candidate-0",
                "parent_candidate_id": None,
                "selected": True,
                "score": None,
                "cost": None,
                "transport_mass": 1.0,
                "runtime_seconds": 0.1,
            }
        ],
        schema=LOCALIZATION_SCHEMA,
    )

    with pytest.raises(ValueError, match="score or cost"):
        validate_table(table, LOCALIZATION_SCHEMA)
