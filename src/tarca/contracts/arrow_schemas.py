from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]  # PyArrow 25 ships no py.typed marker.
import pyarrow.compute as pc  # type: ignore[import-untyped]  # See import above.

from .base import CONTRACT_SCHEMA_VERSION, PROTOCOL_ID


def _metadata(artifact_type: str) -> dict[bytes, bytes]:
    return {
        b"contract_schema_version": CONTRACT_SCHEMA_VERSION.encode("ascii"),
        b"protocol_id": PROTOCOL_ID.encode("ascii"),
        b"artifact_type": artifact_type.encode("ascii"),
    }


PREDICTIONS_SCHEMA = pa.schema(
    [
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("forecast_time", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("horizon", pa.int32(), nullable=False),
        pa.field("target", pa.string(), nullable=False),
        pa.field("y_true", pa.float64(), nullable=True),
        pa.field("mean", pa.float64(), nullable=False),
        pa.field("scale", pa.float64(), nullable=True),
    ],
    metadata=_metadata("PREDICTIONS"),
)

INTERVENTION_PAIRS_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
        pa.field("base_window_id", pa.string(), nullable=False),
        pa.field("source_window_id", pa.string(), nullable=False),
        pa.field("concept_name", pa.string(), nullable=False),
        pa.field("regime_relation", pa.string(), nullable=False),
        pa.field("matching_distance", pa.float64(), nullable=False),
        pa.field("concept_delta", pa.float64(), nullable=False),
    ],
    metadata=_metadata("INTERVENTION_PAIRS"),
)

EFFECTS_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("concept_name", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=True),
        pa.field("candidate_id", pa.string(), nullable=True),
        pa.field("horizon", pa.int32(), nullable=False),
        pa.field("target", pa.string(), nullable=False),
        pa.field("effect_component", pa.string(), nullable=False),
        pa.field("quantile_level", pa.float64(), nullable=True),
        pa.field("value", pa.float64(), nullable=False),
    ],
    metadata=_metadata("EFFECTS"),
)

METRICS_SCHEMA = pa.schema(
    [
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("regime", pa.string(), nullable=True),
        pa.field("horizon", pa.int32(), nullable=True),
        pa.field("concept", pa.string(), nullable=True),
    ],
    metadata=_metadata("METRICS"),
)

LOCALIZATION_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("parent_candidate_id", pa.string(), nullable=True),
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
        pa.field("cost", pa.float64(), nullable=True),
        pa.field("transport_mass", pa.float64(), nullable=True),
        pa.field("runtime_seconds", pa.float64(), nullable=False),
    ],
    metadata=_metadata("LOCALIZATION"),
)


def _validate_finite_columns(table: pa.Table) -> None:
    for field in table.schema:
        if pa.types.is_floating(field.type):
            finite = pc.fill_null(pc.is_finite(table[field.name]), True)
            if pc.all(finite).as_py() is False:
                raise ValueError(f"{field.name} must contain only finite values")


def _validate_prediction_values(table: pa.Table) -> None:
    if table.schema.metadata != PREDICTIONS_SCHEMA.metadata:
        return
    nonpositive = pc.fill_null(pc.less_equal(table["scale"], 0.0), False)
    if bool(pc.any(nonpositive).as_py()):
        raise ValueError("prediction scale must be strictly positive when present")


def _validate_localization_values(table: pa.Table) -> None:
    if table.schema.metadata != LOCALIZATION_SCHEMA.metadata:
        return
    missing_both = pc.and_(pc.is_null(table["score"]), pc.is_null(table["cost"]))
    if bool(pc.any(missing_both).as_py()):
        raise ValueError("localization rows require score or cost")


def validate_table(table: pa.Table, expected_schema: pa.Schema) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise TypeError("value must be a pyarrow.Table")
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise ValueError("Arrow schema mismatch")
    _validate_finite_columns(table)
    _validate_prediction_values(table)
    _validate_localization_values(table)
    return table
