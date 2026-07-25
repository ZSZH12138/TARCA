"""Strict Arrow schemas for TARCA's persisted tabular artifacts."""

from __future__ import annotations

import pyarrow as pa

from .version import CONTRACT_SCHEMA_VERSION


def metrics_by_regime_schema() -> pa.Schema:
    """Return the exact long-format metrics-by-regime schema."""

    return pa.schema(
        (
            pa.field("experiment_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("regime", pa.string(), nullable=True),
            pa.field("horizon", pa.int32(), nullable=True),
            pa.field("concept", pa.string(), nullable=True),
        ),
        metadata=_schema_metadata("metrics_by_regime"),
    )


def predictions_schema() -> pa.Schema:
    """Return the exact long-format forecast predictions schema."""

    return pa.schema(
        (
            pa.field("window_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field(
                "forecast_time",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field("horizon", pa.int32(), nullable=False),
            pa.field("target", pa.string(), nullable=False),
            pa.field("y_true", pa.float64(), nullable=True),
            pa.field("mean", pa.float64(), nullable=False),
            pa.field("scale", pa.float64(), nullable=True),
        ),
        metadata=_schema_metadata("predictions"),
    )


def intervention_pairs_schema() -> pa.Schema:
    """Return the exact persisted representation of ``InterventionPair``."""

    return pa.schema(
        (
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("pair_id", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("base_window_id", pa.string(), nullable=False),
            pa.field("source_window_id", pa.string(), nullable=False),
            pa.field("concept_name", pa.string(), nullable=False),
            pa.field("regime_relation", pa.string(), nullable=False),
            pa.field("matching_distance", pa.float64(), nullable=False),
            pa.field("concept_delta", pa.float64(), nullable=False),
        ),
        metadata=_schema_metadata("intervention_pairs"),
    )


def validate_arrow_schema(
    actual: pa.Schema,
    expected: pa.Schema,
    *,
    schema_name: str,
) -> None:
    """Reject any field, ordering, nullability, or metadata mismatch."""

    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError("schema_name must be a non-empty string")
    if not isinstance(actual, pa.Schema):
        raise ValueError(f"{schema_name} schema mismatch: actual must be a pyarrow.Schema")
    if not isinstance(expected, pa.Schema):
        raise ValueError(f"{schema_name} schema mismatch: expected must be a pyarrow.Schema")
    if actual.equals(expected, check_metadata=True):
        return

    differences = _describe_schema_differences(actual, expected)
    detail = "; ".join(differences) or "schemas are not strictly equal"
    raise ValueError(f"{schema_name} schema mismatch: {detail}")


def _schema_metadata(schema_name: str) -> dict[bytes, bytes]:
    return {
        b"contract_schema_version": CONTRACT_SCHEMA_VERSION.encode("utf-8"),
        b"schema_name": schema_name.encode("utf-8"),
    }


def _describe_schema_differences(
    actual: pa.Schema,
    expected: pa.Schema,
) -> tuple[str, ...]:
    differences: list[str] = []
    if len(actual) != len(expected):
        differences.append(f"field count expected {len(expected)}, got {len(actual)}")

    for index, (actual_field, expected_field) in enumerate(zip(actual, expected, strict=False)):
        if actual_field.name != expected_field.name:
            differences.append(
                f"field {index} name expected {expected_field.name!r}, got {actual_field.name!r}"
            )
        if actual_field.type != expected_field.type:
            differences.append(
                f"field {index} ({expected_field.name!r}) type "
                f"expected {expected_field.type}, got {actual_field.type}"
            )
        if actual_field.nullable != expected_field.nullable:
            differences.append(
                f"field {index} ({expected_field.name!r}) nullable "
                f"expected {expected_field.nullable}, got {actual_field.nullable}"
            )
        if actual_field.metadata != expected_field.metadata:
            differences.append(
                f"field {index} ({expected_field.name!r}) metadata "
                f"expected {expected_field.metadata!r}, got {actual_field.metadata!r}"
            )

    if actual.metadata != expected.metadata:
        differences.append(
            f"schema metadata expected {expected.metadata!r}, got {actual.metadata!r}"
        )
    return tuple(differences)
