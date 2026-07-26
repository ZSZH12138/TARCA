"""Independent persisted synthetic-dataset validation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import yaml

from tarca.contracts import WindowBatch

from ._validation_core import ValidationIssue, _add
from ._validation_integrity import _arrow_metadata, _canonical, _digest
from .dataset_builder import PersistedSyntheticDataset, SyntheticDataset

_SPLITS = ("train", "validation", "test_seen_regime", "test_unseen_regime")
_FILES = frozenset(
    {
        "config_resolved.yaml",
        "manifest.json",
        "checksums.json",
        "truth.npz",
        *(f"windows_{name}.arrow" for name in _SPLITS),
        "normalization.json",
    }
)
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TENSOR_FIELDS = (
    ("x", "float"),
    ("y", "float"),
    ("observed_covariates", "float"),
    ("known_future_covariates", "float"),
    ("x_observed_mask", "bool"),
    ("y_observed_mask", "bool"),
    ("observed_covariates_mask", "bool"),
    ("known_future_covariates_mask", "bool"),
)
_NAME_FIELDS = (
    "input_feature_names",
    "target_names",
    "observed_covariate_names",
    "known_future_covariate_names",
)
_TIME_FIELDS = ("feature_start", "feature_end", "prediction_start", "label_end")
ReparseProbe = Callable[[Path], bool]


def _has_reparse(path: Path, is_reparse: ReparseProbe) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() and is_reparse(current):
            return True
    return False


def _trusted(
    path: Path,
    name: str,
    checksums: dict[str, str],
    is_reparse: ReparseProbe,
) -> bool:
    try:
        return (
            name in checksums
            and path.is_file()
            and not is_reparse(path)
            and _digest(path.read_bytes()) == checksums[name]
        )
    except OSError:
        return False


def _arrow_schema(metadata: dict[str, object]) -> pa.Schema:
    scalar = {"float": pa.float64(), "bool": pa.bool_()}
    fields = [pa.field("window_id", pa.string(), nullable=False)]
    fields.extend(
        pa.field(
            name,
            pa.large_list(pa.large_list(scalar[kind])),
            nullable=name != "x",
        )
        for name, kind in _TENSOR_FIELDS
    )
    fields.append(pa.field("regime", pa.int64()))
    fields.extend(
        pa.field(name, pa.large_list(pa.string()), nullable=False) for name in _NAME_FIELDS
    )
    timestamp = pa.timestamp("us", tz="UTC")
    fields.extend(pa.field(name, timestamp, nullable=False) for name in _TIME_FIELDS)
    fields.extend(
        (
            pa.field("forecast_time", pa.large_list(timestamp), nullable=False),
            pa.field("metadata_json", pa.string(), nullable=False),
        )
    )
    encoded = {
        key.encode(): _canonical(value) if isinstance(value, list) else str(value).encode()
        for key, value in metadata.items()
    }
    return pa.schema(fields, metadata=encoded)


def _nested_array(tensor: torch.Tensor, scalar_type: pa.DataType) -> pa.Array:
    values = np.ascontiguousarray(tensor.detach().cpu().numpy())
    batch, rows, columns = values.shape
    flat = pa.array(values.reshape(-1), type=scalar_type)
    inner = pa.LargeListArray.from_arrays(
        pa.array(np.arange(0, batch * rows * columns + 1, columns), type=pa.int64()),
        flat,
    )
    return pa.LargeListArray.from_arrays(
        pa.array(np.arange(0, batch * rows + 1, rows), type=pa.int64()),
        inner,
    )


def canonical_split_hash(batch: WindowBatch, physical_split: str) -> str:
    """Independently serialize one batch under the fixed private Arrow schema."""

    metadata = _arrow_metadata(
        physical_split,
        int(batch.x.shape[1]),
        int(batch.x.shape[2]),
        len(batch.forecast_time[0]),
    )
    schema, count = _arrow_schema(metadata), len(batch.window_id)
    scalar = {"float": pa.float64(), "bool": pa.bool_()}
    arrays: list[pa.Array] = [pa.array(batch.window_id, type=pa.string())]
    for name, kind in _TENSOR_FIELDS:
        tensor = getattr(batch, name)
        arrays.append(
            pa.nulls(count, type=pa.large_list(pa.large_list(scalar[kind])))
            if tensor is None
            else _nested_array(tensor, scalar[kind])
        )
    arrays.append(
        pa.nulls(count, type=pa.int64())
        if batch.regime is None
        else pa.array(batch.regime.detach().cpu().numpy(), type=pa.int64())
    )
    arrays.extend(
        pa.array([list(getattr(batch, name))] * count, type=pa.large_list(pa.string()))
        for name in _NAME_FIELDS
    )
    timestamp = pa.timestamp("us", tz="UTC")
    arrays.extend(pa.array(getattr(batch, name), type=timestamp) for name in _TIME_FIELDS)
    horizon = len(batch.forecast_time[0])
    forecast = pa.array(
        (value for row in batch.forecast_time for value in row),
        type=timestamp,
    )
    arrays.append(
        pa.LargeListArray.from_arrays(
            pa.array(np.arange(0, count * horizon + 1, horizon), type=pa.int64()),
            forecast,
        )
    )
    arrays.append(pa.array([_canonical(metadata).decode()] * count, type=pa.string()))
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(
        metadata_version=pa.ipc.MetadataVersion.V5,
        compression=None,
        use_threads=False,
    )
    with pa.ipc.new_file(sink, schema, options=options) as writer:
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
    return _digest(sink.getvalue().to_pybytes())


def _validate_nested_column(
    table: pa.Table,
    name: str,
    expected: torch.Tensor | None,
) -> None:
    column = table[name].combine_chunks()
    if expected is None:
        if column.null_count != len(column):
            raise ValueError(f"{name}: expected an all-null optional tensor")
        return
    if column.null_count:
        raise ValueError(f"{name}: null tensor rows are forbidden")
    rows, columns = expected.shape[1:]
    inner = column.values
    if not np.array_equal(
        column.offsets.to_numpy(zero_copy_only=False),
        np.arange(0, (len(column) + 1) * rows, rows),
    ):
        raise ValueError(f"{name}: outer offsets differ")
    if not np.array_equal(
        inner.offsets.to_numpy(zero_copy_only=False),
        np.arange(0, (len(inner) + 1) * columns, columns),
    ):
        raise ValueError(f"{name}: inner offsets differ")
    actual = np.asarray(inner.values.to_numpy(zero_copy_only=False))
    expected_values = expected.detach().cpu().numpy().reshape(-1)
    if actual.dtype != expected_values.dtype or not np.array_equal(actual, expected_values):
        raise ValueError(f"{name}: tensor values differ")


def _validate_arrow_table(
    table: pa.Table,
    batch: WindowBatch,
    physical_split: str,
) -> None:
    if len(table) == 0:
        raise ValueError("Arrow table: expected at least one row")
    metadata = _arrow_metadata(
        physical_split,
        int(batch.x.shape[1]),
        int(batch.x.shape[2]),
        len(batch.forecast_time[0]),
    )
    if table.schema != _arrow_schema(metadata):
        raise ValueError("Arrow schema: exact private schema or metadata mismatch")
    if tuple(table["window_id"].to_pylist()) != batch.window_id:
        raise ValueError("window_id: rows differ")
    for name, _kind in _TENSOR_FIELDS:
        _validate_nested_column(table, name, getattr(batch, name))
    regime = table["regime"].combine_chunks()
    if batch.regime is None:
        if regime.null_count != len(regime):
            raise ValueError("regime: expected all-null column")
    elif regime.null_count or not np.array_equal(
        regime.to_numpy(zero_copy_only=False),
        batch.regime.detach().cpu().numpy(),
    ):
        raise ValueError("regime: values differ")
    for name in _NAME_FIELDS:
        if table[name].to_pylist() != [list(getattr(batch, name))] * len(table):
            raise ValueError(f"{name}: rows differ")
    for name in _TIME_FIELDS:
        if tuple(table[name].to_pylist()) != getattr(batch, name):
            raise ValueError(f"{name}: rows differ")
    if tuple(tuple(row) for row in table["forecast_time"].to_pylist()) != batch.forecast_time:
        raise ValueError("forecast_time: rows differ")
    if table["metadata_json"].to_pylist() != [_canonical(metadata).decode()] * len(table):
        raise ValueError("metadata_json: rows differ")


def _validate_scalar_payloads(
    dataset: SyntheticDataset,
    root: Path,
    checksums: dict[str, str],
    is_reparse: ReparseProbe,
    issues: list[ValidationIssue],
) -> None:
    expected_manifest = {
        "data_manifest": dataset.data_manifest.model_dump(mode="json"),
        "synthetic_provenance": dataset.synthetic_provenance.model_dump(mode="json"),
    }
    payloads = (
        (
            "config_resolved.yaml",
            "config",
            lambda text: yaml.safe_load(text),
            dataset.config.model_dump(mode="json"),
        ),
        ("manifest.json", "manifest", json.loads, expected_manifest),
        (
            "normalization.json",
            "normalization",
            json.loads,
            dataset.normalization.model_dump(mode="json"),
        ),
    )
    for filename, code, loader, expected in payloads:
        path = root / filename
        if not _trusted(path, filename, checksums, is_reparse):
            continue
        try:
            if loader(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"{code} differs")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            yaml.YAMLError,
            ValueError,
        ) as error:
            _add(issues, f"persistence.{code}", f"file.{filename}", str(error))


def _validate_truth_payload(
    dataset: SyntheticDataset,
    root: Path,
    checksums: dict[str, str],
    is_reparse: ReparseProbe,
    issues: list[ValidationIssue],
) -> None:
    path = root / "truth.npz"
    if not _trusted(path, path.name, checksums, is_reparse):
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(dataset.truth):
                raise ValueError("truth key set differs")
            for name in archive.files:
                value, expected = archive[name], dataset.truth[name]
                if (
                    value.dtype != expected.dtype
                    or value.shape != expected.shape
                    or value.tobytes() != expected.tobytes()
                ):
                    raise ValueError(f"truth.{name}: value differs")
    except (OSError, ValueError, EOFError) as error:
        _add(issues, "persistence.npz", f"file.{path.name}", str(error))


def _validate_arrow_payloads(
    dataset: SyntheticDataset,
    root: Path,
    checksums: dict[str, str],
    is_reparse: ReparseProbe,
    issues: list[ValidationIssue],
) -> None:
    for split in dataset.physical_splits:
        path = root / f"windows_{split.name}.arrow"
        if not _trusted(path, path.name, checksums, is_reparse):
            continue
        try:
            data = path.read_bytes()
            reader = pa.ipc.open_file(pa.BufferReader(data))
            if reader.num_record_batches != 1:
                raise ValueError("expected exactly one record batch")
            _validate_arrow_table(reader.read_all(), split.batch, split.name)
            if _digest(data) != split.split_hash:
                raise ValueError("Arrow split hash differs")
        except (
            OSError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            pa.ArrowInvalid,
        ) as error:
            _add(issues, "persistence.arrow", f"file.{path.name}", str(error))


def validate_persisted(
    dataset: SyntheticDataset,
    persisted: PersistedSyntheticDataset,
    is_reparse: ReparseProbe,
    issues: list[ValidationIssue],
) -> None:
    """Validate exact files, reparse safety, checksums, and independent round-trips."""

    root = persisted.output_root
    if _has_reparse(root, is_reparse):
        _add(issues, "persistence.reparse", "output_root", "reparse component forbidden")
        return
    try:
        actual = {path.name for path in root.iterdir()} if root.is_dir() else set()
    except OSError as error:
        _add(issues, "persistence.io", "output_root", str(error))
        return
    if actual != _FILES or set(persisted.files) != _FILES:
        extras = actual - _FILES
        if extras:
            for name in sorted(extras):
                _add(issues, "persistence.file_set", f"file.{name}", "unexpected file")
        else:
            _add(issues, "persistence.file_set", "output_root", "expected exact nine files")
    reparse_names = {name for name in actual & _FILES if is_reparse(root / name)}
    for name in sorted(reparse_names):
        _add(issues, "persistence.reparse", f"file.{name}", "reparse file forbidden")
    for name, path in persisted.files.items():
        if Path(path) != root / name:
            _add(issues, "persistence.file_set", f"file.{name}", "mapping escapes root")
    expected_index = _FILES - {"checksums.json"}
    checksums: dict[str, str] = {}
    if "checksums.json" not in reparse_names:
        try:
            payload = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str) and bool(_HASH.fullmatch(value))
                for key, value in payload.items()
            ):
                raise ValueError("expected canonical string checksum mapping")
            checksums = payload
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            _add(issues, "persistence.checksum_index", "file.checksums.json", str(error))
    if set(checksums) != expected_index or dict(persisted.checksums) != checksums:
        _add(
            issues,
            "persistence.checksum_index",
            "file.checksums.json",
            "expected eight non-self entries",
        )
    for name in sorted(expected_index & actual):
        if name in reparse_names:
            continue
        path = root / name
        try:
            actual_hash = _digest(path.read_bytes())
        except OSError as error:
            _add(issues, "persistence.io", f"file.{name}", str(error))
        else:
            if name in checksums and actual_hash != checksums[name]:
                _add(issues, "persistence.checksum", f"checksum.{name}", "digest differs")
    if persisted.dataset_hash != dataset.dataset_hash:
        _add(issues, "persistence.manifest", "persisted.dataset_hash", "hash differs")
    _validate_scalar_payloads(dataset, root, checksums, is_reparse, issues)
    _validate_truth_payload(dataset, root, checksums, is_reparse, issues)
    _validate_arrow_payloads(dataset, root, checksums, is_reparse, issues)
