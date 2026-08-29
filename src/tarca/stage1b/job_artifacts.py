from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch

from tarca.artifacts.store import LocalArtifactStore
from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash
from tarca.execution.contracts import TaskSpec
from tarca.stage1b.config import QualificationPartition
from tarca.stage1b.dataset import (
    NormalizationStatistics,
    QualificationDataset,
    WindowLineage,
    WindowSample,
    stack_samples,
)
from tarca.stage1b.sources import SourceMaterializationReceipt

_SCHEMA_VERSION = "2.0.0"


def _artifact_root(repo_root: Path) -> Path:
    raw = os.environ.get("TARCA_STAGE1B_ARTIFACT_ROOT", "").strip()
    root = (repo_root / raw if raw else repo_root / "artifacts/stage1b").resolve()
    try:
        root.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("Stage1B artifact root must stay inside the repository") from error
    return root

def stage1b_artifact_store(
    repo_root: Path,
    task: TaskSpec | None = None,
) -> LocalArtifactStore:
    identity_hash = "0" * 64 if task is None else canonical_json_hash(task.identity)
    store_root = (_artifact_root(repo_root) / "runtime/store").relative_to(repo_root.resolve())
    return LocalArtifactStore(
        repo_root,
        producer_stage="stage1b",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash=identity_hash,
        dependencies=() if task is None else task.inputs,
        store_relative_root=store_root.as_posix(),
    )


def _publish_json(repo_root: Path, task: TaskSpec, value: object) -> ArtifactRef:
    return stage1b_artifact_store(repo_root, task).publish_bytes(
        canonical_json_bytes(value) + b"\n",
        task.output_artifact_type,
        "application/json",
        _SCHEMA_VERSION,
    )


def _load_json(repo_root: Path, ref: ArtifactRef) -> dict[str, Any]:
    payload = stage1b_artifact_store(repo_root).load_bytes(ref)
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Stage1B JSON artifact must contain an object")
    return cast(dict[str, Any], decoded)


def _torch_bytes(value: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_torch(repo_root: Path, ref: ArtifactRef) -> dict[str, Any]:
    payload = stage1b_artifact_store(repo_root).load_bytes(ref)
    decoded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(decoded, dict):
        raise ValueError("Stage1B tensor artifact must contain a mapping")
    return cast(dict[str, Any], decoded)


def _publish_torch(repo_root: Path, task: TaskSpec, value: dict[str, Any]) -> ArtifactRef:
    return stage1b_artifact_store(repo_root, task).publish_bytes(
        _torch_bytes(value),
        task.output_artifact_type,
        "application/x-pytorch-state-dict",
        _SCHEMA_VERSION,
    )


def _source_receipt_payload(
    repo_root: Path,
    receipt: SourceMaterializationReceipt,
) -> dict[str, Any]:
    checkout = receipt.checkout_root.resolve()
    relative = checkout.relative_to(repo_root.resolve()).as_posix()
    return {
        "source_id": receipt.source_id,
        "repository_url": receipt.repository_url,
        "commit": receipt.commit,
        "checkout_relative_path": relative,
        "tree_sha256": receipt.tree_sha256,
        "asset_sha256": [list(item) for item in receipt.asset_sha256],
        "authorization_id": receipt.authorization_id,
        "materialized_at_utc": receipt.materialized_at_utc.isoformat(),
    }


def _source_receipt(repo_root: Path, payload: dict[str, Any]) -> SourceMaterializationReceipt:
    relative = Path(str(payload["checkout_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source receipt checkout path is unsafe")
    checkout = (repo_root.resolve() / relative).resolve()
    if repo_root.resolve() not in checkout.parents:
        raise ValueError("source receipt checkout escapes the repository")
    raw_assets = payload["asset_sha256"]
    if not isinstance(raw_assets, list):
        raise ValueError("source receipt asset hashes are invalid")
    return SourceMaterializationReceipt(
        source_id=str(payload["source_id"]),
        repository_url=str(payload["repository_url"]),
        commit=str(payload["commit"]),
        checkout_root=checkout,
        tree_sha256=str(payload["tree_sha256"]),
        asset_sha256=tuple((str(item[0]), str(item[1])) for item in raw_assets),
        authorization_id=str(payload["authorization_id"]),
        materialized_at_utc=datetime.fromisoformat(str(payload["materialized_at_utc"])),
    )

def _dataset_payload(dataset: QualificationDataset) -> dict[str, Any]:
    partitions: dict[str, Any] = {}
    for partition in QualificationPartition:
        samples = dataset.for_partition(partition)
        histories, targets = stack_samples(samples)
        partitions[partition.value] = {
            "histories": histories.detach().cpu(),
            "targets": targets.detach().cpu(),
            "lineages": [
                {**asdict(sample.lineage), "partition": sample.lineage.partition.value}
                for sample in samples
            ],
        }
    return {
        "schema_version": _SCHEMA_VERSION,
        "history_length": dataset.history_length,
        "horizon": dataset.horizon,
        "normalization_mean": dataset.statistics.mean.detach().cpu(),
        "normalization_standard_deviation": dataset.statistics.standard_deviation.detach().cpu(),
        "fitted_partition": dataset.statistics.fitted_partition.value,
        "partitions": partitions,
    }


def _dataset_from_payload(payload: dict[str, Any]) -> QualificationDataset:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("qualification dataset schema is unsupported")
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, dict) or set(raw_partitions) != {
        item.value for item in QualificationPartition
    }:
        raise ValueError("qualification dataset must contain exactly four partitions")
    entries: list[tuple[QualificationPartition, tuple[WindowSample, ...]]] = []
    seen_ids: set[str] = set()
    for partition in QualificationPartition:
        raw = raw_partitions[partition.value]
        histories = raw["histories"]
        targets = raw["targets"]
        lineages = raw["lineages"]
        if not isinstance(histories, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError("qualification dataset tensors are missing")
        if histories.shape[0] != targets.shape[0] or histories.shape[0] != len(lineages):
            raise ValueError("qualification dataset rows are misaligned")
        samples: list[WindowSample] = []
        for index, raw_lineage in enumerate(lineages):
            lineage = WindowLineage(
                **{**raw_lineage, "partition": QualificationPartition(raw_lineage["partition"])}
            )
            if lineage.partition is not partition or lineage.window_id in seen_ids:
                raise ValueError("qualification dataset lineage is duplicated or crossed")
            seen_ids.add(lineage.window_id)
            samples.append(
                WindowSample(
                    history=histories[index].clone(),
                    target=targets[index].clone(),
                    lineage=lineage,
                )
            )
        entries.append((partition, tuple(samples)))
    mean = payload["normalization_mean"]
    standard_deviation = payload["normalization_standard_deviation"]
    if not isinstance(mean, torch.Tensor) or not isinstance(standard_deviation, torch.Tensor):
        raise ValueError("qualification normalization tensors are missing")
    return QualificationDataset(
        statistics=NormalizationStatistics(
            mean=mean.clone(),
            standard_deviation=standard_deviation.clone(),
            fitted_partition=QualificationPartition(str(payload["fitted_partition"])),
        ),
        samples=tuple(entries),
        history_length=int(payload["history_length"]),
        horizon=int(payload["horizon"]),
    )

