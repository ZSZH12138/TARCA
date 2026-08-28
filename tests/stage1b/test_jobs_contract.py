from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tarca.contracts import PROTOCOL_ID
from tarca.execution import ResourceRequest, ScientificIdentity, TaskSpec
from tarca.stage1b.config import QualificationPartition
from tarca.stage1b.dataset import (
    NormalizationStatistics,
    QualificationDataset,
    WindowLineage,
    WindowSample,
)
from tarca.stage1b.jobs import (
    _dataset_from_payload,
    _dataset_payload,
    _generation_worker_count,
    _load_json,
    _load_torch,
    _publish_json,
    _publish_torch,
    _source_cache_root,
    _source_receipt,
    _source_receipt_payload,
    _torch_bytes,
    stage1b_artifact_store,
    stage1b_executor_registry,
)
from tarca.stage1b.sources import SourceMaterializationReceipt


def _task(task_id: str, artifact_type: str) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id=PROTOCOL_ID,
            experiment_id="stage1b-qualification-v2",
            task_id=task_id,
            model_id="model-none",
            data_id="data-none",
            seed=0,
        ),
        phase="UNIT_CONTRACT",
        inputs=(),
        output_artifact_type=artifact_type,
        resource_request=ResourceRequest(
            cpu_threads=1,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=1.0,
        ),
    )


def _dataset() -> QualificationDataset:
    entries: list[tuple[QualificationPartition, tuple[WindowSample, ...]]] = []
    for index, partition in enumerate(QualificationPartition):
        lineage = WindowLineage(
            window_id=f"window-{index}",
            trajectory_id=f"trajectory-{index}",
            world_id="world",
            family_id="family",
            regime_id="unseen" if partition is QualificationPartition.QUAL_UNSEEN else "seen",
            partition=partition,
            seed=101 + index,
            history_start=0,
            history_end=2,
            target_end=3,
            graph_sha256="a" * 64,
            future_noise_sha256="b" * 64,
            source_commit="c" * 40,
            config_sha256="d" * 64,
        )
        entries.append(
            (
                partition,
                (
                    WindowSample(
                        history=torch.full((2, 2), float(index)),
                        target=torch.full((1, 2), float(index + 1)),
                        lineage=lineage,
                    ),
                ),
            )
        )
    return QualificationDataset(
        statistics=NormalizationStatistics(
            mean=torch.zeros(2),
            standard_deviation=torch.ones(2),
        ),
        samples=tuple(entries),
        history_length=2,
        horizon=1,
    )


def test_json_and_torch_artifacts_round_trip_through_immutable_store(tmp_path: Path) -> None:
    json_task = _task("json-task", "UNIT_JSON")
    json_ref = _publish_json(tmp_path, json_task, {"value": 7})
    assert _load_json(tmp_path, json_ref) == {"value": 7}

    torch_task = _task("torch-task", "UNIT_TORCH")
    torch_ref = _publish_torch(tmp_path, torch_task, {"tensor": torch.arange(4)})
    loaded = _load_torch(tmp_path, torch_ref)
    torch.testing.assert_close(loaded["tensor"], torch.arange(4))
    assert _torch_bytes({"tensor": torch.ones(1)})
    assert stage1b_artifact_store(tmp_path).verify_artifact(json_ref)


def test_artifact_decoders_reject_wrong_top_level_types(tmp_path: Path) -> None:
    json_task = _task("bad-json", "BAD_JSON")
    bad_json = stage1b_artifact_store(tmp_path, json_task).publish_bytes(
        b"[]\n", "BAD_JSON", "application/json", "2.0.0"
    )
    with pytest.raises(ValueError, match="JSON artifact"):
        _load_json(tmp_path, bad_json)

    torch_task = _task("bad-torch", "BAD_TORCH")
    bad_torch = stage1b_artifact_store(tmp_path, torch_task).publish_bytes(
        _torch_bytes(cast(Any, [torch.ones(1)])),
        "BAD_TORCH",
        "application/x-pytorch-state-dict",
        "2.0.0",
    )
    with pytest.raises(ValueError, match="tensor artifact"):
        _load_torch(tmp_path, bad_torch)


def test_source_receipt_round_trip_and_cache_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "third_party/stage1b/source"
    checkout.mkdir(parents=True)
    receipt = SourceMaterializationReceipt(
        source_id="source",
        repository_url="https://github.com/example/source.git",
        commit="a" * 40,
        checkout_root=checkout,
        tree_sha256="b" * 64,
        asset_sha256=(("asset.py", "c" * 64),),
        authorization_id="authorization-v2",
        materialized_at_utc=datetime(2026, 8, 26, tzinfo=UTC),
    )
    payload = _source_receipt_payload(tmp_path, receipt)
    restored = _source_receipt(tmp_path, payload)
    assert restored == receipt
    assert _source_cache_root(tmp_path) == tmp_path / "third_party/stage1b"

    override = tmp_path / "official_sources"
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", str(override))
    assert _source_cache_root(tmp_path) == override.resolve()


def test_generation_worker_count_uses_the_scheduler_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARCA_CPU_AFFINITY", "2,4,6,8")
    assert _generation_worker_count() == 4

    monkeypatch.setenv("TARCA_CPU_AFFINITY", "2,2")
    with pytest.raises(ValueError, match="unique"):
        _generation_worker_count()


@pytest.mark.parametrize("unsafe", ["../escape", "C:/escape"])
def test_source_receipt_rejects_unsafe_checkout(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _source_receipt(
            tmp_path,
            {
                "checkout_relative_path": unsafe,
                "asset_sha256": [],
            },
        )


def test_dataset_payload_round_trip_preserves_all_partitions() -> None:
    original = _dataset()
    restored = _dataset_from_payload(_dataset_payload(original))

    assert restored.history_length == 2
    assert restored.horizon == 1
    assert restored.statistics.fitted_partition is QualificationPartition.QUAL_TRAIN
    assert tuple(
        restored.for_partition(partition)[0].lineage.window_id
        for partition in QualificationPartition
    ) == tuple(f"window-{index}" for index in range(4))


def test_dataset_payload_rejects_schema_partition_and_tensor_drift() -> None:
    bad_schema = _dataset_payload(_dataset())
    bad_schema["schema_version"] = "1.0.0"
    with pytest.raises(ValueError, match="schema"):
        _dataset_from_payload(bad_schema)

    bad_partitions = _dataset_payload(_dataset())
    del bad_partitions["partitions"]["QUAL_UNSEEN"]
    with pytest.raises(ValueError, match="four partitions"):
        _dataset_from_payload(bad_partitions)

    bad_tensors = _dataset_payload(_dataset())
    bad_tensors["partitions"]["QUAL_TRAIN"]["histories"] = []
    with pytest.raises(ValueError, match="tensors"):
        _dataset_from_payload(bad_tensors)


def test_dataset_payload_rejects_misaligned_or_crossed_lineage() -> None:
    misaligned = _dataset_payload(_dataset())
    misaligned["partitions"]["QUAL_TRAIN"]["lineages"] = []
    with pytest.raises(ValueError, match="misaligned"):
        _dataset_from_payload(misaligned)

    crossed = copy.deepcopy(_dataset_payload(_dataset()))
    crossed["partitions"]["QUAL_TUNE"]["lineages"][0]["partition"] = "QUAL_TRAIN"
    with pytest.raises(ValueError, match="duplicated or crossed"):
        _dataset_from_payload(crossed)

    bad_statistics = _dataset_payload(_dataset())
    bad_statistics["normalization_mean"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="normalization tensors"):
        _dataset_from_payload(bad_statistics)


def test_executor_registry_contains_only_fixed_stage1b_jobs(tmp_path: Path) -> None:
    registry = stage1b_executor_registry(tmp_path)
    assert registry.keys == (
        "stage1b.aggregate_qualification",
        "stage1b.check_world_health",
        "stage1b.freeze_check_model",
        "stage1b.generate_dataset",
        "stage1b.materialize_source",
        "stage1b.publish_qualification_receipt",
        "stage1b.reproduce_official_case",
        "stage1b.score_bootstrap",
        "stage1b.score_var",
        "stage1b.train_neural",
        "stage1b.validate_dataset",
    )
    assert callable(registry.resolve("stage1b.score_var"))


def test_store_manifest_is_json_serializable(tmp_path: Path) -> None:
    ref = _publish_json(tmp_path, _task("serializable", "SERIALIZABLE"), {"ok": True})
    manifest_path = next((tmp_path / "artifacts/stage1b/runtime/store").rglob("*.manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact"]["artifact_id"] == ref.artifact_id
