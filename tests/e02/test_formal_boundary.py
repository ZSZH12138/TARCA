from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from tarca.contracts import (
    ArtifactRef,
    DatasetSpec,
    DatasetWindowPartition,
    SealedAccessGrant,
)
from tarca.e02.config import load_e02_config
from tarca.stage2.config import load_stage2_config

NOW = datetime(2026, 8, 31, 12, tzinfo=UTC)
DATASET = DatasetSpec(name="lorenz96_twoscale_v2", version="e02-v1")


def _data_module():
    return importlib.import_module("tarca.stage2.data")


def _grant() -> SealedAccessGrant:
    return SealedAccessGrant(
        grant_id="e02-formal-grant-v1",
        dataset=DATASET,
        scope_name="e02_predictor_validity_v1-formal",
        allowed_partitions=(
            DatasetWindowPartition.TEST_SEEN_REGIME,
            DatasetWindowPartition.TEST_UNSEEN_REGIME,
        ),
        authorization_ref=ArtifactRef(
            artifact_id="e02-formal-authorization-v1",
            artifact_type="SEALED_ACCESS_AUTHORIZATION",
            content_hash="b" * 64,
            schema_version="1.0.0",
            relative_path=None,
        ),
        issued_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(minutes=5),
    )


def _formal_record(identifier: str, partition: DatasetWindowPartition, data_seed: int):
    module = _data_module()
    return module.Stage2Trajectory(
        trajectory_id=identifier,
        world_id="lorenz96_twoscale_v2",
        regime_id="seen" if partition is DatasetWindowPartition.TEST_SEEN_REGIME else "unseen",
        partition=partition,
        data_seed=data_seed,
        trajectory_seed=data_seed + int(identifier.rsplit("-", 1)[1]),
        source_commit="6f28942f6a703c2b52501d01258ca2708539f209",
        config_sha256="a" * 64,
        values=torch.arange(90 * 2, dtype=torch.float64).reshape(90, 2),
    )


def test_formal_bundle_refuses_read_without_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _data_module()
    called = False

    def forbidden_reader():
        nonlocal called
        called = True
        raise AssertionError("formal storage was opened before authorization")

    monkeypatch.setattr(module, "_read_formal_storage", forbidden_reader)

    with pytest.raises(PermissionError, match="sealed access requires a grant"):
        module.open_formal_bundle(
            load_stage2_config(Path("configs/stage2/stage2_v1.yaml")),
            load_e02_config(Path("configs/e02/e02_v1.yaml")),
            None,
            accessed_at=NOW,
        )
    assert called is False


def test_formal_bundle_reads_only_after_exact_two_partition_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _data_module()
    called = False

    def reader():
        nonlocal called
        called = True
        seeds = (1729, 2718, 3141, 5772, 8111)
        return tuple(
            _formal_record(
                f"seen-{seed_index * 12 + index}",
                DatasetWindowPartition.TEST_SEEN_REGIME,
                seed,
            )
            for seed_index, seed in enumerate(seeds)
            for index in range(12)
        ) + tuple(
            _formal_record(
                f"unseen-{seed_index * 12 + index}",
                DatasetWindowPartition.TEST_UNSEEN_REGIME,
                seed,
            )
            for seed_index, seed in enumerate(seeds)
            for index in range(12)
        )

    monkeypatch.setattr(module, "_read_formal_storage", reader)
    normalizer = module.Stage2NormalizationStatistics(
        mean=torch.zeros(2, dtype=torch.float64),
        standard_deviation=torch.ones(2, dtype=torch.float64),
        fitted_partition=DatasetWindowPartition.TRAIN,
        trajectory_ids=("frozen-train",),
    )

    bundle = module.open_formal_bundle(
        load_stage2_config(Path("configs/stage2/stage2_v1.yaml")),
        load_e02_config(Path("configs/e02/e02_v1.yaml")),
        _grant(),
        accessed_at=NOW,
        normalizer=normalizer,
    )

    assert called is True
    assert bundle.trajectory_count(DatasetWindowPartition.TEST_SEEN_REGIME) == 60
    assert bundle.trajectory_count(DatasetWindowPartition.TEST_UNSEEN_REGIME) == 60


def test_formal_bundle_rejects_a_grant_missing_unseen_partition() -> None:
    module = _data_module()
    grant = _grant().model_copy(
        update={"allowed_partitions": (DatasetWindowPartition.TEST_SEEN_REGIME,)}
    )

    with pytest.raises(PermissionError, match="partition mismatch"):
        module.open_formal_bundle(
            load_stage2_config(Path("configs/stage2/stage2_v1.yaml")),
            load_e02_config(Path("configs/e02/e02_v1.yaml")),
            grant,
            accessed_at=NOW,
        )
