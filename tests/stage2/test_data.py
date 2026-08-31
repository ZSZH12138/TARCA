from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tarca.contracts import DatasetWindowPartition
from tarca.stage1b.config import RegimeSplitRole

STAGE2_CONFIG_PATH = Path("configs/stage2/stage2_v1.yaml")


def _data_module():
    return importlib.import_module("tarca.stage2.data")


def _record(identifier: str, partition: DatasetWindowPartition, value: float, seed: int):
    module = _data_module()
    values = torch.stack(
        (
            torch.linspace(value, value + 1.0, 7),
            torch.linspace(value + 2.0, value + 3.0, 7),
        ),
        dim=1,
    )
    return module.Stage2Trajectory(
        trajectory_id=identifier,
        world_id="lorenz96_twoscale_v2",
        regime_id="seen-baseline",
        partition=partition,
        data_seed=seed,
        trajectory_seed=seed + 1000,
        source_commit="6f28942f6a703c2b52501d01258ca2708539f209",
        config_sha256="a" * 64,
        values=values,
    )


def test_development_bundle_has_exact_trajectory_counts_and_train_lineage() -> None:
    module = _data_module()
    records = tuple(
        _record(f"train-{index}", DatasetWindowPartition.TRAIN, float(index), 10 + index)
        for index in range(72)
    ) + tuple(
        _record(
            f"validation-{index}",
            DatasetWindowPartition.VALIDATION,
            1000.0 + index,
            100 + index,
        )
        for index in range(24)
    )

    bundle = module.prepare_stage2_bundle(records, history=4, horizon=2)

    assert bundle.trajectory_count(DatasetWindowPartition.TRAIN) == 72
    assert bundle.trajectory_count(DatasetWindowPartition.VALIDATION) == 24
    assert set(bundle.normalizer.trajectory_ids) == set(
        bundle.trajectory_ids(DatasetWindowPartition.TRAIN)
    )
    assert bundle.normalizer.fitted_partition is DatasetWindowPartition.TRAIN


def test_validation_values_do_not_change_training_normalizer() -> None:
    module = _data_module()
    train = (
        _record("train-a", DatasetWindowPartition.TRAIN, 0.0, 11),
        _record("train-b", DatasetWindowPartition.TRAIN, 2.0, 12),
    )
    normal = module.prepare_stage2_bundle(
        (*train, _record("validation", DatasetWindowPartition.VALIDATION, 10.0, 21)),
        history=4,
        horizon=2,
    )
    extreme = module.prepare_stage2_bundle(
        (*train, _record("validation", DatasetWindowPartition.VALIDATION, 1_000_000.0, 21)),
        history=4,
        horizon=2,
    )

    assert torch.equal(normal.normalizer.mean, extreme.normalizer.mean)
    assert torch.equal(
        normal.normalizer.standard_deviation,
        extreme.normalizer.standard_deviation,
    )


def test_windows_never_cross_trajectory_or_partition_boundaries() -> None:
    module = _data_module()
    records = (
        _record("train-a", DatasetWindowPartition.TRAIN, 0.0, 11),
        _record("train-b", DatasetWindowPartition.TRAIN, 2.0, 12),
        _record("validation", DatasetWindowPartition.VALIDATION, 4.0, 21),
    )
    bundle = module.prepare_stage2_bundle(records, history=4, horizon=2)

    for partition in (DatasetWindowPartition.TRAIN, DatasetWindowPartition.VALIDATION):
        for sample in bundle.for_partition(partition):
            assert sample.lineage.partition is partition
            assert sample.lineage.trajectory_id in bundle.trajectory_ids(partition)
            assert sample.lineage.target_end - sample.lineage.history_start == 6


def test_stack_partition_preserves_window_trajectory_ids() -> None:
    module = _data_module()
    bundle = module.prepare_stage2_bundle(
        (
            _record("train-a", DatasetWindowPartition.TRAIN, 0.0, 11),
            _record("validation", DatasetWindowPartition.VALIDATION, 4.0, 21),
        ),
        history=4,
        horizon=2,
    )

    x, y, trajectory_ids = module.stack_partition(bundle, DatasetWindowPartition.TRAIN)

    assert x.shape == (2, 4, 2)
    assert y.shape == (2, 2, 2)
    assert trajectory_ids == ("train-a", "train-a")


def test_bundle_rejects_duplicate_trajectory_ids() -> None:
    module = _data_module()
    duplicate = _record("same", DatasetWindowPartition.TRAIN, 0.0, 11)
    with pytest.raises(ValueError, match="trajectory IDs must be unique"):
        module.prepare_stage2_bundle(
            (
                duplicate,
                duplicate,
                _record("validation", DatasetWindowPartition.VALIDATION, 4.0, 21),
            ),
            history=4,
            horizon=2,
        )


class _TinyDevelopmentWorld:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            world_id="lorenz96_twoscale_v2",
            regimes=(
                SimpleNamespace(
                    regime_id="seen-baseline",
                    split_role=RegimeSplitRole.SEEN,
                ),
            ),
        )

    def simulate(self, request):  # type: ignore[no-untyped-def]
        value = float(request.seed % 1000)
        values = torch.stack(
            (
                torch.linspace(value, value + 1.0, request.length),
                torch.linspace(value + 2.0, value + 3.0, request.length),
            ),
            dim=1,
        )
        return SimpleNamespace(values=values)


def test_generate_development_bundle_uses_exact_frozen_design() -> None:
    module = _data_module()
    config = importlib.import_module("tarca.stage2.config").load_stage2_config(
        STAGE2_CONFIG_PATH
    )

    first = module.generate_development_bundle(config, _TinyDevelopmentWorld(), worker_count=1)
    replay = module.generate_development_bundle(config, _TinyDevelopmentWorld(), worker_count=1)

    assert first.trajectory_count(DatasetWindowPartition.TRAIN) == 72
    assert first.trajectory_count(DatasetWindowPartition.VALIDATION) == 24
    assert {record.data_seed for record in first.records} == set(config.data.development_seeds)
    assert all(record.values.shape == (512, 2) for record in first.records)
    assert first.manifest_sha256 == replay.manifest_sha256
    assert tuple(record.trajectory_seed for record in first.records) == tuple(
        record.trajectory_seed for record in replay.records
    )


def test_generate_development_bundle_rejects_invalid_worker_count() -> None:
    module = _data_module()
    config = importlib.import_module("tarca.stage2.config").load_stage2_config(
        STAGE2_CONFIG_PATH
    )

    with pytest.raises(ValueError, match="worker count"):
        module.generate_development_bundle(config, _TinyDevelopmentWorld(), worker_count=0)
