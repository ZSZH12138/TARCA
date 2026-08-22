from __future__ import annotations

from pathlib import Path

import torch

from tarca.stage1b.config import (
    QualificationPartition,
    TrajectoryPartitionCounts,
    load_qualification_config,
    load_world_suite,
)
from tarca.stage1b.dataset import (
    TrajectoryRecord,
    generate_world_split,
    prepare_dataset,
)
from tarca.stage1b.splits import build_qualification_split
from tarca.stage1b.worlds import build_world

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(
    trajectory_id: str,
    partition: QualificationPartition,
    values: torch.Tensor,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        world_id="world",
        family_id="family",
        regime_id="seen" if partition is not QualificationPartition.QUAL_UNSEEN else "unseen",
        partition=partition,
        seed=11,
        graph_sha256="a" * 64,
        future_noise_sha256="d" * 64,
        source_commit="b" * 40,
        config_sha256="c" * 64,
        values=values.to(torch.float64),
    )


def test_normalizer_uses_qual_train_only() -> None:
    train_values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    extreme_values = torch.full((4, 2), 1_000_000.0)
    split = build_qualification_split(
        (
            _record("train", QualificationPartition.QUAL_TRAIN, train_values),
            _record("tune", QualificationPartition.QUAL_TUNE, extreme_values),
            _record("seen", QualificationPartition.QUAL_SEEN, extreme_values),
            _record("unseen", QualificationPartition.QUAL_UNSEEN, extreme_values),
        )
    )

    dataset = prepare_dataset(split, history_length=2, horizon=1)

    torch.testing.assert_close(
        dataset.statistics.mean,
        torch.tensor([3.0, 4.0], dtype=torch.float64),
    )
    assert dataset.statistics.fitted_partition is QualificationPartition.QUAL_TRAIN


def test_window_lineage_stays_inside_owning_trajectory_and_partition() -> None:
    values = torch.arange(24, dtype=torch.float64).reshape(12, 2)
    split = build_qualification_split(
        tuple(
            _record(partition.value, partition, values)
            for partition in QualificationPartition
        )
    )

    dataset = prepare_dataset(split, history_length=4, horizon=2, stride=2)

    for partition in QualificationPartition:
        samples = dataset.for_partition(partition)
        assert samples
        assert all(sample.lineage.partition is partition for sample in samples)
        assert all(sample.lineage.trajectory_id == partition.value for sample in samples)
        assert all(sample.history.shape == (4, 2) for sample in samples)
        assert all(sample.target.shape == (2, 2) for sample in samples)


def test_dataset_has_no_formal_test_surface() -> None:
    assert {partition.value for partition in QualificationPartition} == {
        "QUAL_TRAIN",
        "QUAL_TUNE",
        "QUAL_SEEN",
        "QUAL_UNSEEN",
    }


def test_real_world_generation_uses_all_qualification_partitions_deterministically() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    qualification = load_qualification_config(
        REPOSITORY_ROOT / "configs/stage1b/qualification_v1.yaml"
    ).model_copy(
        update={
            "trajectory_length": 48,
            "warmup_steps": 8,
            "trajectories_per_partition": TrajectoryPartitionCounts(
                QUAL_TRAIN=1,
                QUAL_TUNE=1,
                QUAL_SEEN=1,
                QUAL_UNSEEN=1,
            ),
        }
    )
    world = build_world(suite.world("network_cml_v1"))

    first = generate_world_split(
        world,
        qualification,
        qualification_seed=104729,
        source_commit=suite.sources[0].commit,
    )
    second = generate_world_split(
        world,
        qualification,
        qualification_seed=104729,
        source_commit=suite.sources[0].commit,
    )

    assert len(first.records) == 4
    assert tuple(record.trajectory_id for record in first.records) == tuple(
        record.trajectory_id for record in second.records
    )
    for left, right in zip(first.records, second.records, strict=True):
        torch.testing.assert_close(left.values, right.values, rtol=0.0, atol=0.0)
    assert all("TEST" not in record.partition.value for record in first.records)
