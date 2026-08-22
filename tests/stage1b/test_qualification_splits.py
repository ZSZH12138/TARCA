from __future__ import annotations

import pytest
import torch

from tarca.stage1b.config import QualificationPartition
from tarca.stage1b.dataset import TrajectoryRecord
from tarca.stage1b.splits import SplitValidationError, build_qualification_split


def _record(trajectory_id: str, partition: QualificationPartition) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        world_id="world",
        family_id="family",
        regime_id="seen",
        partition=partition,
        seed=1,
        graph_sha256="a" * 64,
        future_noise_sha256="d" * 64,
        source_commit="b" * 40,
        config_sha256="c" * 64,
        values=torch.arange(64, dtype=torch.float64).reshape(32, 2),
    )


def test_trajectory_owner_is_unique_across_partitions() -> None:
    split = build_qualification_split(
        (
            _record("train-0", QualificationPartition.QUAL_TRAIN),
            _record("tune-0", QualificationPartition.QUAL_TUNE),
            _record("seen-0", QualificationPartition.QUAL_SEEN),
            _record("unseen-0", QualificationPartition.QUAL_UNSEEN),
        )
    )

    owners = split.partition_by_trajectory_id()

    assert all(len(partitions) == 1 for partitions in owners.values())
    assert set(split.partitions()) == set(QualificationPartition)


def test_reused_trajectory_id_across_partitions_fails_closed() -> None:
    with pytest.raises(SplitValidationError, match="more than one qualification partition"):
        build_qualification_split(
            (
                _record("duplicate", QualificationPartition.QUAL_TRAIN),
                _record("duplicate", QualificationPartition.QUAL_SEEN),
            )
        )


def test_empty_required_partition_fails_closed() -> None:
    with pytest.raises(SplitValidationError, match="all four qualification partitions"):
        build_qualification_split(
            (
                _record("train-0", QualificationPartition.QUAL_TRAIN),
                _record("tune-0", QualificationPartition.QUAL_TUNE),
            )
        )
