from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tarca.contracts import DatasetWindowPartition, WindowBatch
from tarca.stage1b.config import QualificationPartition
from tarca.stage1b.dataset import (
    bridge_qualification_windows,
    partition_for_qualification,
    validate_qualification_window,
)

from .model_helpers import window_batch


@pytest.mark.parametrize(
    ("qualification", "physical"),
    [
        (QualificationPartition.QUAL_TRAIN, DatasetWindowPartition.TRAIN),
        (QualificationPartition.QUAL_TUNE, DatasetWindowPartition.VALIDATION),
        (QualificationPartition.QUAL_SEEN, DatasetWindowPartition.TEST_SEEN_REGIME),
        (QualificationPartition.QUAL_UNSEEN, DatasetWindowPartition.TEST_UNSEEN_REGIME),
    ],
)
def test_fixed_partition_mapping(
    qualification: QualificationPartition,
    physical: DatasetWindowPartition,
) -> None:
    assert partition_for_qualification(qualification) is physical


def test_truth_manifest_is_rejected_from_window_metadata() -> None:
    batch = window_batch(torch.ones((2, 4, 2)), torch.ones((2, 2, 2)))
    contaminated = replace(batch, metadata={**batch.metadata, "scm_truth": "hidden"})

    with pytest.raises(ValueError, match="truth"):
        validate_qualification_window(contaminated)


def _physical_batch(
    qualification: QualificationPartition,
    value: float,
) -> WindowBatch:
    physical = partition_for_qualification(qualification)
    batch = window_batch(
        torch.full((2, 4, 2), value),
        torch.full((2, 2, 2), value),
        prefix=qualification.value.lower(),
    )
    return replace(
        batch,
        metadata={
            "qualification_partition": qualification.value,
            "physical_partition": physical.value,
        },
    )


def test_partition_bridge_preserves_batches_and_fits_train_only() -> None:
    batches = {
        QualificationPartition.QUAL_TRAIN: _physical_batch(QualificationPartition.QUAL_TRAIN, 2.0),
        QualificationPartition.QUAL_TUNE: _physical_batch(
            QualificationPartition.QUAL_TUNE, 1_000_000.0
        ),
        QualificationPartition.QUAL_SEEN: _physical_batch(
            QualificationPartition.QUAL_SEEN, 1_000_000.0
        ),
        QualificationPartition.QUAL_UNSEEN: _physical_batch(
            QualificationPartition.QUAL_UNSEEN, 1_000_000.0
        ),
    }

    bridge = bridge_qualification_windows(batches)

    assert (
        bridge.batch_for(QualificationPartition.QUAL_TRAIN)
        is batches[QualificationPartition.QUAL_TRAIN]
    )
    assert bridge.fitted_partition is DatasetWindowPartition.TRAIN
    torch.testing.assert_close(
        bridge.normalization_mean,
        torch.tensor([2.0, 2.0], dtype=torch.float64),
    )


def test_partition_bridge_rejects_cross_partition_window_reuse() -> None:
    train = _physical_batch(QualificationPartition.QUAL_TRAIN, 1.0)
    tune = _physical_batch(QualificationPartition.QUAL_TUNE, 2.0)
    tune = replace(tune, window_id=train.window_id)
    batches = {
        QualificationPartition.QUAL_TRAIN: train,
        QualificationPartition.QUAL_TUNE: tune,
        QualificationPartition.QUAL_SEEN: _physical_batch(QualificationPartition.QUAL_SEEN, 3.0),
        QualificationPartition.QUAL_UNSEEN: _physical_batch(
            QualificationPartition.QUAL_UNSEEN, 4.0
        ),
    }

    with pytest.raises(ValueError, match="partition isolation"):
        bridge_qualification_windows(batches)
