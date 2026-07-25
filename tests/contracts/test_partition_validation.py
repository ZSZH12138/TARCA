from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

import pytest

from tarca.contracts.manifests import (
    InterventionPair,
    validate_disjoint_window_partitions,
    validate_intervention_pair_partitions,
)
from tarca.contracts.types import RegimeRelation, SplitPartition

_T = TypeVar("_T")


class _SinglePassIterable(Generic[_T]):
    def __init__(self, values: tuple[_T, ...]) -> None:
        self._values = values
        self._iterations = 0

    def __iter__(self) -> Iterator[_T]:
        self._iterations += 1
        if self._iterations > 1:
            raise AssertionError("iterable was consumed more than once")
        return iter(self._values)


def _pair(
    *,
    partition: SplitPartition,
    base_window_id: str,
    source_window_id: str,
    concept_name: str = "temperature",
) -> InterventionPair:
    return InterventionPair.build(
        partition=partition,
        base_window_id=base_window_id,
        source_window_id=source_window_id,
        concept_name=concept_name,
        regime_relation=RegimeRelation.CROSS,
        matching_distance=0.5,
        concept_delta=1.0,
    )


def test_disjoint_validator_accepts_generators_and_consumes_each_iterable_once() -> None:
    partitions = {
        SplitPartition.TRAIN: _SinglePassIterable(("train-a", "train-b")),
        SplitPartition.VALIDATION: _SinglePassIterable(("validation-a",)),
        SplitPartition.TEST: _SinglePassIterable(("test-a",)),
    }
    assert validate_disjoint_window_partitions(partitions) is None


def test_disjoint_validator_requires_every_partition() -> None:
    with pytest.raises(ValueError, match=r"missing.*test"):
        validate_disjoint_window_partitions(
            {
                SplitPartition.TRAIN: ("train-a",),
                SplitPartition.VALIDATION: ("validation-a",),
            }
        )


@pytest.mark.parametrize("invalid_ids", ["window-id", "", ("",), ("   ",), (3,)])
def test_disjoint_validator_rejects_invalid_id_iterables(invalid_ids: object) -> None:
    with pytest.raises((TypeError, ValueError), match=r"train"):
        validate_disjoint_window_partitions(
            {
                SplitPartition.TRAIN: invalid_ids,  # type: ignore[dict-item]
                SplitPartition.VALIDATION: ("validation-a",),
                SplitPartition.TEST: ("test-a",),
            }
        )


def test_disjoint_validator_detects_cross_partition_leakage() -> None:
    with pytest.raises(ValueError, match=r"shared-window") as error:
        validate_disjoint_window_partitions(
            {
                SplitPartition.TRAIN: ("train-a", "shared-window"),
                SplitPartition.VALIDATION: ("shared-window",),
                SplitPartition.TEST: ("test-a",),
            }
        )
    message = str(error.value)
    assert "train" in message
    assert "validation" in message


def test_disjoint_validator_reports_deterministic_bounded_evidence() -> None:
    conflicts = tuple(f"shared-{index:02d}" for index in range(10))
    forward = {
        SplitPartition.TRAIN: iter(reversed(conflicts)),
        SplitPartition.VALIDATION: iter(conflicts),
        SplitPartition.TEST: iter(("test-only",)),
    }
    reverse = {
        SplitPartition.TRAIN: iter(conflicts),
        SplitPartition.VALIDATION: iter(reversed(conflicts)),
        SplitPartition.TEST: iter(("test-only",)),
    }

    with pytest.raises(ValueError) as forward_error:
        validate_disjoint_window_partitions(forward)
    with pytest.raises(ValueError) as reverse_error:
        validate_disjoint_window_partitions(reverse)

    forward_message = str(forward_error.value)
    assert forward_message == str(reverse_error.value)
    for conflict in conflicts[:5]:
        assert conflict in forward_message
    assert conflicts[5] not in forward_message


def test_pair_validator_accepts_a_generator_and_same_partition_reuse() -> None:
    pairs = (
        _pair(
            partition=SplitPartition.TRAIN,
            base_window_id="shared-in-train",
            source_window_id="source-a",
        ),
        _pair(
            partition=SplitPartition.TRAIN,
            base_window_id="base-b",
            source_window_id="shared-in-train",
        ),
        _pair(
            partition=SplitPartition.TEST,
            base_window_id="test-base",
            source_window_id="test-source",
        ),
    )
    assert validate_intervention_pair_partitions(pair for pair in pairs) is None


def test_pair_validator_consumes_the_pair_iterable_once() -> None:
    pairs = (
        _pair(
            partition=SplitPartition.TRAIN,
            base_window_id="train-base",
            source_window_id="train-source",
        ),
        _pair(
            partition=SplitPartition.TEST,
            base_window_id="test-base",
            source_window_id="test-source",
        ),
    )
    assert validate_intervention_pair_partitions(_SinglePassIterable(pairs)) is None


def test_pair_validator_detects_window_crossing_between_base_and_source_roles() -> None:
    train_pair = _pair(
        partition=SplitPartition.TRAIN,
        base_window_id="shared-window",
        source_window_id="train-source",
    )
    validation_pair = _pair(
        partition=SplitPartition.VALIDATION,
        base_window_id="validation-base",
        source_window_id="shared-window",
    )

    with pytest.raises(ValueError, match=r"shared-window") as error:
        validate_intervention_pair_partitions((train_pair, validation_pair))

    message = str(error.value)
    assert "train" in message
    assert "validation" in message


def test_pair_validator_detects_a_pair_id_crossing_partitions() -> None:
    train_pair = _pair(
        partition=SplitPartition.TRAIN,
        base_window_id="base",
        source_window_id="source",
    )
    test_pair = _pair(
        partition=SplitPartition.TEST,
        base_window_id="base",
        source_window_id="source",
    )
    assert test_pair.pair_id == train_pair.pair_id

    with pytest.raises(ValueError) as error:
        validate_intervention_pair_partitions((train_pair, test_pair))

    message = str(error.value)
    assert "pair_id" in message
    assert train_pair.pair_id in message
    assert "train" in message
    assert "test" in message


def test_pair_validator_reports_deterministic_bounded_evidence() -> None:
    train_pairs = tuple(
        _pair(
            partition=SplitPartition.TRAIN,
            base_window_id=f"base-{index}",
            source_window_id=f"source-{index}",
        )
        for index in range(7)
    )
    test_pairs = tuple(
        _pair(
            partition=SplitPartition.TEST,
            base_window_id=f"base-{index}",
            source_window_id=f"source-{index}",
        )
        for index in range(7)
    )
    sorted_pair_ids = sorted(pair.pair_id for pair in train_pairs)

    with pytest.raises(ValueError) as forward_error:
        validate_intervention_pair_partitions((*train_pairs, *test_pairs))
    with pytest.raises(ValueError) as reverse_error:
        validate_intervention_pair_partitions((*reversed(test_pairs), *reversed(train_pairs)))

    forward_message = str(forward_error.value)
    assert forward_message == str(reverse_error.value)
    for pair_id in sorted_pair_ids[:5]:
        assert pair_id in forward_message
    assert sorted_pair_ids[5] not in forward_message
