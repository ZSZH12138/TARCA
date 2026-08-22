from __future__ import annotations

from dataclasses import dataclass

from tarca.stage1b.config import QualificationPartition
from tarca.stage1b.dataset import TrajectoryRecord


class SplitValidationError(ValueError):
    """Raised when whole-trajectory qualification isolation is violated."""


@dataclass(frozen=True, slots=True)
class QualificationSplit:
    records: tuple[TrajectoryRecord, ...]

    def partitions(self) -> tuple[QualificationPartition, ...]:
        return tuple(
            partition
            for partition in QualificationPartition
            if any(record.partition is partition for record in self.records)
        )

    def records_for_partition(
        self, partition: QualificationPartition
    ) -> tuple[TrajectoryRecord, ...]:
        return tuple(record for record in self.records if record.partition is partition)

    def partition_by_trajectory_id(
        self,
    ) -> dict[str, tuple[QualificationPartition, ...]]:
        owners: dict[str, set[QualificationPartition]] = {}
        for record in self.records:
            owners.setdefault(record.trajectory_id, set()).add(record.partition)
        return {
            trajectory_id: tuple(
                partition for partition in QualificationPartition if partition in partitions
            )
            for trajectory_id, partitions in owners.items()
        }


def build_qualification_split(records: tuple[TrajectoryRecord, ...]) -> QualificationSplit:
    if not records:
        raise SplitValidationError("qualification split must contain trajectories")
    split = QualificationSplit(records=records)
    owners = split.partition_by_trajectory_id()
    if any(len(partitions) != 1 for partitions in owners.values()):
        raise SplitValidationError("a trajectory belongs to more than one qualification partition")
    if set(split.partitions()) != set(QualificationPartition):
        raise SplitValidationError("split must contain all four qualification partitions")
    return split
