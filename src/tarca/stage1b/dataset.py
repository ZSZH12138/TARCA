from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tarca.stage1b.config import (
    QualificationConfig,
    QualificationPartition,
    RegimeSplitRole,
)
from tarca.stage1b.worlds import PublishedWorldAdapter, SimulationRequest

if TYPE_CHECKING:
    from tarca.stage1b.splits import QualificationSplit


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    trajectory_id: str
    world_id: str
    family_id: str
    regime_id: str
    partition: QualificationPartition
    seed: int
    graph_sha256: str
    future_noise_sha256: str
    source_commit: str
    config_sha256: str
    values: torch.Tensor

    def __post_init__(self) -> None:
        labels = (self.trajectory_id, self.world_id, self.family_id, self.regime_id)
        if any(not label.strip() for label in labels):
            raise ValueError("trajectory lineage labels must not be blank")
        if self.values.ndim != 2 or self.values.shape[0] < 2:
            raise ValueError("trajectory values must be a two-dimensional time series")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("trajectory values must be finite")


@dataclass(frozen=True, slots=True)
class NormalizationStatistics:
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    fitted_partition: QualificationPartition = QualificationPartition.QUAL_TRAIN


@dataclass(frozen=True, slots=True)
class WindowLineage:
    window_id: str
    trajectory_id: str
    world_id: str
    family_id: str
    regime_id: str
    partition: QualificationPartition
    seed: int
    history_start: int
    history_end: int
    target_end: int
    graph_sha256: str
    future_noise_sha256: str
    source_commit: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class WindowSample:
    history: torch.Tensor
    target: torch.Tensor
    lineage: WindowLineage


@dataclass(frozen=True, slots=True)
class QualificationDataset:
    statistics: NormalizationStatistics
    samples: tuple[tuple[QualificationPartition, tuple[WindowSample, ...]], ...]
    history_length: int
    horizon: int

    def for_partition(self, partition: QualificationPartition) -> tuple[WindowSample, ...]:
        for candidate, samples in self.samples:
            if candidate is partition:
                return samples
        raise KeyError(partition)


def stack_samples(samples: tuple[WindowSample, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("cannot stack an empty qualification sample set")
    return (
        torch.stack(tuple(sample.history for sample in samples)),
        torch.stack(tuple(sample.target for sample in samples)),
    )


def stack_partition(
    dataset: QualificationDataset,
    partition: QualificationPartition,
) -> tuple[torch.Tensor, torch.Tensor]:
    return stack_samples(dataset.for_partition(partition))


def _window_id(record: TrajectoryRecord, start: int, history: int, horizon: int) -> str:
    identity = (
        f"{record.trajectory_id}|{record.partition.value}|{start}|{history}|{horizon}"
    ).encode()
    return hashlib.sha256(identity).hexdigest()


def _windows_for_record(
    record: TrajectoryRecord,
    mean: torch.Tensor,
    standard_deviation: torch.Tensor,
    history_length: int,
    horizon: int,
    stride: int,
) -> tuple[WindowSample, ...]:
    normalized = (record.values.to(torch.float64) - mean) / standard_deviation
    last_start = normalized.shape[0] - history_length - horizon
    samples: list[WindowSample] = []
    for start in range(0, last_start + 1, stride):
        history_end = start + history_length
        target_end = history_end + horizon
        lineage = WindowLineage(
            window_id=_window_id(record, start, history_length, horizon),
            trajectory_id=record.trajectory_id,
            world_id=record.world_id,
            family_id=record.family_id,
            regime_id=record.regime_id,
            partition=record.partition,
            seed=record.seed,
            history_start=start,
            history_end=history_end,
            target_end=target_end,
            graph_sha256=record.graph_sha256,
            future_noise_sha256=record.future_noise_sha256,
            source_commit=record.source_commit,
            config_sha256=record.config_sha256,
        )
        samples.append(
            WindowSample(
                history=normalized[start:history_end].to(torch.float32).clone(),
                target=normalized[history_end:target_end].to(torch.float32).clone(),
                lineage=lineage,
            )
        )
    return tuple(samples)


def prepare_dataset(
    split: QualificationSplit,
    history_length: int,
    horizon: int,
    stride: int = 1,
) -> QualificationDataset:
    if history_length <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("history, horizon, and stride must be positive")
    train_records = split.records_for_partition(QualificationPartition.QUAL_TRAIN)
    if not train_records:
        raise ValueError("QUAL_TRAIN trajectories are required for normalization")
    training_values = torch.cat(
        tuple(record.values.to(torch.float64) for record in train_records), dim=0
    )
    mean = training_values.mean(dim=0)
    standard_deviation = training_values.std(dim=0, unbiased=False).clamp_min(1e-8)
    statistics = NormalizationStatistics(
        mean=mean.clone(),
        standard_deviation=standard_deviation.clone(),
    )
    partition_samples: list[tuple[QualificationPartition, tuple[WindowSample, ...]]] = []
    for partition in QualificationPartition:
        samples = tuple(
            sample
            for record in split.records_for_partition(partition)
            for sample in _windows_for_record(
                record,
                mean,
                standard_deviation,
                history_length,
                horizon,
                stride,
            )
        )
        if not samples:
            raise ValueError(f"{partition.value} does not contain any forecast windows")
        partition_samples.append((partition, samples))
    return QualificationDataset(
        statistics=statistics,
        samples=tuple(partition_samples),
        history_length=history_length,
        horizon=horizon,
    )


def _config_sha256(
    world: PublishedWorldAdapter,
    qualification: QualificationConfig,
) -> str:
    payload = {
        "world": world.config.model_dump(mode="json"),
        "qualification": qualification.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_seed(
    world_id: str,
    qualification_seed: int,
    partition: QualificationPartition,
    index: int,
) -> int:
    payload = f"{world_id}|{qualification_seed}|{partition.value}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def generate_world_split(
    world: PublishedWorldAdapter,
    qualification: QualificationConfig,
    qualification_seed: int,
    source_commit: str,
) -> QualificationSplit:
    from tarca.stage1b.splits import build_qualification_split

    if qualification_seed not in qualification.qualification_seeds:
        raise ValueError("generation seed is outside the qualification namespace")
    counts = {
        QualificationPartition.QUAL_TRAIN: qualification.trajectories_per_partition.qual_train,
        QualificationPartition.QUAL_TUNE: qualification.trajectories_per_partition.qual_tune,
        QualificationPartition.QUAL_SEEN: qualification.trajectories_per_partition.qual_seen,
        QualificationPartition.QUAL_UNSEEN: qualification.trajectories_per_partition.qual_unseen,
    }
    regimes = {
        RegimeSplitRole.SEEN: tuple(
            regime for regime in world.config.regimes if regime.split_role is RegimeSplitRole.SEEN
        ),
        RegimeSplitRole.UNSEEN: tuple(
            regime for regime in world.config.regimes if regime.split_role is RegimeSplitRole.UNSEEN
        ),
    }
    config_sha256 = _config_sha256(world, qualification)
    records: list[TrajectoryRecord] = []
    for partition in QualificationPartition:
        split_role = (
            RegimeSplitRole.UNSEEN
            if partition is QualificationPartition.QUAL_UNSEEN
            else RegimeSplitRole.SEEN
        )
        eligible_regimes = regimes[split_role]
        if not eligible_regimes:
            raise ValueError(f"world has no {split_role.value} qualification regime")
        for index in range(counts[partition]):
            regime = eligible_regimes[index % len(eligible_regimes)]
            seed = _trajectory_seed(
                world.config.world_id,
                qualification_seed,
                partition,
                index,
            )
            trajectory = world.simulate(
                SimulationRequest(
                    seed=seed,
                    partition=partition,
                    regime_id=regime.regime_id,
                    length=qualification.trajectory_length,
                    warmup_steps=qualification.warmup_steps,
                )
            )
            identity = hashlib.sha256(
                (
                    f"{world.config.world_id}|{qualification_seed}|{partition.value}|{index}|{seed}"
                ).encode()
            ).hexdigest()[:24]
            records.append(
                TrajectoryRecord(
                    trajectory_id=f"{world.config.world_id}-{identity}",
                    world_id=world.config.world_id,
                    family_id=world.config.family_id,
                    regime_id=regime.regime_id,
                    partition=partition,
                    seed=seed,
                    graph_sha256=trajectory.truth.graph_sha256,
                    future_noise_sha256=trajectory.future_noise_sha256,
                    source_commit=source_commit,
                    config_sha256=config_sha256,
                    values=trajectory.values.clone(),
                )
            )
    return build_qualification_split(tuple(records))
