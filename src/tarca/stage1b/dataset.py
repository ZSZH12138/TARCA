from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tarca.contracts import (
    DatasetWindowPartition,
    WindowBatch,
    audit_partition_isolation,
    validate_window_batch,
)
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


_PHYSICAL_PARTITIONS = {
    QualificationPartition.QUAL_TRAIN: DatasetWindowPartition.TRAIN,
    QualificationPartition.QUAL_TUNE: DatasetWindowPartition.VALIDATION,
    QualificationPartition.QUAL_SEEN: DatasetWindowPartition.TEST_SEEN_REGIME,
    QualificationPartition.QUAL_UNSEEN: DatasetWindowPartition.TEST_UNSEEN_REGIME,
}


def partition_for_qualification(
    partition: QualificationPartition,
) -> DatasetWindowPartition:
    try:
        return _PHYSICAL_PARTITIONS[partition]
    except KeyError as error:
        raise ValueError("unknown qualification partition") from error


_TRUTH_METADATA_FRAGMENTS = (
    "truth",
    "oracle",
    "future_noise",
    "shock_sequence",
    "latent_concept",
    "true_graph",
    "true_lag",
)


def validate_qualification_window(batch: WindowBatch) -> WindowBatch:
    validated = validate_window_batch(batch)
    reserved = tuple(
        key
        for key in validated.metadata
        if any(fragment in key.lower() for fragment in _TRUTH_METADATA_FRAGMENTS)
    )
    if reserved:
        raise ValueError(
            "truth and oracle metadata are forbidden in qualification windows: "
            f"{', '.join(sorted(reserved))}"
        )
    return validated


@dataclass(frozen=True, slots=True)
class QualificationWindowBridge:
    entries: tuple[tuple[QualificationPartition, DatasetWindowPartition, WindowBatch], ...]
    normalization_mean: torch.Tensor
    normalization_standard_deviation: torch.Tensor
    fitted_partition: DatasetWindowPartition

    def batch_for(self, partition: QualificationPartition) -> WindowBatch:
        matches = tuple(batch for candidate, _, batch in self.entries if candidate is partition)
        if len(matches) != 1:
            raise KeyError(partition)
        return matches[0]


def bridge_qualification_windows(
    batches: Mapping[QualificationPartition, WindowBatch],
) -> QualificationWindowBridge:
    if set(batches) != set(QualificationPartition):
        raise ValueError("qualification bridge requires exactly four registered partitions")
    entries: list[tuple[QualificationPartition, DatasetWindowPartition, WindowBatch]] = []
    physical_batches: dict[DatasetWindowPartition, WindowBatch] = {}
    for partition in QualificationPartition:
        physical = partition_for_qualification(partition)
        batch = validate_qualification_window(batches[partition])
        if batch.metadata.get("qualification_partition") != partition.value:
            raise ValueError("qualification window metadata has the wrong qualification partition")
        if batch.metadata.get("physical_partition") != physical.value:
            raise ValueError("qualification window metadata has the wrong physical partition")
        entries.append((partition, physical, batch))
        physical_batches[physical] = batch
    audit = audit_partition_isolation(physical_batches)
    if not audit.passed:
        raise ValueError(f"partition isolation failed: {'; '.join(audit.findings)}")
    train = batches[QualificationPartition.QUAL_TRAIN].x.to(torch.float64)
    flattened = train.reshape(-1, train.shape[-1])
    mean = flattened.mean(dim=0)
    deviation = flattened.std(dim=0, unbiased=False).clamp_min(1e-8)
    return QualificationWindowBridge(
        entries=tuple(entries),
        normalization_mean=mean.to(train.dtype).clone(),
        normalization_standard_deviation=deviation.to(train.dtype).clone(),
        fitted_partition=DatasetWindowPartition.TRAIN,
    )


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


@dataclass(frozen=True, slots=True)
class _TrajectoryWork:
    world: PublishedWorldAdapter
    qualification_seed: int
    partition: QualificationPartition
    index: int
    regime_id: str
    length: int
    warmup_steps: int
    source_commit: str
    config_sha256: str


def _limit_generation_worker_threads() -> None:
    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_variables:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _generate_trajectory_record(work: _TrajectoryWork) -> TrajectoryRecord:
    seed = _trajectory_seed(
        work.world.config.world_id,
        work.qualification_seed,
        work.partition,
        work.index,
    )
    trajectory = work.world.simulate(
        SimulationRequest(
            seed=seed,
            partition=work.partition,
            regime_id=work.regime_id,
            length=work.length,
            warmup_steps=work.warmup_steps,
        )
    )
    identity = hashlib.sha256(
        (
            f"{work.world.config.world_id}|{work.qualification_seed}|"
            f"{work.partition.value}|{work.index}|{seed}"
        ).encode()
    ).hexdigest()[:24]
    return TrajectoryRecord(
        trajectory_id=f"{work.world.config.world_id}-{identity}",
        world_id=work.world.config.world_id,
        family_id=work.world.config.family_id,
        regime_id=work.regime_id,
        partition=work.partition,
        seed=seed,
        graph_sha256=trajectory.truth.graph_sha256,
        future_noise_sha256=trajectory.future_noise_sha256,
        source_commit=work.source_commit,
        config_sha256=work.config_sha256,
        values=trajectory.values.clone(),
    )


def generate_world_split(
    world: PublishedWorldAdapter,
    qualification: QualificationConfig,
    qualification_seed: int,
    source_commit: str,
    *,
    worker_count: int = 1,
) -> QualificationSplit:
    from tarca.stage1b.splits import build_qualification_split

    if qualification_seed not in qualification.qualification_seeds:
        raise ValueError("generation seed is outside the qualification namespace")
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("generation worker count must be a positive integer")
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
    work_items: list[_TrajectoryWork] = []
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
            work_items.append(
                _TrajectoryWork(
                    world=world,
                    qualification_seed=qualification_seed,
                    partition=partition,
                    index=index,
                    regime_id=regime.regime_id,
                    length=qualification.trajectory_length,
                    warmup_steps=qualification.warmup_steps,
                    source_commit=source_commit,
                    config_sha256=config_sha256,
                )
            )
    resolved_workers = min(worker_count, len(work_items))
    if resolved_workers == 1:
        records = tuple(_generate_trajectory_record(item) for item in work_items)
    else:
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            initializer=_limit_generation_worker_threads,
        ) as executor:
            records = tuple(executor.map(_generate_trajectory_record, work_items, chunksize=1))
    return build_qualification_split(tuple(records))
