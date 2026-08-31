from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from tarca.contracts import (
    AccessScope,
    DatasetSpec,
    DatasetWindowPartition,
    SealedAccessGrant,
    validate_sealed_access,
)
from tarca.stage1b.config import QualificationPartition, RegimeSplitRole
from tarca.stage1b.worlds import PublishedWorldAdapter, SimulationRequest
from tarca.stage2.config import Stage2Config

if TYPE_CHECKING:
    from tarca.e02.config import E02Config

_DATASET = DatasetSpec(name="lorenz96_twoscale_v2", version="e02-v1")
_FORMAL_SCOPE = AccessScope(sealed=True, scope_name="e02_predictor_validity_v1-formal")
_PARTITION_ORDER = (
    DatasetWindowPartition.TRAIN,
    DatasetWindowPartition.VALIDATION,
    DatasetWindowPartition.TEST_SEEN_REGIME,
    DatasetWindowPartition.TEST_UNSEEN_REGIME,
)


@dataclass(frozen=True, slots=True)
class Stage2Trajectory:
    trajectory_id: str
    world_id: str
    regime_id: str
    partition: DatasetWindowPartition
    data_seed: int
    trajectory_seed: int
    source_commit: str
    config_sha256: str
    values: Tensor

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.trajectory_id, self.world_id, self.regime_id)):
            raise ValueError("trajectory identity fields must not be blank")
        if type(self.data_seed) is not int or type(self.trajectory_seed) is not int:
            raise ValueError("trajectory seeds must be integers")
        if min(self.data_seed, self.trajectory_seed) <= 0:
            raise ValueError("trajectory seeds must be positive")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("trajectory source commit must be a Git SHA-1")
        if re.fullmatch(r"[0-9a-f]{64}", self.config_sha256) is None:
            raise ValueError("trajectory config identity must be a SHA-256")
        if self.values.ndim != 2 or self.values.shape[0] < 2 or self.values.shape[1] < 1:
            raise ValueError("trajectory values must be a nonempty rank-two time series")
        if not self.values.is_floating_point() or not bool(torch.isfinite(self.values).all()):
            raise ValueError("trajectory values must be finite floating-point values")


@dataclass(frozen=True, slots=True)
class Stage2NormalizationStatistics:
    mean: Tensor
    standard_deviation: Tensor
    fitted_partition: DatasetWindowPartition
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fitted_partition is not DatasetWindowPartition.TRAIN:
            raise ValueError("Stage 2 normalizer must be fitted on TRAIN")
        if self.mean.ndim != 1 or self.standard_deviation.shape != self.mean.shape:
            raise ValueError("normalizer statistics must be aligned vectors")
        if not bool(torch.isfinite(self.mean).all()) or not bool(
            torch.isfinite(self.standard_deviation).all()
        ):
            raise ValueError("normalizer statistics must be finite")
        if bool((self.standard_deviation <= 0).any()):
            raise ValueError("normalizer standard deviation must be positive")
        if not self.trajectory_ids or len(self.trajectory_ids) != len(set(self.trajectory_ids)):
            raise ValueError("normalizer TRAIN trajectory IDs must be nonempty and unique")


@dataclass(frozen=True, slots=True)
class Stage2WindowLineage:
    window_id: str
    trajectory_id: str
    regime_id: str
    partition: DatasetWindowPartition
    data_seed: int
    trajectory_seed: int
    history_start: int
    history_end: int
    target_end: int


@dataclass(frozen=True, slots=True)
class Stage2WindowSample:
    history: Tensor
    target: Tensor
    lineage: Stage2WindowLineage


@dataclass(frozen=True, slots=True)
class Stage2WindowSet:
    partition: DatasetWindowPartition
    samples: tuple[Stage2WindowSample, ...]


@dataclass(frozen=True, slots=True)
class Stage2DataBundle:
    dataset_id: str
    records: tuple[Stage2Trajectory, ...]
    window_sets: tuple[Stage2WindowSet, ...]
    normalizer: Stage2NormalizationStatistics
    history: int
    horizon: int
    manifest_sha256: str

    def trajectory_ids(self, partition: DatasetWindowPartition) -> tuple[str, ...]:
        return tuple(
            record.trajectory_id for record in self.records if record.partition is partition
        )

    def trajectory_count(self, partition: DatasetWindowPartition) -> int:
        return len(self.trajectory_ids(partition))

    def for_partition(self, partition: DatasetWindowPartition) -> tuple[Stage2WindowSample, ...]:
        for window_set in self.window_sets:
            if window_set.partition is partition:
                return window_set.samples
        raise KeyError(partition)


@dataclass(frozen=True, slots=True)
class _DevelopmentTrajectoryWork:
    world: PublishedWorldAdapter
    data_seed: int
    partition: DatasetWindowPartition
    qualification_partition: QualificationPartition
    index: int
    regime_id: str
    length: int
    warmup_steps: int
    source_commit: str
    config_sha256: str


def _development_trajectory_seed(work: _DevelopmentTrajectoryWork) -> int:
    payload = (
        f"{work.world.config.world_id}|stage2|{work.data_seed}|"
        f"{work.partition.value}|{work.index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1) + 1


def _generate_development_trajectory(work: _DevelopmentTrajectoryWork) -> Stage2Trajectory:
    trajectory_seed = _development_trajectory_seed(work)
    generated = work.world.simulate(
        SimulationRequest(
            seed=trajectory_seed,
            partition=work.qualification_partition,
            regime_id=work.regime_id,
            length=work.length,
            warmup_steps=work.warmup_steps,
        )
    )
    identity = hashlib.sha256(
        (
            f"{work.world.config.world_id}|stage2|{work.data_seed}|"
            f"{work.partition.value}|{work.index}|{trajectory_seed}"
        ).encode()
    ).hexdigest()[:24]
    return Stage2Trajectory(
        trajectory_id=f"{work.world.config.world_id}-stage2-{identity}",
        world_id=work.world.config.world_id,
        regime_id=work.regime_id,
        partition=work.partition,
        data_seed=work.data_seed,
        trajectory_seed=trajectory_seed,
        source_commit=work.source_commit,
        config_sha256=work.config_sha256,
        values=generated.values.clone(),
    )


def _limit_generation_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def generate_development_bundle(
    config: Stage2Config,
    world: PublishedWorldAdapter,
    *,
    worker_count: int,
) -> Stage2DataBundle:
    if type(worker_count) is not int or not (
        1 <= worker_count <= config.runtime_profile.maximum_work_cores
    ):
        raise ValueError("development generation worker count must be between 1 and 24")
    if world.config.world_id != config.upstream.world_id:
        raise ValueError("development world identity does not match the frozen Stage 2 world")
    seen_regimes = tuple(
        regime
        for regime in world.config.regimes
        if regime.split_role is RegimeSplitRole.SEEN
    )
    if not seen_regimes:
        raise ValueError("Stage 2 development world requires at least one seen regime")
    partition_design = (
        (
            DatasetWindowPartition.TRAIN,
            QualificationPartition.QUAL_TRAIN,
            config.data.train_trajectories_per_seed,
        ),
        (
            DatasetWindowPartition.VALIDATION,
            QualificationPartition.QUAL_TUNE,
            config.data.validation_trajectories_per_seed,
        ),
    )
    source_commit = config.source("scoring_rules_l96").commit
    config_sha256 = config.scientific_hash()
    work_items = tuple(
        _DevelopmentTrajectoryWork(
            world=world,
            data_seed=data_seed,
            partition=partition,
            qualification_partition=qualification_partition,
            index=index,
            regime_id=seen_regimes[index % len(seen_regimes)].regime_id,
            length=config.data.trajectory_length,
            warmup_steps=config.data.warmup_steps,
            source_commit=source_commit,
            config_sha256=config_sha256,
        )
        for data_seed in config.data.development_seeds
        for partition, qualification_partition, count in partition_design
        for index in range(count)
    )
    resolved_workers = min(worker_count, len(work_items))
    if resolved_workers == 1:
        records = tuple(_generate_development_trajectory(item) for item in work_items)
    else:
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            initializer=_limit_generation_worker_threads,
        ) as executor:
            records = tuple(
                executor.map(_generate_development_trajectory, work_items, chunksize=1)
            )
    return prepare_stage2_bundle(
        records,
        history=config.data.history,
        horizon=config.data.horizon,
    )


def _normalizer(records: tuple[Stage2Trajectory, ...]) -> Stage2NormalizationStatistics:
    training = tuple(
        record for record in records if record.partition is DatasetWindowPartition.TRAIN
    )
    if not training:
        raise ValueError("TRAIN trajectories are required to fit the Stage 2 normalizer")
    values = torch.cat(tuple(record.values.to(torch.float64) for record in training), dim=0)
    return Stage2NormalizationStatistics(
        mean=values.mean(dim=0).clone(),
        standard_deviation=values.std(dim=0, unbiased=False).clamp_min(1e-8).clone(),
        fitted_partition=DatasetWindowPartition.TRAIN,
        trajectory_ids=tuple(record.trajectory_id for record in training),
    )


def _window_id(record: Stage2Trajectory, start: int, history: int, horizon: int) -> str:
    payload = (
        f"{record.trajectory_id}|{record.partition.value}|{start}|{history}|{horizon}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _windows_for_record(
    record: Stage2Trajectory,
    normalizer: Stage2NormalizationStatistics,
    history: int,
    horizon: int,
    stride: int,
) -> tuple[Stage2WindowSample, ...]:
    normalized = (
        record.values.to(torch.float64) - normalizer.mean
    ) / normalizer.standard_deviation
    last_start = normalized.shape[0] - history - horizon
    if last_start < 0:
        raise ValueError("trajectory is shorter than the requested history and horizon")
    samples: list[Stage2WindowSample] = []
    for start in range(0, last_start + 1, stride):
        history_end = start + history
        target_end = history_end + horizon
        lineage = Stage2WindowLineage(
            window_id=_window_id(record, start, history, horizon),
            trajectory_id=record.trajectory_id,
            regime_id=record.regime_id,
            partition=record.partition,
            data_seed=record.data_seed,
            trajectory_seed=record.trajectory_seed,
            history_start=start,
            history_end=history_end,
            target_end=target_end,
        )
        samples.append(
            Stage2WindowSample(
                history=normalized[start:history_end].to(torch.float32).clone(),
                target=normalized[history_end:target_end].to(torch.float32).clone(),
                lineage=lineage,
            )
        )
    return tuple(samples)


def _manifest_hash(
    records: tuple[Stage2Trajectory, ...],
    normalizer: Stage2NormalizationStatistics,
    history: int,
    horizon: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"stage2-data-v1|{history}|{horizon}".encode())
    for record in records:
        metadata = (
            record.trajectory_id,
            record.world_id,
            record.regime_id,
            record.partition.value,
            str(record.data_seed),
            str(record.trajectory_seed),
            record.source_commit,
            record.config_sha256,
        )
        digest.update("|".join(metadata).encode())
        digest.update(record.values.detach().cpu().contiguous().numpy().tobytes())
    digest.update(normalizer.mean.detach().cpu().contiguous().numpy().tobytes())
    digest.update(normalizer.standard_deviation.detach().cpu().contiguous().numpy().tobytes())
    digest.update("|".join(normalizer.trajectory_ids).encode())
    return digest.hexdigest()


def prepare_stage2_bundle(
    records: tuple[Stage2Trajectory, ...],
    *,
    history: int,
    horizon: int,
    stride: int = 1,
    normalizer: Stage2NormalizationStatistics | None = None,
) -> Stage2DataBundle:
    if not records:
        raise ValueError("Stage 2 data bundle requires trajectories")
    if min(history, horizon, stride) <= 0:
        raise ValueError("history, horizon, and stride must be positive")
    identities = tuple(record.trajectory_id for record in records)
    if len(identities) != len(set(identities)):
        raise ValueError("Stage 2 trajectory IDs must be unique")
    dimensions = {record.values.shape[1] for record in records}
    if len(dimensions) != 1:
        raise ValueError("Stage 2 trajectories must share a feature dimension")
    resolved_normalizer = normalizer or _normalizer(records)
    if resolved_normalizer.mean.shape[0] != next(iter(dimensions)):
        raise ValueError("normalizer feature dimension does not match trajectories")
    window_sets = tuple(
        Stage2WindowSet(
            partition=partition,
            samples=tuple(
                sample
                for record in records
                if record.partition is partition
                for sample in _windows_for_record(
                    record,
                    resolved_normalizer,
                    history,
                    horizon,
                    stride,
                )
            ),
        )
        for partition in _PARTITION_ORDER
        if any(record.partition is partition for record in records)
    )
    return Stage2DataBundle(
        dataset_id="lorenz96_twoscale_v2",
        records=records,
        window_sets=window_sets,
        normalizer=resolved_normalizer,
        history=history,
        horizon=horizon,
        manifest_sha256=_manifest_hash(records, resolved_normalizer, history, horizon),
    )


def stack_partition(
    bundle: Stage2DataBundle,
    partition: DatasetWindowPartition,
) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    samples = bundle.for_partition(partition)
    if not samples:
        raise ValueError("cannot stack an empty Stage 2 partition")
    return (
        torch.stack(tuple(sample.history for sample in samples)),
        torch.stack(tuple(sample.target for sample in samples)),
        tuple(sample.lineage.trajectory_id for sample in samples),
    )


def _read_formal_storage() -> tuple[Stage2Trajectory, ...]:
    raise FileNotFoundError("formal E02 storage has not been configured")


def _validate_formal_counts(records: tuple[Stage2Trajectory, ...], config: E02Config) -> None:
    allowed = {
        DatasetWindowPartition.TEST_SEEN_REGIME: config.trajectories_seen_per_seed,
        DatasetWindowPartition.TEST_UNSEEN_REGIME: config.trajectories_unseen_per_seed,
    }
    if any(record.partition not in allowed for record in records):
        raise ValueError("formal E02 records contain a non-formal partition")
    for seed in config.formal_seeds:
        for partition, expected in allowed.items():
            observed = sum(
                record.data_seed == seed and record.partition is partition for record in records
            )
            if observed != expected:
                raise ValueError("formal E02 trajectory counts do not match the frozen design")
    if len(records) != config.gate.required_completed_trajectories:
        raise ValueError("formal E02 must contain exactly 120 trajectories")


def open_formal_bundle(
    stage2_config: Stage2Config,
    e02_config: E02Config,
    grant: SealedAccessGrant | None,
    *,
    accessed_at: datetime,
    normalizer: Stage2NormalizationStatistics | None = None,
) -> Stage2DataBundle:
    if stage2_config.upstream.world_id != _DATASET.name:
        raise ValueError("Stage 2 world identity does not match the E02 dataset")
    for partition in (
        DatasetWindowPartition.TEST_SEEN_REGIME,
        DatasetWindowPartition.TEST_UNSEEN_REGIME,
    ):
        validate_sealed_access(_DATASET, partition, _FORMAL_SCOPE, grant, accessed_at)
    if normalizer is None:
        raise ValueError("formal E02 access requires the frozen Stage 2 TRAIN normalizer")
    records = _read_formal_storage()
    _validate_formal_counts(records, e02_config)
    return prepare_stage2_bundle(
        records,
        history=stage2_config.data.history,
        horizon=stage2_config.data.horizon,
        normalizer=normalizer,
    )
