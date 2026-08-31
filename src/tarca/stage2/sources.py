from __future__ import annotations

from pathlib import Path

from tarca.stage1b.config import SourceConfig
from tarca.stage1b.source_capsules import (
    SourceCapsuleReceipt,
    build_source_capsule,
    import_source_capsule,
)
from tarca.stage1b.sources import (
    GitRunner,
    MaterializedSources,
    SourceAcquisitionMode,
    SourceMaterializationReceipt,
    SubprocessGitRunner,
    materialize_source,
)
from tarca.stage2.config import Stage2Config
from tarca.stage2.dlinear import DLinearModelConfig

_SOURCE_IDS = ("dlinear", "itransformer", "patchtst", "scoring_rules_l96")


def stage2_sources(config: Stage2Config) -> tuple[SourceConfig, ...]:
    if tuple(source.source_id for source in config.sources) != _SOURCE_IDS:
        raise ValueError("Stage 2 requires the exact frozen four-source set")
    return config.sources


def dlinear_model_config(config: Stage2Config, *, dimension: int) -> DLinearModelConfig:
    source = config.source("dlinear")
    assets = tuple(asset for asset in source.assets if asset.asset_id == "dlinear_model")
    if len(assets) != 1:
        raise ValueError("Stage 2 DLinear source must contain exactly one model asset")
    model = config.model("DLINEAR")
    return DLinearModelConfig(
        sequence_length=config.data.history,
        prediction_length=config.data.horizon,
        dimension=dimension,
        individual=bool(model.parameter("individual")),
        moving_average_kernel=int(model.parameter("moving_average_kernel")),
        asset_relative_path=assets[0].relative_path,
        asset_sha256=assets[0].sha256,
    )


def materialize_stage2_sources(
    config: Stage2Config,
    cache_root: Path,
    *,
    source_ids: tuple[str, ...] = (),
    mode: SourceAcquisitionMode = SourceAcquisitionMode.ONLINE,
    runner: GitRunner | None = None,
) -> MaterializedSources:
    registered = {source.source_id: source for source in stage2_sources(config)}
    unknown = tuple(sorted(set(source_ids) - set(registered)))
    if unknown:
        raise ValueError(f"unknown Stage 2 source IDs: {', '.join(unknown)}")
    selected = (
        stage2_sources(config)
        if not source_ids
        else tuple(registered[source_id] for source_id in source_ids)
    )
    resolved_runner = runner or SubprocessGitRunner.discover()
    return MaterializedSources(
        receipts=tuple(
            materialize_source(
                source,
                cache_root.resolve(),
                resolved_runner,
                mode=mode,
            )
            for source in selected
        )
    )


def build_stage2_source_capsule(
    config: Stage2Config,
    cache_root: Path,
    output_path: Path,
    *,
    runner: GitRunner | None = None,
) -> SourceCapsuleReceipt:
    return build_source_capsule(
        stage2_sources(config),
        cache_root.resolve(),
        output_path.resolve(),
        runner or SubprocessGitRunner.discover(),
    )


def import_stage2_source_capsule(
    config: Stage2Config,
    capsule_path: Path,
    receipt_path: Path,
    cache_root: Path,
    *,
    runner: GitRunner | None = None,
) -> tuple[SourceMaterializationReceipt, ...]:
    return import_source_capsule(
        stage2_sources(config),
        capsule_path.resolve(),
        receipt_path.resolve(),
        cache_root.resolve(),
        runner or SubprocessGitRunner.discover(),
    )
