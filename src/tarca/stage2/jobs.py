from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from tarca.artifacts import LocalArtifactStore
from tarca.contracts import (
    ArtifactRef,
    DatasetWindowPartition,
    WindowBatch,
    canonical_json_bytes,
    canonical_json_hash,
    sha256_file,
)
from tarca.execution import ExecutionContext, ExecutorRegistry, ProgressSink, TaskSpec
from tarca.stage1b.config import load_world_suite
from tarca.stage1b.metrics import gaussian_crps
from tarca.stage1b.modeling import OfficialITransformerPredictor, OfficialPatchTSTPredictor
from tarca.stage1b.sources import SourceAcquisitionMode
from tarca.stage1b.worlds import build_world
from tarca.stage2.baselines import LastValueGaussian, SeasonalNaiveGaussian, Stage2VARGaussian
from tarca.stage2.config import Stage2Config, load_stage2_config
from tarca.stage2.data import (
    Stage2DataBundle,
    Stage2NormalizationStatistics,
    Stage2Trajectory,
    generate_development_bundle,
    prepare_stage2_bundle,
    stack_partition,
)
from tarca.stage2.dlinear import (
    DLinearGaussian,
    dlinear_state_sha256,
    fit_dlinear_cross_fitted,
    load_official_dlinear,
)
from tarca.stage2.freeze import freeze_stage2_suite
from tarca.stage2.manifest import Stage2CompilationInputs, compile_stage2_manifest
from tarca.stage2.selection import (
    ModelSelection,
    ValidationScore,
    select_primary_initialization,
    select_strongest_linear,
)
from tarca.stage2.sources import dlinear_model_config
from tarca.stage2.training import (
    Stage2TrainingPolicy,
    forecast_fixed_batch_on_model_device,
    train_stage2_neural,
)

_SCHEMA = "tarca-stage2-job-v1"


def _artifact_root(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2").strip()
    root = (repository_root.resolve() / raw).resolve()
    if repository_root.resolve() not in root.parents:
        raise ValueError("Stage 2 artifact root must stay inside the repository")
    return root


def stage2_artifact_store(
    repository_root: Path, task: TaskSpec | None = None
) -> LocalArtifactStore:
    relative = (_artifact_root(repository_root) / "runtime/store").relative_to(
        repository_root.resolve()
    )
    return LocalArtifactStore(
        repository_root,
        producer_stage="stage2",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash="0" * 64 if task is None else canonical_json_hash(task.identity),
        dependencies=() if task is None else task.inputs,
        store_relative_root=relative.as_posix(),
    )


def _config(root: Path) -> Stage2Config:
    return load_stage2_config(root / "configs/stage2/stage2_v1.yaml")


def _neural_runtime_device() -> torch.device:
    """Select the single GPU exposed to a GPU-scoped worker."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _publish_json(root: Path, task: TaskSpec, value: object) -> ArtifactRef:
    return stage2_artifact_store(root, task).publish_bytes(
        canonical_json_bytes(value) + b"\n",
        task.output_artifact_type,
        "application/json",
        _SCHEMA,
    )


def _publish_torch(root: Path, task: TaskSpec, value: dict[str, Any]) -> ArtifactRef:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return stage2_artifact_store(root, task).publish_bytes(
        buffer.getvalue(), task.output_artifact_type, "application/x-pytorch", _SCHEMA
    )


def _load_torch(root: Path, ref: ArtifactRef) -> dict[str, Any]:
    value = torch.load(
        io.BytesIO(stage2_artifact_store(root).load_bytes(ref)),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(value, dict):
        raise ValueError("Stage 2 tensor artifact must contain a mapping")
    return cast(dict[str, Any], value)


def _load_json(root: Path, ref: ArtifactRef) -> dict[str, Any]:
    value = json.loads(stage2_artifact_store(root).load_bytes(ref))
    if not isinstance(value, dict):
        raise ValueError("Stage 2 JSON artifact must contain an object")
    return cast(dict[str, Any], value)


def _bundle_payload(bundle: Stage2DataBundle) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA,
        "records": [
            {
                "trajectory_id": item.trajectory_id,
                "world_id": item.world_id,
                "regime_id": item.regime_id,
                "partition": item.partition.value,
                "data_seed": item.data_seed,
                "trajectory_seed": item.trajectory_seed,
                "source_commit": item.source_commit,
                "config_sha256": item.config_sha256,
                "values": item.values.detach().cpu(),
            }
            for item in bundle.records
        ],
        "normalizer": {
            "mean": bundle.normalizer.mean.detach().cpu(),
            "standard_deviation": bundle.normalizer.standard_deviation.detach().cpu(),
            "fitted_partition": bundle.normalizer.fitted_partition.value,
            "trajectory_ids": bundle.normalizer.trajectory_ids,
        },
        "history": bundle.history,
        "horizon": bundle.horizon,
        "manifest_sha256": bundle.manifest_sha256,
    }


def _bundle_from_payload(payload: dict[str, Any]) -> Stage2DataBundle:
    raw = cast(dict[str, Any], payload["normalizer"])
    normalizer = Stage2NormalizationStatistics(
        mean=cast(Tensor, raw["mean"]),
        standard_deviation=cast(Tensor, raw["standard_deviation"]),
        fitted_partition=DatasetWindowPartition(raw["fitted_partition"]),
        trajectory_ids=tuple(raw["trajectory_ids"]),
    )
    records = tuple(
        Stage2Trajectory(
            trajectory_id=item["trajectory_id"],
            world_id=item["world_id"],
            regime_id=item["regime_id"],
            partition=DatasetWindowPartition(item["partition"]),
            data_seed=int(item["data_seed"]),
            trajectory_seed=int(item["trajectory_seed"]),
            source_commit=item["source_commit"],
            config_sha256=item["config_sha256"],
            values=item["values"],
        )
        for item in payload["records"]
    )
    bundle = prepare_stage2_bundle(
        records,
        history=int(payload["history"]),
        horizon=int(payload["horizon"]),
        normalizer=normalizer,
    )
    if bundle.manifest_sha256 != payload["manifest_sha256"]:
        raise ValueError("Stage 2 data manifest drifted during artifact reload")
    return bundle


def _data_input(root: Path, task: TaskSpec) -> Stage2DataBundle:
    refs = tuple(ref for ref in task.inputs if ref.artifact_type == "STAGE2_DEVELOPMENT_DATA")
    if len(refs) != 1:
        raise ValueError("Stage 2 job requires exactly one development data artifact")
    return _bundle_from_payload(_load_torch(root, refs[0]))


def _target_names(bundle: Stage2DataBundle) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(bundle.records[0].values.shape[1]))


def _window_batch(bundle: Stage2DataBundle) -> WindowBatch:
    partition = DatasetWindowPartition.VALIDATION
    samples = bundle.for_partition(partition)
    x, y, _ = stack_partition(bundle, partition)
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    names = _target_names(bundle)
    return WindowBatch(
        x=x,
        y=y,
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=None,
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=tuple(item.lineage.window_id for item in samples),
        input_feature_names=names,
        target_names=names,
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=tuple(origin for _ in samples),
        feature_end=tuple(origin + timedelta(hours=bundle.history - 1) for _ in samples),
        prediction_start=tuple(origin + timedelta(hours=bundle.history) for _ in samples),
        label_end=tuple(
            origin + timedelta(hours=bundle.history + bundle.horizon - 1) for _ in samples
        ),
        forecast_time=tuple(
            tuple(origin + timedelta(hours=bundle.history + step) for step in range(bundle.horizon))
            for _ in samples
        ),
        metadata={"partition": partition.value},
    )


def verify_upstream_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    config = _config(root)
    expected = {
        "STAGE1B": config.upstream.stage1b_manifest_sha256,
        "E01": config.upstream.e01_receipt_sha256,
    }[task.identity.model_id]
    if len(task.inputs) != 1 or task.inputs[0].content_hash != expected:
        raise ValueError("upstream handoff hash does not match the frozen config")
    ref = task.inputs[0]
    if ref.relative_path is not None:
        path = root / ref.relative_path
        if task.identity.model_id == "E01":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("receipt_sha256") != expected:
                raise ValueError("E01 handoff receipt identity verification failed")
        elif sha256_file(path) != expected:
            raise ValueError("upstream handoff file hash verification failed")
    return _publish_json(root, task, {"status": "VERIFIED", "upstream_sha256": expected})


def verify_source_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    if len(task.inputs) != 1 or task.inputs[0].artifact_type != "STAGE2_SOURCE_CAPSULE":
        raise ValueError("source verification requires exactly one Stage 2 source capsule")
    capsule_ref = task.inputs[0]
    if capsule_ref.relative_path is None:
        raise ValueError("Stage 2 source capsule requires a repository-relative path")
    capsule_path = (root / capsule_ref.relative_path).resolve()
    if root.resolve() not in capsule_path.parents or not capsule_path.is_file():
        raise ValueError("Stage 2 source capsule is missing or outside the repository")
    if sha256_file(capsule_path) != capsule_ref.content_hash:
        raise ValueError("Stage 2 source capsule hash verification failed")
    source = _config(root).source(task.identity.data_id)
    checkout = root / "third_party/stage2" / source.source_id / source.commit
    if not checkout.is_dir():
        raise ValueError("pinned Stage 2 source checkout is missing")
    assets = []
    for asset in source.assets:
        observed = sha256_file(checkout / asset.relative_path)
        if observed != asset.sha256:
            raise ValueError("Stage 2 source asset hash verification failed")
        assets.append((asset.relative_path, observed))
    return _publish_json(
        root,
        task,
        {
            "status": "VERIFIED",
            "source_id": source.source_id,
            "commit": source.commit,
            "assets": assets,
        },
    )


def generate_development_data_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    config = _config(root)
    world = build_world(
        load_world_suite(root / "configs/stage1b/worlds_v2.yaml").world(config.upstream.world_id)
    )
    workers = min(int(os.environ.get("TARCA_STAGE2_DATA_WORKERS", "24")), 24)
    return _publish_torch(
        root,
        task,
        _bundle_payload(generate_development_bundle(config, world, worker_count=workers)),
    )


def _baseline_payload(model: object, model_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": _SCHEMA, "model_id": model_id}
    if isinstance(model, LastValueGaussian):
        payload.update(scale=model.scale, target_names=model.target_names)
    elif isinstance(model, SeasonalNaiveGaussian):
        payload.update(
            selected_lag=model.selected_lag,
            scale=model.scale,
            target_names=model.target_names,
            validation_crps=model.validation_crps,
        )
    elif isinstance(model, Stage2VARGaussian):
        payload.update(
            coefficients=model.coefficients,
            intercept=model.intercept,
            scale=model.scale,
            selected_lag=model.selected_lag,
            selected_ridge=model.selected_ridge,
            target_names=model.target_names,
            validation_crps=model.validation_crps,
        )
    elif isinstance(model, DLinearGaussian):
        payload.update(
            state_dict={
                name: value.detach().cpu() for name, value in model.mean_model.state_dict().items()
            },
            scale=model.scale,
            target_names=model.target_names,
            checkpoint_sha256=model.checkpoint_sha256,
        )
    else:
        raise TypeError("unsupported Stage 2 baseline")
    payload["model_sha256"] = cast(Any, model).model_hash
    return payload


def fit_baseline_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    config = _config(root)
    bundle = _data_input(root, task)
    train_x, train_y, trajectory_ids = stack_partition(bundle, DatasetWindowPartition.TRAIN)
    val_x, val_y, _ = stack_partition(bundle, DatasetWindowPartition.VALIDATION)
    names = _target_names(bundle)
    common = {
        "floor": config.training.scale_floor,
        "ceiling_multiplier": config.training.scale_ceiling_multiplier,
        "absolute_ceiling": config.training.scale_absolute_ceiling,
    }
    model_id = task.identity.model_id
    if model_id == "LAST_VALUE":
        model: object = LastValueGaussian.fit(train_x, train_y, names, **common)
    elif model_id == "SEASONAL_NAIVE":
        raw = config.model("SEASONAL_NAIVE").parameter("candidate_lags")
        model = SeasonalNaiveGaussian.fit(
            train_x,
            train_y,
            val_x,
            val_y,
            lags=tuple(raw),
            target_names=names,
            **common,
        )
    elif model_id == "VAR":
        cfg = config.model("VAR")
        model = Stage2VARGaussian.fit(
            train_x,
            train_y,
            val_x,
            val_y,
            lag_orders=tuple(cfg.parameter("lag_orders")),
            ridge_values=tuple(cfg.parameter("ridge")),
            target_names=names,
            **common,
        )
    elif model_id == "DLINEAR":
        cfg = config.model("DLINEAR")
        definition = dlinear_model_config(config, dimension=len(names))
        source = root / "third_party/stage2/dlinear" / config.source("dlinear").commit
        trained = fit_dlinear_cross_fitted(
            partial(load_official_dlinear, source, definition),
            train_x,
            train_y,
            trajectory_ids,
            val_x,
            val_y,
            target_names=names,
            fold_seeds=config.training.dlinear_fold_seeds,
            final_seed=config.training.initialization_seeds[0],
            batch_size=int(cfg.parameter("batch_size")),
            max_epochs=int(cfg.parameter("max_epochs")),
            patience=int(cfg.parameter("patience")),
            learning_rate=float(cfg.parameter("learning_rate")),
            weight_decay=config.training.dlinear_weight_decay,
            floor=config.training.scale_floor,
            ceiling_multiplier=config.training.scale_ceiling_multiplier,
            absolute_ceiling=config.training.scale_absolute_ceiling,
        )
        model = trained.predictor
    else:
        raise ValueError("baseline model is not allowlisted")
    return _publish_torch(root, task, _baseline_payload(model, model_id))


def _new_neural(root: Path, config: Stage2Config, model_id: str, dimension: int) -> Any:
    os.environ["TARCA_STAGE1B_SOURCE_CACHE_ROOT"] = "third_party/stage2"
    os.environ["TARCA_STAGE1B_SOURCE_MODE"] = SourceAcquisitionMode.OFFLINE_CAPSULE.value
    cfg = config.model(cast(Any, model_id))
    del root
    if model_id == "PATCHTST":
        return OfficialPatchTSTPredictor(
            history_length=config.data.history,
            horizon=config.data.horizon,
            input_dimension=dimension,
            d_model=int(cfg.parameter("d_model")),
            n_layers=int(cfg.parameter("n_layers")),
            n_heads=int(cfg.parameter("n_heads")),
            d_ff=int(cfg.parameter("d_ff")),
            dropout=float(cfg.parameter("dropout")),
            patch_length=int(cfg.parameter("patch_length")),
            patch_stride=int(cfg.parameter("patch_stride")),
        )
    if model_id == "ITRANSFORMER":
        return OfficialITransformerPredictor(
            history_length=config.data.history,
            horizon=config.data.horizon,
            input_dimension=dimension,
            d_model=int(cfg.parameter("d_model")),
            n_layers=int(cfg.parameter("n_layers")),
            n_heads=int(cfg.parameter("n_heads")),
            d_ff=int(cfg.parameter("d_ff")),
            dropout=float(cfg.parameter("dropout")),
        )
    raise ValueError("neural model is not allowlisted")


def train_neural_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context
    config = _config(root)
    bundle = _data_input(root, task)
    train_x, train_y, _ = stack_partition(bundle, DatasetWindowPartition.TRAIN)
    val_x, val_y, _ = stack_partition(bundle, DatasetWindowPartition.VALIDATION)
    model_id = task.identity.model_id
    cfg = config.model(cast(Any, model_id))
    model = _new_neural(root, config, model_id, train_x.shape[2])
    recovery_mode = os.environ.get("TARCA_STAGE2_RECOVERY_MODE", "")
    if recovery_mode not in {"", "DEVICE_MISMATCH_V1"}:
        raise ValueError("Stage 2 recovery mode is not allowlisted")
    policy = Stage2TrainingPolicy(
        model_id=cast(Any, model_id),
        device="cuda",
        precision=cast(Any, os.environ.get("TARCA_STAGE2_PRECISION", "FP32")),
        batch_size=int(cfg.parameter("batch_size")),
        max_epochs=int(cfg.parameter("max_epochs")),
        patience=int(cfg.parameter("patience")),
        learning_rate=float(cfg.parameter("learning_rate")),
        dataloader_workers=int(os.environ.get("TARCA_STAGE2_DATALOADER_WORKERS", "3")),
        checkpoint_root=_artifact_root(root) / "runtime/checkpoints",
    )
    result = train_stage2_neural(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        policy=policy,
        seed=task.identity.seed,
        progress=cast(Any, progress),
        resume_if_available=True,
        require_complete_resume=recovery_mode == "DEVICE_MISMATCH_V1",
    )
    if not result.completed:
        raise RuntimeError("Stage 2 neural training did not complete")
    return _publish_torch(
        root,
        task,
        {
            "schema_version": _SCHEMA,
            "model_id": model_id,
            "seed": task.identity.seed,
            "state_dict": {
                name: value.detach().cpu() for name, value in result.model.state_dict().items()
            },
            "model_sha256": result.model_sha256,
            "checkpoint_sha256": result.checkpoint_sha256,
            "fixed_batch_forecast_sha256": result.fixed_batch_forecast_sha256,
            "best_epoch": result.best_epoch,
            "best_validation_nll": result.best_validation_nll,
            "precision": result.precision,
        },
    )


def validate_checkpoint_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    ref = next(ref for ref in task.inputs if ref.artifact_type == "STAGE2_NEURAL_CHECKPOINT")
    payload = _load_torch(root, ref)
    bundle = _data_input(root, task)
    model = _new_neural(root, _config(root), payload["model_id"], bundle.records[0].values.shape[1])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(_neural_runtime_device())
    model.freeze()
    val_x, val_y, _ = stack_partition(bundle, DatasetWindowPartition.VALIDATION)
    distribution = forecast_fixed_batch_on_model_device(model, val_x[:2])
    if distribution.scale is None or distribution.mean.shape != val_y[:2].shape:
        raise ValueError("Stage 2 checkpoint probability shape is invalid")
    if not bool(
        torch.isfinite(distribution.mean).all() and torch.isfinite(distribution.scale).all()
    ) or bool((distribution.scale <= 0).any()):
        raise ValueError("Stage 2 checkpoint probabilities are invalid")
    payload["validated"] = True
    return _publish_torch(root, task, payload)


def _baseline_from_payload(root: Path, config: Stage2Config, value: dict[str, Any]) -> object:
    model_id = value["model_id"]
    if model_id == "LAST_VALUE":
        return LastValueGaussian(value["scale"], tuple(value["target_names"]))
    if model_id == "SEASONAL_NAIVE":
        return SeasonalNaiveGaussian(
            int(value["selected_lag"]),
            value["scale"],
            tuple(value["target_names"]),
            float(value["validation_crps"]),
        )
    if model_id == "VAR":
        return Stage2VARGaussian(
            value["coefficients"],
            value["intercept"],
            value["scale"],
            int(value["selected_lag"]),
            float(value["selected_ridge"]),
            tuple(value["target_names"]),
            float(value["validation_crps"]),
        )
    if model_id == "DLINEAR":
        definition = dlinear_model_config(config, dimension=len(value["target_names"]))
        source = root / "third_party/stage2/dlinear" / config.source("dlinear").commit
        model = load_official_dlinear(source, definition)
        model.load_state_dict(value["state_dict"], strict=True)
        if dlinear_state_sha256(model) != value["checkpoint_sha256"]:
            raise ValueError("DLinear checkpoint state hash drifted")
        return DLinearGaussian(
            model,
            value["scale"],
            tuple(value["target_names"]),
            value["checkpoint_sha256"],
        )
    raise ValueError("baseline artifact model is not allowlisted")


def predict_validation_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    config = _config(root)
    bundle = _data_input(root, task)
    batch = _window_batch(bundle)
    ref = next(ref for ref in task.inputs if ref.artifact_type != "STAGE2_DEVELOPMENT_DATA")
    value = _load_torch(root, ref)
    if ref.artifact_type == "STAGE2_PREDICTOR":
        model = _baseline_from_payload(root, config, value)
        distribution = cast(Any, model).predict_distribution(batch)
        model_sha256 = cast(Any, model).model_hash
    else:
        model = _new_neural(root, config, task.identity.model_id, batch.x.shape[2])
        model.load_state_dict(value["state_dict"], strict=True)
        model.freeze()
        with torch.no_grad():
            distribution = model.forward_distribution(batch.x)
        model_sha256 = value["model_sha256"]
    if distribution.scale is None or batch.y is None:
        raise ValueError("validation prediction requires Gaussian scale and targets")
    score = float(
        gaussian_crps(distribution.mean[:, :6], distribution.scale[:, :6], batch.y[:, :6]).mean()
    )
    return _publish_torch(
        root,
        task,
        {
            "schema_version": _SCHEMA,
            "model_id": task.identity.model_id,
            "seed": task.identity.seed,
            "model_sha256": model_sha256,
            "validation_crps": score,
            "artifact_ref": (
                f"VALIDATION/{task.identity.model_id.lower()}-{task.identity.seed}.pt"
            ),
        },
    )


def select_model_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    scores = tuple(
        ValidationScore(
            model_id=value["model_id"],
            seed=int(value["seed"]) or None,
            crps=float(value["validation_crps"]),
            artifact_ref=value["artifact_ref"],
        )
        for ref in task.inputs
        for value in (_load_torch(root, ref),)
    )
    selected = (
        select_strongest_linear(scores)
        if task.identity.model_id == "STRONGEST_LINEAR"
        else select_primary_initialization(
            "ITRANSFORMER", scores, seed_order=_config(root).training.initialization_seeds
        )
    )
    return _publish_json(root, task, asdict(selected))


def _normalizer_sha256(normalizer: Stage2NormalizationStatistics) -> str:
    digest = hashlib.sha256()
    digest.update(normalizer.mean.detach().cpu().contiguous().numpy().tobytes())
    digest.update(normalizer.standard_deviation.detach().cpu().contiguous().numpy().tobytes())
    digest.update("|".join(normalizer.trajectory_ids).encode())
    return digest.hexdigest()


def freeze_candidate_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    config = _config(root)
    data_ref = next(ref for ref in task.inputs if ref.artifact_type == "STAGE2_DEVELOPMENT_DATA")
    bundle = _bundle_from_payload(_load_torch(root, data_ref))
    predictions = tuple(
        (ref, _load_torch(root, ref))
        for ref in task.inputs
        if ref.artifact_type == "STAGE2_VALIDATION_PREDICTION"
    )
    by_model: dict[str, list[ArtifactRef]] = {}
    for ref, value in predictions:
        by_model.setdefault(value["model_id"], []).append(ref)
    baseline_refs = {
        value["model_id"]: ref
        for ref in task.inputs
        if ref.artifact_type == "STAGE2_PREDICTOR"
        for value in (_load_torch(root, ref),)
    }
    model_order = (
        "LAST_VALUE",
        "SEASONAL_NAIVE",
        "VAR",
        "DLINEAR",
        "PATCHTST",
        "ITRANSFORMER",
    )
    checkpoints = tuple(
        (value["model_id"], int(value["seed"]), ref.content_hash)
        for ref in task.inputs
        if ref.artifact_type == "VALIDATED_STAGE2_CHECKPOINT"
        for value in (_load_torch(root, ref),)
    )
    checkpoint_by_model: dict[str, list[str]] = {}
    for model_id, _, digest in checkpoints:
        checkpoint_by_model.setdefault(model_id, []).append(digest)
    predictor_sha = tuple(
        (
            model_id,
            baseline_refs[model_id].content_hash
            if model_id in baseline_refs
            else canonical_json_hash(sorted(checkpoint_by_model[model_id])),
        )
        for model_id in model_order
    )
    selections = tuple(
        _load_json(root, ref)
        for ref in task.inputs
        if ref.artifact_type == "STAGE2_MODEL_SELECTION"
    )
    strongest = next(item for item in selections if item["model_id"] in {"VAR", "DLINEAR"})
    primary = next(item for item in selections if item["model_id"] == "ITRANSFORMER")
    manifest = compile_stage2_manifest(
        Stage2CompilationInputs(
            scientific_config_sha256=config.scientific_hash(),
            stage1b_manifest_sha256=config.upstream.stage1b_manifest_sha256,
            e01_receipt_sha256=config.upstream.e01_receipt_sha256,
            source_receipt_sha256=canonical_json_hash(
                sorted(
                    ref.content_hash
                    for ref in task.inputs
                    if ref.artifact_type == "VERIFIED_STAGE2_SOURCE"
                )
            ),
            normalizer_sha256=_normalizer_sha256(bundle.normalizer),
            data_manifest_sha256=bundle.manifest_sha256,
            precision_receipt_sha256=sha256_file(
                _artifact_root(root) / "runtime/preflight_receipt.json"
            ),
            predictor_sha256=predictor_sha,
            neural_checkpoint_sha256=tuple(sorted(checkpoints)),
            strongest_linear=ModelSelection(**strongest),
            primary_itransformer=ModelSelection(**primary),
            runtime_failure_ids=(),
            formal_access_event_count=0,
            gpu_order=(0, 1),
        )
    )
    return _publish_json(root, task, manifest.payload())


def publish_receipt_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    from tarca.stage2.manifest import stage2_manifest_from_payload

    manifest = stage2_manifest_from_payload(_load_json(root, task.inputs[0]))
    receipt = freeze_stage2_suite(_artifact_root(root), manifest)
    return _publish_json(root, task, receipt.model_dump(mode="json"))


def stage2_executor_registry(repository_root: Path) -> ExecutorRegistry:
    root = repository_root.resolve()
    return ExecutorRegistry(
        {
            "stage2.verify_upstream": partial(verify_upstream_job, root),
            "stage2.verify_source": partial(verify_source_job, root),
            "stage2.generate_development_data": partial(generate_development_data_job, root),
            "stage2.fit_baseline": partial(fit_baseline_job, root),
            "stage2.train_neural": partial(train_neural_job, root),
            "stage2.validate_checkpoint": partial(validate_checkpoint_job, root),
            "stage2.predict_validation": partial(predict_validation_job, root),
            "stage2.select_model": partial(select_model_job, root),
            "stage2.freeze_candidate": partial(freeze_candidate_job, root),
            "stage2.publish_receipt": partial(publish_receipt_job, root),
        }
    )
