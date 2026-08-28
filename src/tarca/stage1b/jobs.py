from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from tarca.artifacts.store import LocalArtifactStore
from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash
from tarca.execution.contracts import ExecutionContext, TaskSpec
from tarca.execution.registry import ExecutorRegistry, ProgressSink
from tarca.stage1b.compiler import repository_v2_inputs
from tarca.stage1b.config import (
    QualificationPartition,
    RegimeSplitRole,
    TrajectoryPartitionCounts,
    WorldRole,
)
from tarca.stage1b.dataset import (
    NormalizationStatistics,
    QualificationDataset,
    WindowLineage,
    WindowSample,
    generate_world_split,
    prepare_dataset,
    stack_partition,
    stack_samples,
)
from tarca.stage1b.gates import (
    StructuralCheck,
    SuiteGateEvidence,
    TrajectoryComparison,
    WorldGateEvidence,
    evaluate_suite_gate,
    evaluate_world_gate,
)
from tarca.stage1b.predictors import TunedVAR
from tarca.stage1b.reproduction import ReproductionReceipt, run_reproduction
from tarca.stage1b.runner import (
    _aggregate_comparisons,
    _file_sha256,
    _jsonable,
    _load_hardware_receipt,
    _new_model,
    _operability_smoke,
    _structural_checks,
    _training_seed,
    _write_json,
    validate_qualification_receipt_boundaries,
)
from tarca.stage1b.sources import (
    MaterializedSources,
    SourceMaterializationReceipt,
    SubprocessGitRunner,
    materialize_source,
    source_acquisition_mode_from_environment,
    source_cache_root_from_environment,
    verify_materialized_source,
)
from tarca.stage1b.training import TrainingPolicy, train_candidate
from tarca.stage1b.worlds import build_world

_STORE_ROOT = "artifacts/stage1b/runtime/store"
_SCHEMA_VERSION = "2.0.0"


def _source_cache_root(repo_root: Path) -> Path:
    return source_cache_root_from_environment(repo_root)


def stage1b_artifact_store(
    repo_root: Path,
    task: TaskSpec | None = None,
) -> LocalArtifactStore:
    identity_hash = "0" * 64 if task is None else canonical_json_hash(task.identity)
    return LocalArtifactStore(
        repo_root,
        producer_stage="stage1b",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash=identity_hash,
        dependencies=() if task is None else task.inputs,
        store_relative_root=_STORE_ROOT,
    )


def _publish_json(repo_root: Path, task: TaskSpec, value: object) -> ArtifactRef:
    return stage1b_artifact_store(repo_root, task).publish_bytes(
        canonical_json_bytes(value) + b"\n",
        task.output_artifact_type,
        "application/json",
        _SCHEMA_VERSION,
    )


def _load_json(repo_root: Path, ref: ArtifactRef) -> dict[str, Any]:
    payload = stage1b_artifact_store(repo_root).load_bytes(ref)
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Stage1B JSON artifact must contain an object")
    return cast(dict[str, Any], decoded)


def _torch_bytes(value: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_torch(repo_root: Path, ref: ArtifactRef) -> dict[str, Any]:
    payload = stage1b_artifact_store(repo_root).load_bytes(ref)
    decoded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(decoded, dict):
        raise ValueError("Stage1B tensor artifact must contain a mapping")
    return cast(dict[str, Any], decoded)


def _publish_torch(repo_root: Path, task: TaskSpec, value: dict[str, Any]) -> ArtifactRef:
    return stage1b_artifact_store(repo_root, task).publish_bytes(
        _torch_bytes(value),
        task.output_artifact_type,
        "application/x-pytorch-state-dict",
        _SCHEMA_VERSION,
    )


def _source_receipt_payload(
    repo_root: Path,
    receipt: SourceMaterializationReceipt,
) -> dict[str, Any]:
    checkout = receipt.checkout_root.resolve()
    relative = checkout.relative_to(repo_root.resolve()).as_posix()
    return {
        "source_id": receipt.source_id,
        "repository_url": receipt.repository_url,
        "commit": receipt.commit,
        "checkout_relative_path": relative,
        "tree_sha256": receipt.tree_sha256,
        "asset_sha256": [list(item) for item in receipt.asset_sha256],
        "authorization_id": receipt.authorization_id,
        "materialized_at_utc": receipt.materialized_at_utc.isoformat(),
    }


def _source_receipt(repo_root: Path, payload: dict[str, Any]) -> SourceMaterializationReceipt:
    relative = Path(str(payload["checkout_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source receipt checkout path is unsafe")
    checkout = (repo_root.resolve() / relative).resolve()
    if repo_root.resolve() not in checkout.parents:
        raise ValueError("source receipt checkout escapes the repository")
    raw_assets = payload["asset_sha256"]
    if not isinstance(raw_assets, list):
        raise ValueError("source receipt asset hashes are invalid")
    return SourceMaterializationReceipt(
        source_id=str(payload["source_id"]),
        repository_url=str(payload["repository_url"]),
        commit=str(payload["commit"]),
        checkout_root=checkout,
        tree_sha256=str(payload["tree_sha256"]),
        asset_sha256=tuple((str(item[0]), str(item[1])) for item in raw_assets),
        authorization_id=str(payload["authorization_id"]),
        materialized_at_utc=datetime.fromisoformat(str(payload["materialized_at_utc"])),
    )


def materialize_source_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    source = inputs.world_suite.source(task.identity.data_id)
    receipt = materialize_source(
        source,
        _source_cache_root(repo_root),
        SubprocessGitRunner.discover(),
        mode=source_acquisition_mode_from_environment(),
    )
    return _publish_json(repo_root, task, _source_receipt_payload(repo_root, receipt))


def reproduce_official_case_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    case = next(
        item for item in inputs.reproduction_suite.cases if item.source_id == task.identity.data_id
    )
    receipt = _source_receipt(repo_root, _load_json(repo_root, task.inputs[0]))
    verify_materialized_source(
        receipt,
        _source_cache_root(repo_root),
    )
    result = run_reproduction(
        case,
        MaterializedSources((receipt,)),
        input_root=repo_root,
    )
    if not result.passed:
        raise RuntimeError("official reproduction case did not pass")
    return _publish_json(repo_root, task, result.model_dump(mode="json"))


def check_world_health_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    world_config = inputs.world_suite.world(task.identity.data_id)
    source_receipts = tuple(
        _source_receipt(repo_root, _load_json(repo_root, ref))
        for ref in task.inputs
        if ref.artifact_type == "OFFICIAL_SOURCE_RECEIPT"
    )
    for receipt in source_receipts:
        verify_materialized_source(
            receipt,
            _source_cache_root(repo_root),
        )
    reproductions = tuple(
        ReproductionReceipt.model_validate(_load_json(repo_root, ref))
        for ref in task.inputs
        if ref.artifact_type == "OFFICIAL_REPRODUCTION_RECEIPT"
    )
    if any(not receipt.passed for receipt in reproductions):
        raise RuntimeError("world source reproduction is not verified")
    qualification = inputs.qualification.model_copy(
        update={
            "trajectory_length": max(
                96,
                inputs.qualification.history_length + inputs.qualification.horizon + 8,
            ),
            "trajectories_per_partition": TrajectoryPartitionCounts(
                QUAL_TRAIN=1,
                QUAL_TUNE=1,
                QUAL_SEEN=1,
                QUAL_UNSEEN=1,
            ),
        }
    )
    world = build_world(world_config)
    split = generate_world_split(
        world,
        qualification,
        qualification_seed=inputs.qualification.qualification_seeds[0],
        source_commit=inputs.world_suite.source(world_config.source_id).commit,
    )
    checks = _structural_checks(world_config, world, split, source_verified=True)
    if not all(check.passed for check in checks):
        failed = ", ".join(check.check_id for check in checks if not check.passed)
        raise RuntimeError(f"world structural health checks failed: {failed}")
    return _publish_json(
        repo_root,
        task,
        {"world_id": world_config.world_id, "checks": [asdict(check) for check in checks]},
    )


def _dataset_payload(dataset: QualificationDataset) -> dict[str, Any]:
    partitions: dict[str, Any] = {}
    for partition in QualificationPartition:
        samples = dataset.for_partition(partition)
        histories, targets = stack_samples(samples)
        partitions[partition.value] = {
            "histories": histories.detach().cpu(),
            "targets": targets.detach().cpu(),
            "lineages": [
                {**asdict(sample.lineage), "partition": sample.lineage.partition.value}
                for sample in samples
            ],
        }
    return {
        "schema_version": _SCHEMA_VERSION,
        "history_length": dataset.history_length,
        "horizon": dataset.horizon,
        "normalization_mean": dataset.statistics.mean.detach().cpu(),
        "normalization_standard_deviation": dataset.statistics.standard_deviation.detach().cpu(),
        "fitted_partition": dataset.statistics.fitted_partition.value,
        "partitions": partitions,
    }


def _dataset_from_payload(payload: dict[str, Any]) -> QualificationDataset:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("qualification dataset schema is unsupported")
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, dict) or set(raw_partitions) != {
        item.value for item in QualificationPartition
    }:
        raise ValueError("qualification dataset must contain exactly four partitions")
    entries: list[tuple[QualificationPartition, tuple[WindowSample, ...]]] = []
    seen_ids: set[str] = set()
    for partition in QualificationPartition:
        raw = raw_partitions[partition.value]
        histories = raw["histories"]
        targets = raw["targets"]
        lineages = raw["lineages"]
        if not isinstance(histories, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError("qualification dataset tensors are missing")
        if histories.shape[0] != targets.shape[0] or histories.shape[0] != len(lineages):
            raise ValueError("qualification dataset rows are misaligned")
        samples: list[WindowSample] = []
        for index, raw_lineage in enumerate(lineages):
            lineage = WindowLineage(
                **{**raw_lineage, "partition": QualificationPartition(raw_lineage["partition"])}
            )
            if lineage.partition is not partition or lineage.window_id in seen_ids:
                raise ValueError("qualification dataset lineage is duplicated or crossed")
            seen_ids.add(lineage.window_id)
            samples.append(
                WindowSample(
                    history=histories[index].clone(),
                    target=targets[index].clone(),
                    lineage=lineage,
                )
            )
        entries.append((partition, tuple(samples)))
    mean = payload["normalization_mean"]
    standard_deviation = payload["normalization_standard_deviation"]
    if not isinstance(mean, torch.Tensor) or not isinstance(standard_deviation, torch.Tensor):
        raise ValueError("qualification normalization tensors are missing")
    return QualificationDataset(
        statistics=NormalizationStatistics(
            mean=mean.clone(),
            standard_deviation=standard_deviation.clone(),
            fitted_partition=QualificationPartition(str(payload["fitted_partition"])),
        ),
        samples=tuple(entries),
        history_length=int(payload["history_length"]),
        horizon=int(payload["horizon"]),
    )


def generate_dataset_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    world_config = inputs.world_suite.world(task.identity.data_id)
    world = build_world(world_config)
    split = generate_world_split(
        world,
        inputs.qualification,
        qualification_seed=task.identity.seed,
        source_commit=inputs.world_suite.source(world_config.source_id).commit,
    )
    dataset = prepare_dataset(
        split,
        history_length=inputs.qualification.history_length,
        horizon=inputs.qualification.horizon,
    )
    return _publish_torch(repo_root, task, _dataset_payload(dataset))


def validate_dataset_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    payload = _load_torch(repo_root, task.inputs[0])
    dataset = _dataset_from_payload(payload)
    if dataset.statistics.fitted_partition is not QualificationPartition.QUAL_TRAIN:
        raise ValueError("normalization was not fitted on QUAL_TRAIN")
    for partition in QualificationPartition:
        histories, targets = stack_partition(dataset, partition)
        if not bool(torch.isfinite(histories).all() and torch.isfinite(targets).all()):
            raise ValueError("qualification dataset contains non-finite values")
    return _publish_torch(repo_root, task, payload)


def score_var_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    dataset = _dataset_from_payload(_load_torch(repo_root, task.inputs[0]))
    train_x, train_y = stack_partition(dataset, QualificationPartition.QUAL_TRAIN)
    tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
    dimension = train_x.shape[2]
    model = TunedVAR.fit(
        train_x,
        train_y,
        tune_x,
        tune_y,
        inputs.qualification.var_search.lag_orders,
        inputs.qualification.var_search.ridge,
        tuple(f"x{index}" for index in range(dimension)),
    )
    return _publish_torch(
        repo_root,
        task,
        {
            "coefficients": model.coefficients.detach().cpu(),
            "intercept": model.intercept.detach().cpu(),
            "residual_scale": model.residual_scale.detach().cpu(),
            "selected_lag": model.selected_lag,
            "selected_ridge": model.selected_ridge,
            "target_names": list(model.target_names),
            "model_hash": model.model_hash,
        },
    )


def _model_and_config(repo_root: Path, task: TaskSpec) -> tuple[Any, Any, Any]:
    inputs = repository_v2_inputs(repo_root)
    world = inputs.world_suite.world(task.identity.data_id)
    model_config = next(
        model for model in inputs.qualification.models if model.model_id == task.identity.model_id
    )
    return inputs, world, model_config


def train_neural_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context
    inputs, world_config, model_config = _model_and_config(repo_root, task)
    reproduction = ReproductionReceipt.model_validate(_load_json(repo_root, task.inputs[1]))
    if not reproduction.passed:
        raise RuntimeError("neural official reproduction prerequisite failed")
    dataset = _dataset_from_payload(_load_torch(repo_root, task.inputs[0]))
    train_x, train_y = stack_partition(dataset, QualificationPartition.QUAL_TRAIN)
    tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
    device = os.environ.get("TARCA_STAGE1B_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    precision = os.environ.get("TARCA_STAGE1B_PRECISION", "FP32")
    workers = int(
        os.environ.get(
            "TARCA_STAGE1B_DATALOADER_WORKERS",
            "3" if device == "cuda" else "0",
        )
    )
    policy = TrainingPolicy(
        device=device,
        precision=cast(Any, precision),
        batch_size=model_config.batch_size,
        max_epochs=model_config.max_epochs,
        patience=model_config.patience,
        learning_rate=model_config.learning_rate,
        dataloader_workers=workers,
        checkpoint_root=repo_root / "artifacts/stage1b/runtime/checkpoints",
    )
    resume_flag = os.environ.get("TARCA_STAGE1B_RESUME_CHECKPOINTS", "0")
    if resume_flag not in {"0", "1"}:
        raise ValueError("TARCA_STAGE1B_RESUME_CHECKPOINTS must be 0 or 1")
    model = _new_model(model_config, world_config, inputs.qualification)
    result = train_candidate(
        model,
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=_training_seed(world_config.world_id, task.identity.seed, model.adapter_name),
        policy=policy,
        progress=progress,
        resume_if_available=resume_flag == "1",
    )
    if not result.receipt.completed:
        raise RuntimeError("neural training stopped before a complete checkpoint")
    return _publish_torch(
        repo_root,
        task,
        {
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in result.model.state_dict().items()
            },
            "training_receipt": asdict(result.receipt),
        },
    )


def freeze_check_model_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs, world_config, model_config = _model_and_config(repo_root, task)
    trained = _load_torch(repo_root, task.inputs[0])
    dataset = _dataset_from_payload(_load_torch(repo_root, task.inputs[1]))
    model = _new_model(model_config, world_config, inputs.qualification)
    model.load_state_dict(trained["model_state"])
    model.freeze()
    operable = _operability_smoke(model, dataset)
    if not operable:
        raise RuntimeError("frozen neural model failed mechanistic operability")
    receipt = dict(trained["training_receipt"])
    if receipt.get("model_hash") != model.model_hash:
        raise ValueError("frozen model hash drifted from training receipt")
    return _publish_torch(
        repo_root,
        task,
        {"model_state": trained["model_state"], "training_receipt": receipt, "operable": True},
    )


def score_bootstrap_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs, world_config, model_config = _model_and_config(repo_root, task)
    dataset = _dataset_from_payload(_load_torch(repo_root, task.inputs[0]))
    var_payload = _load_torch(repo_root, task.inputs[1])
    frozen = _load_torch(repo_root, task.inputs[2])
    var = TunedVAR(
        coefficients=var_payload["coefficients"],
        intercept=var_payload["intercept"],
        residual_scale=var_payload["residual_scale"],
        selected_lag=int(var_payload["selected_lag"]),
        selected_ridge=float(var_payload["selected_ridge"]),
        target_names=tuple(str(item) for item in var_payload["target_names"]),
    )
    if var.model_hash != var_payload["model_hash"]:
        raise ValueError("VAR artifact model hash drifted")
    model = _new_model(model_config, world_config, inputs.qualification)
    model.load_state_dict(frozen["model_state"])
    model.freeze()
    if frozen.get("operable") is not True:
        raise ValueError("neural model is not marked operable")
    samples = dataset.for_partition(QualificationPartition.QUAL_SEEN) + dataset.for_partition(
        QualificationPartition.QUAL_UNSEEN
    )
    histories, targets = stack_samples(samples)
    var_distribution = var.predict_tensors(histories, inputs.qualification.horizon)
    with torch.no_grad():
        neural_distribution = model.forward_distribution(histories)
    if var_distribution.scale is None or neural_distribution.scale is None:
        raise RuntimeError("qualification scoring requires probabilistic scales")
    comparisons = _aggregate_comparisons(
        seed=task.identity.seed,
        neural_adapter=model.adapter_name,
        samples=samples,
        targets=targets,
        var_mean=var_distribution.mean,
        var_scale=var_distribution.scale,
        neural_mean=neural_distribution.mean,
        neural_scale=neural_distribution.scale,
        horizon_groups=inputs.qualification.horizon_groups,
    )
    return _publish_json(
        repo_root,
        task,
        {
            "world_id": world_config.world_id,
            "model_id": model_config.model_id,
            "seed": task.identity.seed,
            "adapter_name": model.adapter_name,
            "operable": True,
            "training_receipt": frozen["training_receipt"],
            "comparisons": [asdict(item) for item in comparisons],
        },
    )


def aggregate_qualification_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    health = {
        payload["world_id"]: payload
        for ref in task.inputs
        if ref.artifact_type == "WORLD_HEALTH_RECEIPT"
        for payload in (_load_json(repo_root, ref),)
    }
    score_payloads = tuple(
        _load_json(repo_root, ref)
        for ref in task.inputs
        if ref.artifact_type == "QUALIFICATION_COMPARISON"
    )
    expected_scores = (
        sum(world.role is WorldRole.PRIMARY_MECHANISTIC for world in inputs.world_suite.worlds)
        * len(inputs.qualification.qualification_seeds)
        * len(inputs.qualification.models)
    )
    if len(score_payloads) != expected_scores:
        raise RuntimeError("qualification aggregate is missing required comparison artifacts")
    decisions = []
    all_comparisons: list[TrajectoryComparison] = []
    training_receipts: list[dict[str, Any]] = []
    for world in inputs.world_suite.worlds:
        if world.role is WorldRole.REFERENCE_ONLY:
            continue
        health_payload = health.get(world.world_id)
        if health_payload is None:
            raise RuntimeError(f"world health evidence is missing: {world.world_id}")
        checks = tuple(StructuralCheck(**item) for item in health_payload["checks"])
        world_scores = tuple(item for item in score_payloads if item["world_id"] == world.world_id)
        comparisons = tuple(
            TrajectoryComparison(**row) for score in world_scores for row in score["comparisons"]
        )
        all_comparisons.extend(comparisons)
        training_receipts.extend(
            {
                "world_id": world.world_id,
                "qualification_seed": score["seed"],
                "operability_passed": score["operable"],
                **score["training_receipt"],
            }
            for score in world_scores
        )
        adapters = tuple(
            sorted(
                {str(score["adapter_name"]) for score in world_scores if score["operable"] is True}
            )
        )
        evidence = WorldGateEvidence(
            world_id=world.world_id,
            family_id=world.family_id,
            role=world.role,
            expected_seeds=inputs.qualification.qualification_seeds,
            structural_checks=checks,
            operable_adapters=adapters,
            comparisons=comparisons,
            unseen_regime_ids=tuple(
                regime.regime_id
                for regime in world.regimes
                if regime.split_role is RegimeSplitRole.UNSEEN
            ),
        )
        gate = inputs.qualification.gate
        decisions.append(
            evaluate_world_gate(
                evidence,
                bootstrap_replicates=gate.bootstrap_replicates,
                confidence_level=gate.confidence_level,
                guardrail_relative_tolerance=gate.guardrail_relative_tolerance,
                minimum_comparison_units=gate.minimum_comparison_units,
                minimum_win_rate=gate.minimum_win_rate,
                minimum_skill_score=gate.minimum_skill_score,
                require_seen_and_unseen_majority=gate.require_seen_and_unseen_majority,
            )
        )
    suite_decision = evaluate_suite_gate(
        SuiteGateEvidence(world_decisions=tuple(decisions)),
        minimum_primary_families=inputs.qualification.gate.minimum_primary_families,
    )
    return _publish_json(
        repo_root,
        task,
        {
            "training_receipts": training_receipts,
            "world_decisions": _jsonable(tuple(decisions)),
            "suite_decision": _jsonable(suite_decision),
            "comparisons": _jsonable(tuple(all_comparisons)),
            "failure_ledger": [],
        },
    )


def publish_qualification_receipt_job(
    repo_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context, progress
    inputs = repository_v2_inputs(repo_root)
    aggregate = _load_json(repo_root, task.inputs[0])
    reproductions = tuple(
        ReproductionReceipt.model_validate(_load_json(repo_root, ref)) for ref in task.inputs[1:]
    )
    if not reproductions or any(not item.passed for item in reproductions):
        raise RuntimeError("official reproduction suite is incomplete")
    runtime_root = repo_root / "artifacts/stage1b/runtime"
    _, hardware_hash = _load_hardware_receipt(
        runtime_root,
        repo_root / "configs/stage1b/worlds_v2.yaml",
        repo_root / "configs/stage1b/qualification_v2.yaml",
        inputs.world_suite.source_manifest_sha256(),
    )
    execution_evidence_path = runtime_root / "qualification_execution_evidence_v2.json"
    if not execution_evidence_path.is_file():
        raise RuntimeError("qualification execution evidence is missing")
    execution_evidence_document = json.loads(execution_evidence_path.read_text(encoding="utf-8"))
    if not isinstance(execution_evidence_document, dict) or not isinstance(
        execution_evidence_document.get("qualification_evidence"), dict
    ):
        raise RuntimeError("qualification execution evidence is invalid")
    qualification_evidence = cast(
        dict[str, Any], execution_evidence_document["qualification_evidence"]
    )
    if qualification_evidence.get("hardware_receipt_sha256") != hardware_hash:
        raise RuntimeError("qualification execution evidence hardware identity drifted")
    receipt = {
        "schema_version": _SCHEMA_VERSION,
        "qualification_id": inputs.qualification.qualification_id,
        "suite_id": inputs.world_suite.suite_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": inputs.world_suite.source_manifest_sha256(),
        "source_commits": {
            source.source_id: source.commit for source in inputs.world_suite.sources
        },
        "source_evidence_verified": True,
        "world_config_sha256": _file_sha256(repo_root / "configs/stage1b/worlds_v2.yaml"),
        "qualification_config_sha256": _file_sha256(
            repo_root / "configs/stage1b/qualification_v2.yaml"
        ),
        "hardware_receipt_sha256": hardware_hash,
        "qualification_evidence": qualification_evidence,
        "qualification_seeds": list(inputs.qualification.qualification_seeds),
        "reserved_formal_seeds": list(inputs.qualification.reserved_formal_seeds),
        "partition_names": [partition.value for partition in inputs.qualification.partitions],
        "experiment_ids": [],
        **aggregate,
    }
    validate_qualification_receipt_boundaries(receipt)
    artifact = _publish_json(repo_root, task, receipt)
    _write_json(repo_root / "artifacts/stage1b/qualification_v2_summary.json", receipt)
    return artifact


def _bound(repo_root: Path, function: Any) -> Any:
    return partial(function, repo_root.resolve())


def stage1b_executor_registry(repo_root: Path) -> ExecutorRegistry:
    return ExecutorRegistry(
        {
            "stage1b.materialize_source": _bound(repo_root, materialize_source_job),
            "stage1b.reproduce_official_case": _bound(repo_root, reproduce_official_case_job),
            "stage1b.check_world_health": _bound(repo_root, check_world_health_job),
            "stage1b.generate_dataset": _bound(repo_root, generate_dataset_job),
            "stage1b.validate_dataset": _bound(repo_root, validate_dataset_job),
            "stage1b.score_var": _bound(repo_root, score_var_job),
            "stage1b.train_neural": _bound(repo_root, train_neural_job),
            "stage1b.freeze_check_model": _bound(repo_root, freeze_check_model_job),
            "stage1b.score_bootstrap": _bound(repo_root, score_bootstrap_job),
            "stage1b.aggregate_qualification": _bound(repo_root, aggregate_qualification_job),
            "stage1b.publish_qualification_receipt": _bound(
                repo_root, publish_qualification_receipt_job
            ),
        }
    )
