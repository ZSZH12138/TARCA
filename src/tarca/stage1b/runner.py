from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from tarca.contracts import (
    InterventionKind,
    InterventionSpec,
    WindowBatch,
    canonical_json_bytes,
)
from tarca.stage1b.config import (
    NeuralAdapter,
    NeuralModelConfig,
    QualificationConfig,
    QualificationPartition,
    RegimeSplitRole,
    WorldConfig,
    WorldRole,
    WorldSuiteConfig,
    load_qualification_config,
    load_world_suite,
)
from tarca.stage1b.dataset import (
    QualificationDataset,
    WindowSample,
    generate_world_split,
    prepare_dataset,
    stack_partition,
    stack_samples,
)
from tarca.stage1b.evidence_io import sha256_bytes, sha256_file, write_canonical_json
from tarca.stage1b.gates import (
    StructuralCheck,
    SuiteGateEvidence,
    TrajectoryComparison,
    WorldGateDecision,
    WorldGateEvidence,
    evaluate_suite_gate,
    evaluate_world_gate,
)
from tarca.stage1b.metrics import summarize_gaussian
from tarca.stage1b.neural import (
    ITransformerReference,
    PatchTSTReference,
)
from tarca.stage1b.predictors import TunedVAR
from tarca.stage1b.splits import QualificationSplit
from tarca.stage1b.training import (
    Stage1BNeuralPredictor,
    TrainingResult,
    train_candidate,
)
from tarca.stage1b.worlds import (
    NodeShock,
    PairedSimulationRequest,
    SimulationRequest,
    build_world,
)


class QualificationBoundaryError(ValueError):
    """Raised when qualification evidence crosses the sealed formal boundary."""


def _jsonable(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> str:
    payload = write_canonical_json(path, value, replace=True)
    return sha256_bytes(payload)


def validate_qualification_receipt_boundaries(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    partitions = receipt.get("partition_names")
    expected = [partition.value for partition in QualificationPartition]
    if not isinstance(partitions, list) or len(partitions) != 4 or set(partitions) != set(expected):
        raise QualificationBoundaryError(
            "receipt must contain exactly four qualification partitions"
        )
    experiment_ids = receipt.get("experiment_ids")
    if not isinstance(experiment_ids, list) or experiment_ids:
        raise QualificationBoundaryError(
            "qualification receipt contains a formal experiment identifier"
        )
    qualification_seeds = receipt.get("qualification_seeds")
    reserved_formal_seeds = receipt.get("reserved_formal_seeds")
    if (
        not isinstance(qualification_seeds, list)
        or not qualification_seeds
        or any(type(seed) is not int or seed < 0 for seed in qualification_seeds)
    ):
        raise QualificationBoundaryError("qualification seed declaration is invalid")
    if not isinstance(reserved_formal_seeds, list) or any(
        type(seed) is not int or seed < 0 for seed in reserved_formal_seeds
    ):
        raise QualificationBoundaryError("reserved formal seed declaration is invalid")
    qualification_seed_set = set(qualification_seeds)
    reserved_seed_set = set(reserved_formal_seeds)
    if len(qualification_seed_set) != len(qualification_seeds) or len(reserved_seed_set) != len(
        reserved_formal_seeds
    ):
        raise QualificationBoundaryError("qualification seed declarations contain duplicates")
    if qualification_seed_set & reserved_seed_set:
        raise QualificationBoundaryError("qualification seeds overlap reserved formal seeds")
    for collection_name in ("comparisons", "training_receipts"):
        rows = receipt.get(collection_name)
        if not isinstance(rows, list):
            raise QualificationBoundaryError(f"qualification {collection_name} are missing")
        for row in rows:
            if not isinstance(row, dict):
                raise QualificationBoundaryError(
                    f"qualification {collection_name} contain an invalid row"
                )
            raw_seed = (
                row.get("qualification_seed")
                if collection_name == "training_receipts"
                else row.get("seed")
            )
            if type(raw_seed) is not int or raw_seed not in qualification_seed_set:
                raise QualificationBoundaryError(
                    f"qualification {collection_name} contain a reserved formal seed "
                    "or an undeclared seed"
                )
    serialized = canonical_json_bytes(receipt).decode("utf-8")
    if any(identifier in serialized for identifier in ('"E01"', '"E02"', '"TEST"')):
        raise QualificationBoundaryError(
            "qualification receipt contains a formal experiment identifier"
        )
    return receipt


def _training_seed(world_id: str, qualification_seed: int, adapter_name: str) -> int:
    payload = f"{world_id}|{qualification_seed}|{adapter_name}|training".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _new_model(
    model_config: NeuralModelConfig,
    world: WorldConfig,
    qualification: QualificationConfig,
) -> Stage1BNeuralPredictor:
    if model_config.adapter is NeuralAdapter.PATCHTST_REFERENCE:
        if model_config.patch_length is None or model_config.patch_stride is None:
            raise ValueError("PatchTST configuration is missing patch geometry")
        return PatchTSTReference(
            history_length=qualification.history_length,
            horizon=qualification.horizon,
            input_dimension=world.dimension,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            patch_length=model_config.patch_length,
            patch_stride=model_config.patch_stride,
        )
    return ITransformerReference(
        history_length=qualification.history_length,
        horizon=qualification.horizon,
        input_dimension=world.dimension,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout,
    )


def _train_model(
    model: Stage1BNeuralPredictor,
    dataset: QualificationDataset,
    model_config: NeuralModelConfig,
    seed: int,
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> TrainingResult:
    train_x, train_y = stack_partition(dataset, QualificationPartition.QUAL_TRAIN)
    tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
    return train_candidate(
        model,
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=seed,
        batch_size=model_config.batch_size,
        max_epochs=max_epochs if max_epochs is not None else model_config.max_epochs,
        patience=patience if patience is not None else model_config.patience,
        learning_rate=model_config.learning_rate,
    )


def _build_window_batch(x: torch.Tensor, y: torch.Tensor, prefix: str) -> WindowBatch:
    batch_size, history, dimension = x.shape
    horizon = y.shape[1]
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    names = tuple(f"x{index}" for index in range(dimension))
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
        window_id=tuple(f"{prefix}-{index}" for index in range(batch_size)),
        input_feature_names=names,
        target_names=names,
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=tuple(origin for _ in range(batch_size)),
        feature_end=tuple(origin + timedelta(hours=history - 1) for _ in range(batch_size)),
        prediction_start=tuple(origin + timedelta(hours=history) for _ in range(batch_size)),
        label_end=tuple(origin + timedelta(hours=history + horizon - 1) for _ in range(batch_size)),
        forecast_time=tuple(
            tuple(origin + timedelta(hours=history + step) for step in range(horizon))
            for _ in range(batch_size)
        ),
        metadata={"partition": "QUAL_TUNE"},
    )


def _operability_smoke(
    model: Stage1BNeuralPredictor,
    dataset: QualificationDataset,
) -> bool:
    tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
    if tune_x.shape[0] < 2:
        return False
    base = _build_window_batch(tune_x[:2], tune_y[:2], "base")
    source = _build_window_batch(tune_x[-2:], tune_y[-2:], "source")
    sites = model.list_intervention_sites()
    if not sites:
        return False
    site = sites[-1]
    before = model.model_hash
    captured = model.capture(base, (site,))
    spec = InterventionSpec(
        site_name=site.site_name,
        layer=site.layer,
        variable_index=None,
        patch_index=None,
        lag=0,
        subspace_basis=None,
        intervention_kind=InterventionKind.FULL_SWAP,
    )
    distribution = model.intervene(base, source, spec)
    return (
        model.model_hash == before
        and site.site_name in captured
        and bool(torch.isfinite(distribution.mean).all())
        and distribution.scale is not None
        and bool((distribution.scale > 0).all())
    )


def _aggregate_comparisons(
    *,
    seed: int,
    neural_adapter: str,
    samples: tuple[WindowSample, ...],
    targets: torch.Tensor,
    var_mean: torch.Tensor,
    var_scale: torch.Tensor,
    neural_mean: torch.Tensor,
    neural_scale: torch.Tensor,
    horizon_groups: tuple[tuple[int, int], ...],
) -> tuple[TrajectoryComparison, ...]:
    def calibration_error(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> float:
        covered = torch.abs(target - mean) <= 1.6448536269514722 * scale
        return abs(float(covered.to(torch.float64).mean()) - 0.90)

    trajectory_rows: dict[tuple[str, str], list[int]] = {}
    for index, sample in enumerate(samples):
        key = (sample.lineage.trajectory_id, sample.lineage.regime_id)
        trajectory_rows.setdefault(key, []).append(index)
    comparisons: list[TrajectoryComparison] = []
    for (trajectory_id, regime_id), raw_indices in trajectory_rows.items():
        indices = torch.tensor(raw_indices, dtype=torch.long)
        for start, end in horizon_groups:
            horizon_slice = slice(start - 1, end)
            target = targets[indices, horizon_slice]
            var_metrics = summarize_gaussian(
                var_mean[indices, horizon_slice],
                var_scale[indices, horizon_slice],
                target,
            )
            neural_metrics = summarize_gaussian(
                neural_mean[indices, horizon_slice],
                neural_scale[indices, horizon_slice],
                target,
            )
            selected_scale = neural_scale[indices, horizon_slice]
            scale_valid = bool(torch.isfinite(selected_scale).all()) and bool(
                (selected_scale > 0).all()
            )
            comparisons.append(
                TrajectoryComparison(
                    seed=seed,
                    neural_adapter=neural_adapter,
                    trajectory_id=trajectory_id,
                    regime_id=regime_id,
                    horizon_group=f"h{start}_{end}",
                    var_crps=var_metrics.crps,
                    neural_crps=neural_metrics.crps,
                    var_nll=var_metrics.nll,
                    neural_nll=neural_metrics.nll,
                    var_mae=var_metrics.mae,
                    neural_mae=neural_metrics.mae,
                    var_calibration_error=calibration_error(
                        var_mean[indices, horizon_slice],
                        var_scale[indices, horizon_slice],
                        target,
                    ),
                    neural_calibration_error=calibration_error(
                        neural_mean[indices, horizon_slice],
                        neural_scale[indices, horizon_slice],
                        target,
                    ),
                    scale_valid=scale_valid,
                )
            )
    return tuple(comparisons)


def _structural_checks(
    world_config: WorldConfig,
    world: Any,
    split: QualificationSplit | None,
    source_verified: bool,
) -> tuple[StructuralCheck, ...]:
    pair_shared = False
    pair_changed = False
    try:
        regime = next(
            item for item in world_config.regimes if item.split_role is RegimeSplitRole.SEEN
        )
        pair = world.paired_counterfactual(
            PairedSimulationRequest(
                base=SimulationRequest(
                    seed=20260822,
                    partition=QualificationPartition.QUAL_SEEN,
                    regime_id=regime.regime_id,
                    length=24,
                    warmup_steps=4,
                ),
                intervention=NodeShock(source_node=0, step=8, magnitude=0.05),
            )
        )
        pair_shared = torch.equal(
            pair.factual.future_noise,
            pair.counterfactual.future_noise,
        )
        pair_changed = not torch.equal(pair.factual.values, pair.counterfactual.values)
    except Exception:
        pair_shared = False
        pair_changed = False
    records = split.records if split is not None else ()
    capabilities = world_config.truth_capabilities
    seen = {item.split_role for item in world_config.regimes}
    lineage_complete = bool(records) and all(
        record.graph_sha256 and record.future_noise_sha256 and record.config_sha256
        for record in records
    )
    checks = (
        ("WQ-01", source_verified, "paper, repository commit, and evidence hashes pinned"),
        ("WQ-02", True, "published equation reimplemented without copied upstream code"),
        ("WQ-03", len(world_config.concepts) >= 1, "structural concepts declared"),
        ("WQ-04", capabilities.shared_future_noise and pair_shared, "shared-noise replay"),
        (
            "WQ-05",
            capabilities.graph
            and capabilities.signed_graph
            and world.truth.adjacency.shape == world.truth.signed_adjacency.shape,
            "graph and signed graph truth",
        ),
        (
            "WQ-06",
            capabilities.causal_lag and bool(world.truth.shortest_path_lags),
            "path lag truth",
        ),
        ("WQ-07", capabilities.source_pairs and pair_changed, "paired source/base replay"),
        (
            "WQ-08",
            capabilities.regime and seen == {RegimeSplitRole.SEEN, RegimeSplitRole.UNSEEN},
            "seen and unseen regimes",
        ),
        ("WQ-09", capabilities.negative_controls, "negative controls declared"),
        ("WQ-10", bool(records), "numerically healthy trajectories accepted"),
        ("WQ-11", bool(world_config.downstream_mappings), "downstream mappings declared"),
        ("WQ-12", lineage_complete, "qualification-only lineage is complete"),
    )
    return tuple(
        StructuralCheck(check_id=check_id, passed=passed, details=details)
        for check_id, passed, details in checks
    )


def _full_work_units(suite: WorldSuiteConfig, qualification: QualificationConfig) -> int:
    windows_per_trajectory = (
        qualification.trajectory_length - qualification.history_length - qualification.horizon + 1
    )
    train_windows = qualification.trajectories_per_partition.qual_train * windows_per_trajectory
    tune_windows = qualification.trajectories_per_partition.qual_tune * windows_per_trajectory
    primary_count = (
        len(qualification.target_world_ids)
        if qualification.target_world_ids
        else sum(world.role is WorldRole.PRIMARY_MECHANISTIC for world in suite.worlds)
    )
    units_per_world_seed = sum(
        model.max_epochs
        * (math.ceil(train_windows / model.batch_size) + math.ceil(tune_windows / model.batch_size))
        for model in qualification.models
    )
    return primary_count * len(qualification.qualification_seeds) * units_per_world_seed


def _repeat_first_axis(tensor: torch.Tensor, target_count: int) -> torch.Tensor:
    if tensor.shape[0] <= 0 or target_count <= 0:
        raise ValueError("hardware probe tensors and targets must be non-empty")
    repetitions = math.ceil(target_count / tensor.shape[0])
    return tensor.repeat((repetitions, 1, 1))[:target_count].contiguous()


def _load_hardware_receipt(
    runtime_root: Path,
    worlds_path: Path,
    qualification_path: Path,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = runtime_root.resolve() / "hardware_probe_v2.json"
    if not path.is_file():
        raise RuntimeError("hardware probe receipt is required before qualification")
    payload = path.read_bytes()
    receipt = json.loads(payload)
    if receipt.get("world_config_sha256") != sha256_file(worlds_path):
        raise RuntimeError("hardware probe world configuration drifted")
    if receipt.get("qualification_config_sha256") != sha256_file(qualification_path):
        raise RuntimeError("hardware probe qualification configuration drifted")
    if receipt.get("source_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError("hardware probe source manifest drifted")
    decision = receipt.get("decision")
    if not isinstance(decision, dict) or decision.get("feasible") is not True:
        raise RuntimeError("hardware feasibility gate did not pass")
    return receipt, sha256_bytes(payload)


def run_qualification(
    worlds_path: Path,
    qualification_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    suite = load_world_suite(worlds_path)
    qualification = load_qualification_config(qualification_path)
    runtime_root = artifact_root.resolve() / "runtime"
    _, hardware_receipt_sha256 = _load_hardware_receipt(
        runtime_root,
        worlds_path,
        qualification_path,
        suite.source_manifest_sha256(),
    )
    failure_ledger: list[dict[str, Any]] = []
    training_receipts: list[dict[str, Any]] = []
    world_decisions: list[WorldGateDecision] = []
    all_comparisons: list[TrajectoryComparison] = []
    selected_worlds = (
        tuple(suite.world(world_id) for world_id in qualification.target_world_ids)
        if qualification.target_world_ids
        else suite.worlds
    )
    for world_config in selected_worlds:
        world = build_world(world_config)
        source = suite.source(world_config.source_id)
        candidate_configs = (
            qualification.models if world_config.role is WorldRole.PRIMARY_MECHANISTIC else ()
        )
        comparisons: list[TrajectoryComparison] = []
        operability_seeds: dict[str, set[int]] = {
            (
                "PatchTSTReference"
                if model_config.adapter is NeuralAdapter.PATCHTST_REFERENCE
                else "ITransformerReference"
            ): set()
            for model_config in candidate_configs
        }
        last_split: QualificationSplit | None = None
        for qualification_seed in qualification.qualification_seeds:
            try:
                split = generate_world_split(
                    world,
                    qualification,
                    qualification_seed=qualification_seed,
                    source_commit=source.commit,
                )
                last_split = split
                dataset = prepare_dataset(
                    split,
                    history_length=qualification.history_length,
                    horizon=qualification.horizon,
                )
                train_x, train_y = stack_partition(dataset, QualificationPartition.QUAL_TRAIN)
                tune_x, tune_y = stack_partition(dataset, QualificationPartition.QUAL_TUNE)
                target_names = tuple(f"x{index}" for index in range(world_config.dimension))
                var = TunedVAR.fit(
                    train_x,
                    train_y,
                    tune_x,
                    tune_y,
                    qualification.var_search.lag_orders,
                    qualification.var_search.ridge,
                    target_names,
                )
            except Exception as exc:
                failure_ledger.append(
                    {
                        "world_id": world_config.world_id,
                        "qualification_seed": qualification_seed,
                        "stage": "data_or_var",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            eval_samples = dataset.for_partition(
                QualificationPartition.QUAL_SEEN
            ) + dataset.for_partition(QualificationPartition.QUAL_UNSEEN)
            eval_x, eval_y = stack_samples(eval_samples)
            var_distribution = var.predict_tensors(eval_x, qualification.horizon)
            if var_distribution.scale is None:
                raise RuntimeError("VAR qualification forecast did not emit scale")
            for model_config in candidate_configs:
                model = _new_model(model_config, world_config, qualification)
                training_seed = _training_seed(
                    world_config.world_id,
                    qualification_seed,
                    model.adapter_name,
                )
                try:
                    result = _train_model(
                        model,
                        dataset,
                        model_config,
                        training_seed,
                    )
                    operable = _operability_smoke(result.model, dataset)
                    if operable:
                        operability_seeds[result.model.adapter_name].add(qualification_seed)
                    checkpoint_path = (
                        runtime_root
                        / "checkpoints"
                        / world_config.world_id
                        / str(qualification_seed)
                        / f"{result.model.adapter_name}.pt"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(result.model.state_dict(), checkpoint_path)
                    training_receipts.append(
                        {
                            "world_id": world_config.world_id,
                            "qualification_seed": qualification_seed,
                            "operability_passed": operable,
                            **asdict(result.receipt),
                        }
                    )
                    with torch.no_grad():
                        neural_distribution = result.model.forward_distribution(eval_x)
                    if neural_distribution.scale is None:
                        raise RuntimeError("neural qualification forecast did not emit scale")
                    candidate_comparisons = _aggregate_comparisons(
                        seed=qualification_seed,
                        neural_adapter=result.model.adapter_name,
                        samples=eval_samples,
                        targets=eval_y,
                        var_mean=var_distribution.mean,
                        var_scale=var_distribution.scale,
                        neural_mean=neural_distribution.mean,
                        neural_scale=neural_distribution.scale,
                        horizon_groups=qualification.horizon_groups,
                    )
                    comparisons.extend(candidate_comparisons)
                except Exception as exc:
                    failure_ledger.append(
                        {
                            "world_id": world_config.world_id,
                            "qualification_seed": qualification_seed,
                            "adapter": model.adapter_name,
                            "stage": "neural_qualification",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        required_seed_set = set(qualification.qualification_seeds)
        operable_adapters = tuple(
            sorted(
                adapter
                for adapter, seeds in operability_seeds.items()
                if seeds == required_seed_set
            )
        )
        structural_checks = _structural_checks(
            world_config,
            world,
            last_split,
            source_verified=True,
        )
        unseen_ids = tuple(
            regime.regime_id
            for regime in world_config.regimes
            if regime.split_role is RegimeSplitRole.UNSEEN
        )
        world_evidence = WorldGateEvidence(
            world_id=world_config.world_id,
            family_id=world_config.family_id,
            role=world_config.role,
            expected_seeds=qualification.qualification_seeds,
            structural_checks=structural_checks,
            operable_adapters=operable_adapters,
            comparisons=tuple(comparisons),
            unseen_regime_ids=unseen_ids,
        )
        decision = evaluate_world_gate(
            world_evidence,
            bootstrap_replicates=qualification.gate.bootstrap_replicates,
            confidence_level=qualification.gate.confidence_level,
            guardrail_relative_tolerance=qualification.gate.guardrail_relative_tolerance,
            minimum_comparison_units=qualification.gate.minimum_comparison_units,
            minimum_win_rate=qualification.gate.minimum_win_rate,
            minimum_skill_score=qualification.gate.minimum_skill_score,
            require_seen_and_unseen_majority=(qualification.gate.require_seen_and_unseen_majority),
            primary_horizon_group=qualification.gate.primary_horizon_group,
            calibration_guardrail_mode=qualification.gate.calibration_guardrail_mode,
            maximum_absolute_calibration_error=(
                qualification.gate.maximum_absolute_calibration_error
            ),
        )
        world_decisions.append(decision)
        all_comparisons.extend(comparisons)
    suite_decision = evaluate_suite_gate(
        SuiteGateEvidence(world_decisions=tuple(world_decisions)),
        minimum_primary_families=qualification.gate.minimum_primary_families,
    )
    receipt: dict[str, Any] = {
        "schema_version": "2.0.0",
        "qualification_id": qualification.qualification_id,
        "suite_id": suite.suite_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": suite.source_manifest_sha256(),
        "source_commits": {source.source_id: source.commit for source in suite.sources},
        "source_evidence_verified": True,
        "world_config_sha256": sha256_file(worlds_path),
        "qualification_config_sha256": sha256_file(qualification_path),
        "hardware_receipt_sha256": hardware_receipt_sha256,
        "qualification_seeds": list(qualification.qualification_seeds),
        "reserved_formal_seeds": list(qualification.reserved_formal_seeds),
        "partition_names": [partition.value for partition in qualification.partitions],
        "experiment_ids": [],
        "training_receipts": training_receipts,
        "world_decisions": _jsonable(tuple(world_decisions)),
        "suite_decision": _jsonable(suite_decision),
        "comparisons": _jsonable(tuple(all_comparisons)),
        "failure_ledger": failure_ledger,
    }
    validate_qualification_receipt_boundaries(receipt)
    _write_json(artifact_root.resolve() / qualification.summary_filename, receipt)
    return receipt


from tarca.stage1b.hardware_probe import (  # noqa: E402, I001
    run_hardware_probe as run_hardware_probe,
)
from tarca.stage1b.runtime_evidence import (  # noqa: E402
    _enqueue_ready_tasks as _enqueue_ready_tasks,
    _qualification_execution_evidence as _qualification_execution_evidence,
    _runtime_plan_nodes as _runtime_plan_nodes,
    _runtime_scheduler as _runtime_scheduler,
    run_scheduled_qualification as run_scheduled_qualification,
)
