from __future__ import annotations

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
    ForecastDistribution,
    SealedAccessGrant,
    WindowBatch,
    canonical_json_bytes,
    canonical_json_hash,
    sha256_file,
    validate_forecast_distribution,
)
from tarca.e02.bootstrap import paired_stratified_bootstrap
from tarca.e02.config import E02Config, load_e02_config
from tarca.e02.decision import E02Evidence, evaluate_e02
from tarca.e02.evidence_io import (
    decision as _decision,
)
from tarca.e02.evidence_io import (
    evidence as _evidence,
)
from tarca.e02.evidence_io import (
    score_set as _score_set,
)
from tarca.e02.evidence_io import (
    trajectory_score as _trajectory_score,
)
from tarca.e02.evidence_io import (
    write_final_e02 as _write_final_e02,
)
from tarca.e02.receipt import build_e02_receipt
from tarca.e02.scoring import (
    TrajectoryLineage,
    TrajectoryScore,
    score_trajectory,
    summarize_scores,
)
from tarca.execution import (
    ExecutionContext,
    ExecutionStateStore,
    ExecutorRegistry,
    ProgressSink,
    TaskSpec,
)
from tarca.stage1b.config import load_world_suite
from tarca.stage1b.worlds import build_world
from tarca.stage2.config import Stage2Config, load_stage2_config
from tarca.stage2.data import Stage2DataBundle, open_formal_bundle
from tarca.stage2.freeze import verify_frozen_stage2_suite
from tarca.stage2.jobs import (
    _artifact_root as _stage2_artifact_root,
)
from tarca.stage2.jobs import (
    _baseline_from_payload,
    _bundle_from_payload,
    _bundle_payload,
    _new_neural,
    _normalizer_sha256,
    stage2_artifact_store,
)
from tarca.stage2.jobs import (
    _load_torch as _load_stage2_torch,
)
from tarca.stage2.manifest import Stage2Manifest, stage2_manifest_from_payload

_SCHEMA = "tarca-e02-job-v1"


def _root(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_E02_ARTIFACT_ROOT", "artifacts/e02").strip()
    root = (repository_root.resolve() / raw).resolve()
    if repository_root.resolve() not in root.parents:
        raise ValueError("E02 artifact root must stay inside the repository")
    return root


def e02_artifact_store(repository_root: Path, task: TaskSpec | None = None) -> LocalArtifactStore:
    relative = (_root(repository_root) / "runtime/store").relative_to(
        repository_root.resolve()
    )
    return LocalArtifactStore(
        repository_root,
        producer_stage="e02",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash="0" * 64
        if task is None
        else canonical_json_hash(task.identity),
        dependencies=() if task is None else task.inputs,
        store_relative_root=relative.as_posix(),
    )


def _config(root: Path) -> E02Config:
    return load_e02_config(root / "configs/e02/e02_v1.yaml")


def _stage2_config(root: Path) -> Stage2Config:
    return load_stage2_config(root / "configs/stage2/stage2_v1.yaml")


def _publish_json(root: Path, task: TaskSpec, value: object) -> ArtifactRef:
    return e02_artifact_store(root, task).publish_bytes(
        canonical_json_bytes(value) + b"\n",
        task.output_artifact_type,
        "application/json",
        _SCHEMA,
    )


def _publish_torch(root: Path, task: TaskSpec, value: dict[str, Any]) -> ArtifactRef:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return e02_artifact_store(root, task).publish_bytes(
        buffer.getvalue(),
        task.output_artifact_type,
        "application/x-pytorch",
        _SCHEMA,
    )


def _load_json(root: Path, ref: ArtifactRef) -> dict[str, Any]:
    value = json.loads(e02_artifact_store(root).load_bytes(ref))
    if not isinstance(value, dict):
        raise ValueError("E02 JSON artifact must contain an object")
    return cast(dict[str, Any], value)


def _load_torch(root: Path, ref: ArtifactRef) -> dict[str, Any]:
    value = torch.load(
        io.BytesIO(e02_artifact_store(root).load_bytes(ref)),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(value, dict):
        raise ValueError("E02 tensor artifact must contain a mapping")
    return cast(dict[str, Any], value)


def _external_path(root: Path, ref: ArtifactRef) -> Path:
    if ref.relative_path is None:
        raise ValueError("external E02 input requires a repository-relative path")
    path = (root / ref.relative_path).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("external E02 input is missing or outside the repository")
    if sha256_file(path) != ref.content_hash:
        raise ValueError("external E02 input hash drifted")
    return path


def _read_external_json(root: Path, ref: ArtifactRef) -> dict[str, Any]:
    value = json.loads(_external_path(root, ref).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("external E02 JSON input must contain an object")
    return cast(dict[str, Any], value)


def _grant(value: object) -> SealedAccessGrant:
    return SealedAccessGrant.model_validate_json(canonical_json_bytes(value))


def verify_grant_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    if len(task.inputs) != 1 or task.inputs[0].artifact_type != "SEALED_ACCESS_GRANT":
        raise ValueError("E02 grant verification requires exactly one sealed grant")
    grant = _grant(_read_external_json(root, task.inputs[0]))
    expected = _config(root)
    if grant.scope_name != f"{expected.experiment_id}-formal":
        raise ValueError("E02 grant scope does not match the frozen experiment")
    return _publish_json(root, task, {"grant": grant.model_dump(mode="json")})


def _stage2_completed(root: Path) -> dict[str, ArtifactRef]:
    runtime = _stage2_artifact_root(root) / "runtime"
    launch_path = runtime / "launch_authorization_receipt.json"
    if not launch_path.is_file():
        raise ValueError("Stage 2 launch receipt is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if not isinstance(launch, dict):
        raise ValueError("Stage 2 launch receipt is invalid")
    unsigned = dict(launch)
    receipt_sha = unsigned.pop("receipt_sha256", None)
    if receipt_sha != canonical_json_hash(unsigned):
        raise ValueError("Stage 2 launch receipt hash drifted")
    run_id = launch.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("Stage 2 launch receipt has no run identity")
    state = ExecutionStateStore(
        runtime / "execution.sqlite3",
        artifact_verifier=stage2_artifact_store(root).verify_artifact,
    )
    completed = state.completed_artifacts(run_id)
    if not completed:
        raise ValueError("Stage 2 execution has no completed artifacts")
    return completed


def _only_ref(refs: tuple[ArtifactRef, ...], label: str) -> ArtifactRef:
    if len(refs) != 1:
        raise ValueError(f"requires exactly one {label} artifact")
    return refs[0]


def _only_payload(values: tuple[dict[str, Any], ...], label: str) -> dict[str, Any]:
    if len(values) != 1:
        raise ValueError(f"requires exactly one {label} payload")
    return values[0]


def _verify_stage2_artifacts(
    root: Path, manifest: Stage2Manifest, completed: dict[str, ArtifactRef]
) -> dict[str, Any]:
    refs = tuple(completed.values())
    data_ref = _only_ref(
        tuple(ref for ref in refs if ref.artifact_type == "STAGE2_DEVELOPMENT_DATA"),
        "development data",
    )
    bundle = _bundle_from_payload(_load_stage2_torch(root, data_ref))
    if (
        bundle.manifest_sha256 != manifest.data_manifest_sha256
        or _normalizer_sha256(bundle.normalizer) != manifest.normalizer_sha256
    ):
        raise ValueError("frozen Stage 2 data or normalizer hash drifted")

    predictor_hashes = dict(manifest.predictor_sha256)
    baseline_refs: dict[str, ArtifactRef] = {}
    for ref in refs:
        if ref.artifact_type != "STAGE2_PREDICTOR":
            continue
        model_id = str(_load_stage2_torch(root, ref)["model_id"])
        if model_id in baseline_refs or predictor_hashes.get(model_id) != ref.content_hash:
            raise ValueError("frozen Stage 2 baseline identity drifted")
        baseline_refs[model_id] = ref
    if set(baseline_refs) != {"LAST_VALUE", "SEASONAL_NAIVE", "VAR", "DLINEAR"}:
        raise ValueError("frozen Stage 2 baseline set is incomplete")

    expected = {
        (model_id, seed): digest
        for model_id, seed, digest in manifest.neural_checkpoint_sha256
    }
    itransformer: dict[int, ArtifactRef] = {}
    for ref in refs:
        if ref.artifact_type != "VALIDATED_STAGE2_CHECKPOINT":
            continue
        value = _load_stage2_torch(root, ref)
        model_id, seed = str(value["model_id"]), int(value["seed"])
        if expected.get((model_id, seed)) != ref.content_hash:
            raise ValueError("frozen Stage 2 checkpoint identity drifted")
        if model_id == "ITRANSFORMER":
            itransformer[seed] = ref
    seed_order = _stage2_config(root).training.initialization_seeds
    if set(itransformer) != set(seed_order):
        raise ValueError("frozen iTransformer checkpoint set is incomplete")
    aggregate = canonical_json_hash(sorted(ref.content_hash for ref in itransformer.values()))
    if predictor_hashes.get("ITRANSFORMER") != aggregate:
        raise ValueError("frozen iTransformer family identity drifted")
    if manifest.primary_itransformer.seed not in itransformer:
        raise ValueError("primary iTransformer seed is not frozen")

    return {
        "stage2_manifest": manifest.payload(),
        "development_data_ref": data_ref.model_dump(mode="json"),
        "baseline_refs": {
            model_id: ref.model_dump(mode="json")
            for model_id, ref in sorted(baseline_refs.items())
        },
        "itransformer_checkpoints": [
            {"seed": seed, "ref": itransformer[seed].model_dump(mode="json")}
            for seed in seed_order
        ],
    }


def verify_stage2_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    if len(task.inputs) != 1 or task.inputs[0].artifact_type != "STAGE2_FREEZE_RECEIPT":
        raise ValueError("E02 requires exactly one Stage 2 freeze receipt")
    _external_path(root, task.inputs[0])
    receipt = verify_frozen_stage2_suite(_stage2_artifact_root(root))
    manifest_path = _stage2_artifact_root(root) / "frozen/v1/stage2_manifest.json"
    manifest = stage2_manifest_from_payload(json.loads(manifest_path.read_text(encoding="utf-8")))
    payload = _verify_stage2_artifacts(root, manifest, _stage2_completed(root))
    return _publish_json(
        root,
        task,
        {**payload, "stage2_freeze_receipt": receipt.model_dump(mode="json")},
    )


def open_formal_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    grant_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "VERIFIED_E02_GRANT"),
        "verified E02 grant",
    )
    stage2_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "VERIFIED_STAGE2_FREEZE"),
        "verified Stage 2 freeze",
    )
    grant = _grant(_load_json(root, grant_ref)["grant"])
    stage2 = _load_json(root, stage2_ref)
    data_ref = ArtifactRef.model_validate(stage2["development_data_ref"])
    development = _bundle_from_payload(_load_stage2_torch(root, data_ref))
    stage2_config = _stage2_config(root)
    world = build_world(
        load_world_suite(root / "configs/stage1b/worlds_v2.yaml").world(
            stage2_config.upstream.world_id
        )
    )
    workers = min(int(os.environ.get("TARCA_E02_DATA_WORKERS", "24")), 24)
    bundle = open_formal_bundle(
        stage2_config,
        _config(root),
        grant,
        accessed_at=datetime.now(UTC),
        normalizer=development.normalizer,
        world=world,
        worker_count=workers,
    )
    return _publish_torch(
        root,
        task,
        {
            **_bundle_payload(bundle),
            "e02_config_sha256": _config(root).scientific_hash(),
            "stage2_freeze_receipt_sha256": stage2["stage2_freeze_receipt"][
                "receipt_sha256"
            ],
        },
    )


def _formal_batch(bundle: Stage2DataBundle) -> WindowBatch:
    samples = tuple(
        sample
        for partition in (
            DatasetWindowPartition.TEST_SEEN_REGIME,
            DatasetWindowPartition.TEST_UNSEEN_REGIME,
        )
        for sample in bundle.for_partition(partition)
    )
    if not samples:
        raise ValueError("formal E02 bundle contains no windows")
    x = torch.stack(tuple(sample.history for sample in samples))
    y = torch.stack(tuple(sample.target for sample in samples))
    names = tuple(f"x{index}" for index in range(x.shape[2]))
    origin = datetime(2026, 1, 1, tzinfo=UTC)
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
        window_id=tuple(sample.lineage.window_id for sample in samples),
        input_feature_names=names,
        target_names=names,
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=tuple(origin for _ in samples),
        feature_end=tuple(origin + timedelta(hours=bundle.history - 1) for _ in samples),
        prediction_start=tuple(origin + timedelta(hours=bundle.history) for _ in samples),
        label_end=tuple(
            origin + timedelta(hours=bundle.history + bundle.horizon - 1)
            for _ in samples
        ),
        forecast_time=tuple(
            tuple(
                origin + timedelta(hours=bundle.history + step)
                for step in range(bundle.horizon)
            )
            for _ in samples
        ),
        metadata={"partition": "E02_FORMAL"},
    )


def _baseline_distribution(model: Any, batch: WindowBatch) -> ForecastDistribution:
    with torch.no_grad():
        return validate_forecast_distribution(model.predict_distribution(batch))


def _neural_distribution(
    model: Any, histories: Tensor, *, precision: str, batch_size: int
) -> ForecastDistribution:
    if not torch.cuda.is_available():
        raise RuntimeError("E02 neural prediction requires CUDA")
    device = torch.device("cuda:0")
    model.to(device).freeze()
    means: list[Tensor] = []
    scales: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, histories.shape[0], batch_size):
            values = histories[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=precision == "AMP_FP16"
            ):
                distribution = validate_forecast_distribution(
                    model.forward_distribution(values)
                )
            if distribution.scale is None:
                raise ValueError("iTransformer formal prediction requires Gaussian scale")
            means.append(distribution.mean.float().cpu())
            scales.append(distribution.scale.float().cpu())
    mean, scale = torch.cat(means), torch.cat(scales)
    return validate_forecast_distribution(
        ForecastDistribution(
            mean=mean,
            scale=scale,
            quantiles={},
            logits=None,
            samples=None,
            window_id=None,
            target_names=tuple(f"x{index}" for index in range(mean.shape[2])),
        )
    )


def _distribution_payload(distribution: ForecastDistribution) -> dict[str, Tensor]:
    if distribution.scale is None:
        raise ValueError("E02 formal prediction requires Gaussian scale")
    return {
        "mean": distribution.mean.detach().cpu(),
        "scale": distribution.scale.detach().cpu(),
    }


def predict_formal_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    formal_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_FORMAL_DATA"),
        "formal data",
    )
    stage2_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "VERIFIED_STAGE2_FREEZE"),
        "verified Stage 2 freeze",
    )
    formal_payload = _load_torch(root, formal_ref)
    bundle = _bundle_from_payload(formal_payload)
    batch = _formal_batch(bundle)
    stage2 = _load_json(root, stage2_ref)
    manifest = stage2_manifest_from_payload(stage2["stage2_manifest"])
    baseline_refs = {
        model_id: ArtifactRef.model_validate(value)
        for model_id, value in stage2["baseline_refs"].items()
    }
    guardrails: dict[str, dict[str, Tensor]] = {}
    if task.identity.model_id == "STRONGEST_LINEAR":
        actual_model_id = manifest.strongest_linear.model_id
        model_ref = baseline_refs[actual_model_id]
        model = _baseline_from_payload(
            root, _stage2_config(root), _load_stage2_torch(root, model_ref)
        )
        distribution = _baseline_distribution(model, batch)
        for model_id in ("LAST_VALUE", "SEASONAL_NAIVE"):
            candidate = _baseline_from_payload(
                root,
                _stage2_config(root),
                _load_stage2_torch(root, baseline_refs[model_id]),
            )
            guardrails[model_id] = _distribution_payload(
                _baseline_distribution(candidate, batch)
            )
        stage2_seed, primary = 0, False
    else:
        index = int(task.identity.model_id.rsplit("_", 1)[1])
        checkpoints = stage2["itransformer_checkpoints"]
        if not 0 <= index < len(checkpoints):
            raise ValueError("E02 iTransformer initialization index is invalid")
        selected = checkpoints[index]
        stage2_seed = int(selected["seed"])
        model_ref = ArtifactRef.model_validate(selected["ref"])
        checkpoint = _load_stage2_torch(root, model_ref)
        model = _new_neural(
            root, _stage2_config(root), "ITRANSFORMER", batch.x.shape[2]
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if model.model_hash != checkpoint["model_sha256"]:
            raise ValueError("frozen iTransformer state identity drifted")
        distribution = _neural_distribution(
            model,
            batch.x,
            precision=str(checkpoint["precision"]),
            batch_size=int(os.environ.get("TARCA_E02_INFERENCE_BATCH_SIZE", "512")),
        )
        actual_model_id = "ITRANSFORMER"
        primary = stage2_seed == manifest.primary_itransformer.seed
    if distribution.scale is None:
        raise ValueError("E02 formal prediction requires Gaussian scale")
    finite = bool(
        torch.isfinite(distribution.mean).all()
        and torch.isfinite(distribution.scale).all()
    )
    positive = bool((distribution.scale > 0).all())
    return _publish_torch(
        root,
        task,
        {
            "schema_version": _SCHEMA,
            "model_role": task.identity.model_id,
            "actual_model_id": actual_model_id,
            "stage2_seed": stage2_seed,
            "primary": primary,
            "window_ids": batch.window_id,
            "target_names": batch.target_names,
            **_distribution_payload(distribution),
            "guardrails": guardrails,
            "finite_probabilities": finite,
            "positive_scales": positive,
            "non_crossing_quantiles": finite and positive,
            "e02_config_sha256": formal_payload["e02_config_sha256"],
            "stage2_freeze_receipt_sha256": formal_payload[
                "stage2_freeze_receipt_sha256"
            ],
        },
    )


def _samples(bundle: Stage2DataBundle) -> tuple[Any, ...]:
    return tuple(
        sample
        for partition in (
            DatasetWindowPartition.TEST_SEEN_REGIME,
            DatasetWindowPartition.TEST_UNSEEN_REGIME,
        )
        for sample in bundle.for_partition(partition)
    )


def _score_distributions(
    bundle: Stage2DataBundle,
    prediction: dict[str, Tensor],
    *,
    window_ids: tuple[str, ...],
    target_names: tuple[str, ...],
) -> tuple[TrajectoryScore, ...]:
    samples = _samples(bundle)
    if tuple(sample.lineage.window_id for sample in samples) != window_ids:
        raise ValueError("E02 prediction windows do not align with formal data")
    mean, scale = prediction["mean"], prediction["scale"]
    if mean.shape[0] != len(samples) or scale.shape != mean.shape:
        raise ValueError("E02 prediction tensor shape does not align with formal data")
    records = {record.trajectory_id: record for record in bundle.records}
    scores: list[TrajectoryScore] = []
    for trajectory_id in sorted(records):
        indices = tuple(
            index
            for index, sample in enumerate(samples)
            if sample.lineage.trajectory_id == trajectory_id
        )
        if not indices:
            raise ValueError("formal trajectory contains no forecast origins")
        record = records[trajectory_id]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        target = torch.stack(tuple(samples[index].target for index in indices))
        forecast = ForecastDistribution(
            mean=mean.index_select(0, index_tensor).to(target.dtype),
            scale=scale.index_select(0, index_tensor).to(target.dtype),
            quantiles={},
            logits=None,
            samples=None,
            window_id=tuple(window_ids[index] for index in indices),
            target_names=target_names,
        )
        regime = (
            "SEEN"
            if record.partition is DatasetWindowPartition.TEST_SEEN_REGIME
            else "UNSEEN"
        )
        scores.append(
            score_trajectory(
                forecast,
                target,
                TrajectoryLineage(
                    trajectory_id=trajectory_id,
                    formal_seed=record.data_seed,
                    regime=cast(Any, regime),
                    origin_count=len(indices),
                ),
            )
        )
    return tuple(scores)


def score_trajectories_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    prediction_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_FORMAL_PREDICTION"),
        "formal prediction",
    )
    formal_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_FORMAL_DATA"),
        "formal data",
    )
    prediction = _load_torch(root, prediction_ref)
    bundle = _bundle_from_payload(_load_torch(root, formal_ref))
    window_ids, target_names = (
        tuple(prediction["window_ids"]),
        tuple(prediction["target_names"]),
    )
    scores = _score_distributions(
        bundle,
        {"mean": prediction["mean"], "scale": prediction["scale"]},
        window_ids=window_ids,
        target_names=target_names,
    )
    guardrail_scores = {
        model_id: [
            asdict(item)
            for item in _score_distributions(
                bundle,
                value,
                window_ids=window_ids,
                target_names=target_names,
            )
        ]
        for model_id, value in prediction["guardrails"].items()
    }
    return _publish_json(
        root,
        task,
        {
            "schema_version": _SCHEMA,
            "model_role": prediction["model_role"],
            "stage2_seed": prediction["stage2_seed"],
            "primary": prediction["primary"],
            "scores": [asdict(item) for item in scores],
            "guardrail_scores": guardrail_scores,
            "finite_probabilities": prediction["finite_probabilities"],
            "positive_scales": prediction["positive_scales"],
            "non_crossing_quantiles": prediction["non_crossing_quantiles"],
            "e02_config_sha256": prediction["e02_config_sha256"],
            "stage2_freeze_receipt_sha256": prediction[
                "stage2_freeze_receipt_sha256"
            ],
        },
    )


def bootstrap_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    payloads = tuple(_load_json(root, ref) for ref in task.inputs)
    linear = _only_payload(
        tuple(item for item in payloads if item["model_role"] == "STRONGEST_LINEAR"),
        "strongest linear scores",
    )
    neural = tuple(item for item in payloads if item["model_role"] != "STRONGEST_LINEAR")
    if len(neural) != 3:
        raise ValueError("E02 bootstrap requires all three iTransformer initializations")
    primary = _only_payload(
        tuple(item for item in neural if item["primary"]),
        "primary iTransformer scores",
    )
    linear_scores, primary_scores = _score_set(linear), _score_set(primary)
    config = _config(root)
    interval = paired_stratified_bootstrap(primary_scores, linear_scores, config.bootstrap)
    summary = summarize_scores(primary_scores, linear_scores)
    positive_initializations = sum(
        summarize_scores(_score_set(item), linear_scores).crps_skill > 0.0
        for item in neural
    )
    guardrails = {
        model_id: tuple(_trajectory_score(item) for item in values)
        for model_id, values in linear["guardrail_scores"].items()
    }
    primary_crps = sum(item.crps for item in primary_scores) / len(primary_scores)
    identities = {
        (item["e02_config_sha256"], item["stage2_freeze_receipt_sha256"])
        for item in payloads
    }
    if len(identities) != 1:
        raise ValueError("E02 score artifacts do not share one frozen identity")
    e02_sha, stage2_sha = next(iter(identities))
    evidence = E02Evidence(
        e02_config_sha256=e02_sha,
        stage2_freeze_receipt_sha256=stage2_sha,
        score_summary=summary,
        bootstrap=interval,
        completed_trajectories=len(primary_scores),
        failed_trajectory_ids=(),
        integrity_violation_ids=(),
        finite_probabilities=all(bool(item["finite_probabilities"]) for item in payloads),
        positive_scales=all(bool(item["positive_scales"]) for item in payloads),
        non_crossing_quantiles=all(
            bool(item["non_crossing_quantiles"]) for item in payloads
        ),
        better_than_last_value=primary_crps
        < sum(item.crps for item in guardrails["LAST_VALUE"])
        / len(guardrails["LAST_VALUE"]),
        better_than_seasonal_naive=primary_crps
        < sum(item.crps for item in guardrails["SEASONAL_NAIVE"])
        / len(guardrails["SEASONAL_NAIVE"]),
        positive_initializations=positive_initializations,
    )
    return _publish_json(root, task, {"evidence": evidence.payload()})


def decide_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    bootstrap_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_BOOTSTRAP_EVIDENCE"),
        "bootstrap evidence",
    )
    evidence = _evidence(_load_json(root, bootstrap_ref)["evidence"])
    return _publish_json(
        root,
        task,
        {"decision": evaluate_e02(evidence, _config(root)).payload()},
    )


def publish_receipt_job(
    root: Path, task: TaskSpec, context: ExecutionContext, progress: ProgressSink
) -> ArtifactRef:
    del context, progress
    bootstrap_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_BOOTSTRAP_EVIDENCE"),
        "bootstrap evidence",
    )
    decision_ref = _only_ref(
        tuple(ref for ref in task.inputs if ref.artifact_type == "E02_DECISION"),
        "E02 decision",
    )
    evidence = _evidence(_load_json(root, bootstrap_ref)["evidence"])
    decision = _decision(_load_json(root, decision_ref)["decision"])
    receipt = build_e02_receipt(decision, evidence)
    _write_final_e02(_root(root), evidence, decision)
    return _publish_json(root, task, receipt.model_dump(mode="json"))


def e02_executor_registry(repository_root: Path) -> ExecutorRegistry:
    root = repository_root.resolve()
    return ExecutorRegistry(
        {
            "e02.verify_grant": partial(verify_grant_job, root),
            "e02.verify_stage2": partial(verify_stage2_job, root),
            "e02.open_formal": partial(open_formal_job, root),
            "e02.predict_formal": partial(predict_formal_job, root),
            "e02.score_trajectories": partial(score_trajectories_job, root),
            "e02.bootstrap": partial(bootstrap_job, root),
            "e02.decide": partial(decide_job, root),
            "e02.publish_receipt": partial(publish_receipt_job, root),
        }
    )
