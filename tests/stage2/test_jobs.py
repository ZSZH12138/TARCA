from __future__ import annotations

import shutil
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from tarca.contracts import (
    ArtifactRef,
    DatasetWindowPartition,
    ForecastDistribution,
    sha256_file,
)
from tarca.execution import ExecutionContext, ResourceRequest, ScientificIdentity, TaskSpec
from tarca.stage2.config import load_stage2_config
from tarca.stage2.data import Stage2Trajectory, prepare_stage2_bundle
from tarca.stage2.jobs import (
    _bundle_from_payload,
    _bundle_payload,
    _load_json,
    _load_torch,
    _publish_json,
    _publish_torch,
    fit_baseline_job,
    freeze_candidate_job,
    generate_development_data_job,
    predict_validation_job,
    publish_receipt_job,
    select_model_job,
    train_neural_job,
    validate_checkpoint_job,
    verify_source_job,
    verify_upstream_job,
)

ROOT = Path(__file__).resolve().parents[2]


class _Progress:
    def report(self, progress: object) -> None:
        del progress


def _task(
    task_id: str,
    phase: str,
    model_id: str,
    output: str,
    *,
    seed: int = 0,
    inputs: tuple[ArtifactRef, ...] = (),
    data_id: str = "lorenz96_twoscale_v2",
) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
            experiment_id="stage2_probabilistic_forecasting_v1",
            task_id=task_id,
            model_id=model_id,
            data_id=data_id,
            seed=seed,
        ),
        phase=phase,
        inputs=inputs,
        output_artifact_type=output,
        resource_request=ResourceRequest(
            cpu_threads=1,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=1.0,
        ),
    )


def _context(task: TaskSpec) -> ExecutionContext:
    return ExecutionContext(
        run_id="stage2-test-run",
        task_id=task.task_id,
        attempt_id=f"{task.task_id}-attempt",
        runtime_identity="test-runtime",
        worker_identity="test-worker",
    )


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", repo / "configs")
    (repo / "artifacts/stage2/runtime").mkdir(parents=True)
    monkeypatch.setenv("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2")
    return repo


def _development_bundle():
    config = load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml")
    records = tuple(
        Stage2Trajectory(
            trajectory_id=f"{partition.value.lower()}-{index}",
            world_id="lorenz96_twoscale_v2",
            regime_id="seen",
            partition=partition,
            data_seed=config.data.development_seeds[index % 3],
            trajectory_seed=1000 + index,
            source_commit=config.source("scoring_rules_l96").commit,
            config_sha256=config.scientific_hash(),
            values=(
                torch.sin(torch.linspace(0, 8, 120, dtype=torch.float64) + index)
                .unsqueeze(1)
                .repeat(1, 2)
            ),
        )
        for partition, count, offset in (
            (DatasetWindowPartition.TRAIN, 6, 0),
            (DatasetWindowPartition.VALIDATION, 3, 6),
        )
        for index in range(offset, offset + count)
    )
    return prepare_stage2_bundle(records, history=64, horizon=24)


def test_baseline_fit_prediction_selection_and_freeze_jobs(repository: Path) -> None:
    bundle = _development_bundle()
    data_task = _task("data", "DEV_DATA", "DATA", "STAGE2_DEVELOPMENT_DATA")
    data_ref = _publish_torch(repository, data_task, _bundle_payload(bundle))
    assert _bundle_from_payload(_load_torch(repository, data_ref)).manifest_sha256 == (
        bundle.manifest_sha256
    )

    predictors: dict[str, ArtifactRef] = {}
    predictions: dict[tuple[str, int], ArtifactRef] = {}
    for model_id in ("LAST_VALUE", "SEASONAL_NAIVE", "VAR"):
        fit_task = _task(
            f"fit-{model_id.lower()}",
            "BASELINE_FIT",
            model_id,
            "STAGE2_PREDICTOR",
            inputs=(data_ref,),
        )
        predictor = fit_baseline_job(
            repository, fit_task, _context(fit_task), _Progress()
        )
        predictors[model_id] = predictor
        predict_task = _task(
            f"predict-{model_id.lower()}",
            "VALIDATION_PREDICT",
            model_id,
            "STAGE2_VALIDATION_PREDICTION",
            inputs=(predictor, data_ref),
        )
        predictions[(model_id, 0)] = predict_validation_job(
            repository, predict_task, _context(predict_task), _Progress()
        )
        assert _load_torch(repository, predictions[(model_id, 0)])[
            "validation_crps"
        ] >= 0.0

    dlinear_predictor_task = _task(
        "dlinear-predictor", "BASELINE_FIT", "DLINEAR", "STAGE2_PREDICTOR"
    )
    predictors["DLINEAR"] = _publish_torch(
        repository,
        dlinear_predictor_task,
        {"model_id": "DLINEAR", "model_sha256": "d" * 64},
    )
    dlinear_prediction_task = _task(
        "dlinear-validation",
        "VALIDATION_PREDICT",
        "DLINEAR",
        "STAGE2_VALIDATION_PREDICTION",
    )
    predictions[("DLINEAR", 0)] = _publish_torch(
        repository,
        dlinear_prediction_task,
        {
            "model_id": "DLINEAR",
            "seed": 0,
            "model_sha256": "d" * 64,
            "validation_crps": 0.0,
            "artifact_ref": "VALIDATION/dlinear-0.pt",
        },
    )
    linear_task = _task(
        "linear-selection",
        "MODEL_SELECT",
        "STRONGEST_LINEAR",
        "STAGE2_MODEL_SELECTION",
        inputs=(predictions[("VAR", 0)], predictions[("DLINEAR", 0)]),
    )
    linear_ref = select_model_job(
        repository, linear_task, _context(linear_task), _Progress()
    )
    assert _load_json(repository, linear_ref)["model_id"] == "DLINEAR"

    config = load_stage2_config(repository / "configs/stage2/stage2_v1.yaml")
    checkpoints: list[ArtifactRef] = []
    for model_id in ("PATCHTST", "ITRANSFORMER"):
        for seed in config.training.initialization_seeds:
            checkpoint_task = _task(
                f"checkpoint-{model_id.lower()}-{seed}",
                "CHECKPOINT_VALIDATE",
                model_id,
                "VALIDATED_STAGE2_CHECKPOINT",
                seed=seed,
            )
            checkpoint = _publish_torch(
                repository,
                checkpoint_task,
                {"model_id": model_id, "seed": seed, "validated": True},
            )
            checkpoints.append(checkpoint)
            prediction_task = _task(
                f"validation-{model_id.lower()}-{seed}",
                "VALIDATION_PREDICT",
                model_id,
                "STAGE2_VALIDATION_PREDICTION",
                seed=seed,
            )
            predictions[(model_id, seed)] = _publish_torch(
                repository,
                prediction_task,
                {
                    "model_id": model_id,
                    "seed": seed,
                    "model_sha256": str(seed).zfill(64)[-64:],
                    "validation_crps": float(seed % 100) / 100.0,
                    "artifact_ref": f"VALIDATION/{model_id.lower()}-{seed}.pt",
                },
            )
    primary_task = _task(
        "primary-selection",
        "MODEL_SELECT",
        "ITRANSFORMER",
        "STAGE2_MODEL_SELECTION",
        inputs=tuple(
            predictions[("ITRANSFORMER", seed)]
            for seed in config.training.initialization_seeds
        ),
    )
    primary_ref = select_model_job(
        repository, primary_task, _context(primary_task), _Progress()
    )

    sources = []
    for index, source in enumerate(config.sources):
        source_task = _task(
            f"source-{index}",
            "SOURCE_VERIFY",
            source.source_id.upper(),
            "VERIFIED_STAGE2_SOURCE",
        )
        sources.append(
            _publish_json(repository, source_task, {"source_id": source.source_id})
        )
    (repository / "artifacts/stage2/runtime/preflight_receipt.json").write_text(
        '{"status":"PREFLIGHT_PASS"}\n', encoding="utf-8"
    )
    freeze_task = _task(
        "freeze-candidate",
        "FREEZE_CANDIDATE",
        "SUITE",
        "STAGE2_FREEZE_CANDIDATE",
        inputs=(
            data_ref,
            *tuple(predictors.values()),
            *tuple(predictions.values()),
            *tuple(checkpoints),
            linear_ref,
            primary_ref,
            *tuple(sources),
        ),
    )
    manifest_ref = freeze_candidate_job(
        repository, freeze_task, _context(freeze_task), _Progress()
    )
    receipt_task = _task(
        "freeze-receipt",
        "STAGE2_RECEIPT",
        "SUITE",
        "STAGE2_FREEZE_RECEIPT",
        inputs=(manifest_ref,),
    )
    receipt_ref = publish_receipt_job(
        repository, receipt_task, _context(receipt_task), _Progress()
    )

    assert _load_json(repository, receipt_ref)["status"] == "FROZEN"
    assert (repository / "artifacts/stage2/frozen/v1/stage2_manifest.json").is_file()


class _FakeNeural(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def freeze(self):
        self.eval()
        return self

    def forward_distribution(self, histories: torch.Tensor) -> ForecastDistribution:
        mean = histories[:, -1:, :].expand(-1, 24, -1) * self.weight
        return ForecastDistribution(
            mean=mean,
            scale=torch.ones_like(mean),
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=None,
            target_names=tuple(f"x{index}" for index in range(mean.shape[2])),
        )


def test_upstream_source_data_and_neural_job_paths(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_stage2_config(repository / "configs/stage2/stage2_v1.yaml")
    stage1_path = repository / "artifacts/stage1b/frozen/v2/manifest.json"
    stage1_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "artifacts/stage1b/frozen/v2/manifest.json", stage1_path)
    e01_path = repository / "artifacts/e01/frozen/v2/qualification_receipt.json"
    e01_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "artifacts/e01/frozen/v2/qualification_receipt.json", e01_path)
    stage1_ref = ArtifactRef(
        artifact_id="stage1b",
        artifact_type="STAGE1B_MANIFEST",
        content_hash=sha256_file(stage1_path),
        schema_version="1.0.0",
        relative_path="artifacts/stage1b/frozen/v2/manifest.json",
    )
    e01_ref = ArtifactRef(
        artifact_id="e01",
        artifact_type="E01_RECEIPT",
        content_hash=config.upstream.e01_receipt_sha256,
        schema_version="1.0.0",
        relative_path="artifacts/e01/frozen/v2/qualification_receipt.json",
    )
    for model_id, ref in (("STAGE1B", stage1_ref), ("E01", e01_ref)):
        task = _task(
            f"verify-{model_id.lower()}",
            "UPSTREAM_VERIFY",
            model_id,
            f"VERIFIED_{model_id}_HANDOFF",
            inputs=(ref,),
        )
        verified = verify_upstream_job(repository, task, _context(task), _Progress())
        assert _load_json(repository, verified)["status"] == "VERIFIED"

    source = config.source("dlinear")
    checkout = repository / "third_party/stage2" / source.source_id / source.commit
    for asset in source.assets:
        destination = checkout / asset.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            ROOT
            / "third_party/stage2"
            / source.source_id
            / source.commit
            / asset.relative_path,
            destination,
        )
    capsule_path = repository / "artifacts/stage2/source-capsules/test.tar.gz"
    capsule_path.parent.mkdir(parents=True)
    capsule_path.write_bytes(b"audited-source-capsule")
    capsule_ref = ArtifactRef(
        artifact_id="capsule",
        artifact_type="STAGE2_SOURCE_CAPSULE",
        content_hash=sha256_file(capsule_path),
        schema_version="1.0.0",
        relative_path="artifacts/stage2/source-capsules/test.tar.gz",
    )
    source_task = _task(
        "verify-source",
        "SOURCE_VERIFY",
        "DLINEAR",
        "VERIFIED_STAGE2_SOURCE",
        inputs=(capsule_ref,),
        data_id="dlinear",
    )
    source_ref = verify_source_job(
        repository, source_task, _context(source_task), _Progress()
    )
    assert _load_json(repository, source_ref)["source_id"] == "dlinear"

    bundle = _development_bundle()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "tarca.stage2.jobs.load_world_suite",
            lambda path: type("Suite", (), {"world": lambda self, world_id: object()})(),
        )
        scoped.setattr("tarca.stage2.jobs.build_world", lambda world: object())
        scoped.setattr(
            "tarca.stage2.jobs.generate_development_bundle",
            lambda config, world, worker_count: bundle,
        )
        data_task = _task("generated-data", "DEV_DATA", "DATA", "STAGE2_DEVELOPMENT_DATA")
        data_ref = generate_development_data_job(
            repository, data_task, _context(data_task), _Progress()
        )
    assert _bundle_from_payload(_load_torch(repository, data_ref)).manifest_sha256 == (
        bundle.manifest_sha256
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            "tarca.stage2.jobs.fit_dlinear_cross_fitted",
            lambda *args, **kwargs: SimpleNamespace(predictor=object()),
        )
        scoped.setattr(
            "tarca.stage2.jobs._baseline_payload",
            lambda model, model_id: {"model_id": model_id, "model_sha256": "d" * 64},
        )
        dlinear_task = _task(
            "fit-dlinear",
            "BASELINE_FIT",
            "DLINEAR",
            "STAGE2_PREDICTOR",
            inputs=(data_ref, source_ref),
        )
        dlinear_ref = fit_baseline_job(
            repository, dlinear_task, _context(dlinear_task), _Progress()
        )
    assert _load_torch(repository, dlinear_ref)["model_id"] == "DLINEAR"

    fake_model = _FakeNeural()
    with monkeypatch.context() as scoped:
        scoped.setattr("tarca.stage2.jobs._new_neural", lambda *args: fake_model)
        scoped.setattr(
            "tarca.stage2.jobs.train_stage2_neural",
            lambda *args, **kwargs: SimpleNamespace(
                completed=True,
                model=fake_model,
                model_sha256="1" * 64,
                checkpoint_sha256="2" * 64,
                fixed_batch_forecast_sha256="3" * 64,
                best_epoch=1,
                best_validation_nll=0.5,
                precision="FP32",
            ),
        )
        train_task = _task(
            "train-patchtst",
            "NEURAL_TRAIN",
            "PATCHTST",
            "STAGE2_NEURAL_CHECKPOINT",
            seed=config.training.initialization_seeds[0],
            inputs=(data_ref, source_ref),
        )
        checkpoint_ref = train_neural_job(
            repository, train_task, _context(train_task), _Progress()
        )
    with monkeypatch.context() as scoped:
        scoped.setattr("tarca.stage2.jobs._new_neural", lambda *args: _FakeNeural())
        validate_task = _task(
            "validate-patchtst",
            "CHECKPOINT_VALIDATE",
            "PATCHTST",
            "VALIDATED_STAGE2_CHECKPOINT",
            seed=config.training.initialization_seeds[0],
            inputs=(checkpoint_ref, data_ref),
        )
        validated_ref = validate_checkpoint_job(
            repository, validate_task, _context(validate_task), _Progress()
        )
        predict_task = _task(
            "predict-patchtst",
            "VALIDATION_PREDICT",
            "PATCHTST",
            "STAGE2_VALIDATION_PREDICTION",
            seed=config.training.initialization_seeds[0],
            inputs=(validated_ref, data_ref),
        )
        prediction_ref = predict_validation_job(
            repository, predict_task, _context(predict_task), _Progress()
        )
    assert _load_torch(repository, prediction_ref)["validation_crps"] >= 0.0
