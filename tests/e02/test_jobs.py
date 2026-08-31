from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

from tarca.contracts import (
    ArtifactRef,
    DatasetWindowPartition,
    ForecastDistribution,
    canonical_json_bytes,
    sha256_file,
)
from tarca.e02.grant import create_e02_grant
from tarca.e02.jobs import (
    _bundle_payload,
    _formal_batch,
    _load_json,
    _load_torch,
    _publish_json,
    _publish_torch,
    _score_distributions,
    _verify_stage2_artifacts,
    bootstrap_job,
    decide_job,
    open_formal_job,
    predict_formal_job,
    publish_receipt_job,
    score_trajectories_job,
    verify_grant_job,
)
from tarca.execution import ExecutionContext, ResourceRequest, ScientificIdentity, TaskSpec
from tarca.stage2.data import (
    Stage2NormalizationStatistics,
    Stage2Trajectory,
    prepare_stage2_bundle,
)
from tests.e02.test_bootstrap import _scores
from tests.stage2.test_manifest import compilation_inputs

ROOT = Path(__file__).resolve().parents[2]


class _Progress:
    def report(self, progress: object) -> None:
        del progress


def _context(task: TaskSpec) -> ExecutionContext:
    return ExecutionContext(
        run_id="run-test",
        task_id=task.task_id,
        attempt_id=f"{task.task_id}-attempt",
        runtime_identity="test-runtime",
        worker_identity="test-worker",
    )


def _task(
    task_id: str,
    phase: str,
    output: str,
    *,
    inputs: tuple[ArtifactRef, ...] = (),
    model_id: str = "TEST",
) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
            experiment_id="e02_predictor_validity_v1",
            task_id=task_id,
            model_id=model_id,
            data_id="TEST",
            seed=0,
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


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", repo / "configs")
    (repo / "artifacts/e02/runtime").mkdir(parents=True)
    monkeypatch.setenv("TARCA_E02_ARTIFACT_ROOT", "artifacts/e02")
    monkeypatch.setenv("TARCA_STAGE2_ARTIFACT_ROOT", "artifacts/stage2")
    return repo


def _formal_bundle():
    normalizer = Stage2NormalizationStatistics(
        mean=torch.zeros(1, dtype=torch.float64),
        standard_deviation=torch.ones(1, dtype=torch.float64),
        fitted_partition=DatasetWindowPartition.TRAIN,
        trajectory_ids=("frozen-train",),
    )
    records = tuple(
        Stage2Trajectory(
            trajectory_id=f"formal-{regime.lower()}",
            world_id="lorenz96_twoscale_v2",
            regime_id=regime.lower(),
            partition=partition,
            data_seed=1729,
            trajectory_seed=2000 + index,
            source_commit="6f28942f6a703c2b52501d01258ca2708539f209",
            config_sha256="a" * 64,
            values=torch.linspace(0, 1, 27, dtype=torch.float64).unsqueeze(1),
        )
        for index, (regime, partition) in enumerate(
            (
                ("SEEN", DatasetWindowPartition.TEST_SEEN_REGIME),
                ("UNSEEN", DatasetWindowPartition.TEST_UNSEEN_REGIME),
            )
        )
    )
    return prepare_stage2_bundle(records, history=2, horizon=24, normalizer=normalizer)


def test_grant_verification_and_formal_trajectory_scoring(repository: Path) -> None:
    authorization = ArtifactRef(
        artifact_id="authorization",
        artifact_type="SEALED_ACCESS_AUTHORIZATION",
        content_hash="a" * 64,
        schema_version="1.0.0",
        relative_path="artifacts/e02/runtime/authorization.json",
    )
    grant = create_e02_grant(authorization, issued_at=datetime.now(UTC))
    grant_path = repository / "artifacts/e02/runtime/sealed_access_grant.json"
    grant_path.write_bytes(canonical_json_bytes(grant) + b"\n")
    grant_ref = ArtifactRef(
        artifact_id="grant",
        artifact_type="SEALED_ACCESS_GRANT",
        content_hash=sha256_file(grant_path),
        schema_version="1.0.0",
        relative_path="artifacts/e02/runtime/sealed_access_grant.json",
    )
    verify_task = _task(
        "verify-grant", "GRANT_VERIFY", "VERIFIED_E02_GRANT", inputs=(grant_ref,)
    )

    verified = verify_grant_job(repository, verify_task, _context(verify_task), _Progress())

    assert _load_json(repository, verified)["grant"]["grant_id"] == grant.grant_id
    bundle = _formal_bundle()
    batch = _formal_batch(bundle)
    prediction = {"mean": batch.y.clone(), "scale": torch.ones_like(batch.y)}
    scores = _score_distributions(
        bundle,
        prediction,
        window_ids=batch.window_id,
        target_names=batch.target_names,
    )
    assert len(scores) == 2
    assert {score.regime for score in scores} == {"SEEN", "UNSEEN"}


def test_score_bootstrap_decision_and_receipt_jobs(repository: Path) -> None:
    bundle = _formal_bundle()
    formal_task = _task("formal-data", "FORMAL_OPEN", "E02_FORMAL_DATA")
    formal_ref = _publish_torch(repository, formal_task, _bundle_payload(bundle))
    batch = _formal_batch(bundle)
    prediction_task = _task("prediction", "FORMAL_PREDICT", "E02_FORMAL_PREDICTION")
    prediction_ref = _publish_torch(
        repository,
        prediction_task,
        {
            "model_role": "ITRANSFORMER_INIT_0",
            "stage2_seed": 1,
            "primary": True,
            "window_ids": batch.window_id,
            "target_names": batch.target_names,
            "mean": batch.y.clone(),
            "scale": torch.ones_like(batch.y),
            "guardrails": {},
            "finite_probabilities": True,
            "positive_scales": True,
            "non_crossing_quantiles": True,
            "e02_config_sha256": "9" * 64,
            "stage2_freeze_receipt_sha256": "a" * 64,
        },
    )
    score_task = _task(
        "score",
        "TRAJECTORY_SCORE",
        "E02_TRAJECTORY_SCORES",
        inputs=(prediction_ref, formal_ref),
    )
    score_ref = score_trajectories_job(
        repository, score_task, _context(score_task), _Progress()
    )
    assert len(_load_json(repository, score_ref)["scores"]) == 2

    neural, baseline = _scores()
    payloads = (
        {
            "model_role": "STRONGEST_LINEAR",
            "stage2_seed": 0,
            "primary": False,
            "scores": [asdict(item) for item in baseline],
            "guardrail_scores": {
                "LAST_VALUE": [asdict(item) for item in baseline],
                "SEASONAL_NAIVE": [asdict(item) for item in baseline],
            },
        },
        *(
            {
                "model_role": f"ITRANSFORMER_INIT_{index}",
                "stage2_seed": index + 1,
                "primary": index == 0,
                "scores": [asdict(item) for item in neural],
                "guardrail_scores": {},
            }
            for index in range(3)
        ),
    )
    score_refs = []
    for index, value in enumerate(payloads):
        task = _task(f"scores-{index}", "TRAJECTORY_SCORE", "E02_TRAJECTORY_SCORES")
        score_refs.append(
            _publish_json(
                repository,
                task,
                {
                    **value,
                    "finite_probabilities": True,
                    "positive_scales": True,
                    "non_crossing_quantiles": True,
                    "e02_config_sha256": (
                        "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c"
                    ),
                    "stage2_freeze_receipt_sha256": "a" * 64,
                },
            )
        )
    bootstrap_task = _task(
        "bootstrap",
        "PAIRED_BOOTSTRAP",
        "E02_BOOTSTRAP_EVIDENCE",
        inputs=tuple(score_refs),
    )
    bootstrap_ref = bootstrap_job(
        repository, bootstrap_task, _context(bootstrap_task), _Progress()
    )
    decision_task = _task(
        "decision",
        "E02_DECISION",
        "E02_DECISION",
        inputs=(bootstrap_ref, *tuple(score_refs)),
    )
    decision_ref = decide_job(
        repository, decision_task, _context(decision_task), _Progress()
    )
    receipt_task = _task(
        "receipt",
        "E02_RECEIPT",
        "E02_RECEIPT",
        inputs=(decision_ref, bootstrap_ref),
    )
    receipt_ref = publish_receipt_job(
        repository, receipt_task, _context(receipt_task), _Progress()
    )

    assert _load_json(repository, receipt_ref)["outcome"] in {
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
        "NOT_EVALUABLE",
    }
    assert (repository / "artifacts/e02/frozen/v1/e02_receipt.json").is_file()


class _FakeBaseline:
    def predict_distribution(self, batch: Any) -> ForecastDistribution:
        return ForecastDistribution(
            mean=torch.zeros_like(batch.y),
            scale=torch.ones_like(batch.y),
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=batch.window_id,
            target_names=batch.target_names,
        )


def test_fixed_formal_prediction_roles_use_frozen_stage2_inputs(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tarca.stage2.manifest import compile_stage2_manifest

    bundle = _formal_bundle()
    formal_task = _task("formal", "FORMAL_OPEN", "E02_FORMAL_DATA")
    formal_ref = _publish_torch(
        repository,
        formal_task,
        {
            **_bundle_payload(bundle),
            "e02_config_sha256": "9" * 64,
            "stage2_freeze_receipt_sha256": "a" * 64,
        },
    )
    manifest = compile_stage2_manifest(compilation_inputs())
    dummy = ArtifactRef(
        artifact_id="stage2-model",
        artifact_type="STAGE2_PREDICTOR",
        content_hash="b" * 64,
        schema_version="1.0.0",
        relative_path=None,
    )
    stage2_task = _task("stage2", "STAGE2_VERIFY", "VERIFIED_STAGE2_FREEZE")
    stage2_ref = _publish_json(
        repository,
        stage2_task,
        {
            "stage2_manifest": manifest.payload(),
            "baseline_refs": {
                model_id: dummy.model_dump(mode="json")
                for model_id in ("LAST_VALUE", "SEASONAL_NAIVE", "VAR", "DLINEAR")
            },
            "itransformer_checkpoints": [
                {"seed": seed, "ref": dummy.model_dump(mode="json")}
                for seed in (1797287582, 883082243, 1933050005)
            ],
        },
    )
    monkeypatch.setattr("tarca.e02.jobs._load_stage2_torch", lambda root, ref: {})
    monkeypatch.setattr(
        "tarca.e02.jobs._baseline_from_payload", lambda root, config, value: _FakeBaseline()
    )
    linear_task = _task(
        "linear-predict",
        "FORMAL_PREDICT",
        "E02_FORMAL_PREDICTION",
        inputs=(formal_ref, stage2_ref),
        model_id="STRONGEST_LINEAR",
    )
    linear_ref = predict_formal_job(
        repository, linear_task, _context(linear_task), _Progress()
    )
    assert set(_load_torch(repository, linear_ref)["guardrails"]) == {
        "LAST_VALUE",
        "SEASONAL_NAIVE",
    }

    class FakeNeural:
        model_hash = "model-hash"

        def load_state_dict(self, state: object, strict: bool) -> None:
            assert state == {} and strict

    monkeypatch.setattr(
        "tarca.e02.jobs._load_stage2_torch",
        lambda root, ref: {
            "state_dict": {},
            "model_sha256": "model-hash",
            "precision": "FP32",
        },
    )
    monkeypatch.setattr("tarca.e02.jobs._new_neural", lambda *args: FakeNeural())
    monkeypatch.setattr(
        "tarca.e02.jobs._neural_distribution",
        lambda model, histories, precision, batch_size: ForecastDistribution(
            mean=torch.zeros((histories.shape[0], 24, 1)),
            scale=torch.ones((histories.shape[0], 24, 1)),
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=None,
            target_names=("x0",),
        ),
    )
    neural_task = _task(
        "neural-predict",
        "FORMAL_PREDICT",
        "E02_FORMAL_PREDICTION",
        inputs=(formal_ref, stage2_ref),
        model_id="ITRANSFORMER_INIT_0",
    )
    neural_ref = predict_formal_job(
        repository, neural_task, _context(neural_task), _Progress()
    )
    assert _load_torch(repository, neural_ref)["stage2_seed"] == 1797287582


def test_stage2_artifact_verification_and_formal_open_job(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tarca.contracts import canonical_json_hash
    from tarca.stage2.manifest import compile_stage2_manifest

    inputs = compilation_inputs()
    itransformer_family = canonical_json_hash(["f" * 64] * 3)
    inputs = replace(
        inputs,
        predictor_sha256=tuple(
            (model_id, itransformer_family if model_id == "ITRANSFORMER" else digest)
            for model_id, digest in inputs.predictor_sha256
        ),
    )
    manifest = compile_stage2_manifest(inputs)

    def ref(identifier: str, artifact_type: str, digest: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=identifier,
            artifact_type=artifact_type,
            content_hash=digest,
            schema_version="1.0.0",
            relative_path=None,
        )

    data_ref = ref("data", "STAGE2_DEVELOPMENT_DATA", "0" * 64)
    completed: dict[str, ArtifactRef] = {"data": data_ref}
    payloads: dict[str, dict[str, Any]] = {"data": {}}
    for model_id, digest in manifest.predictor_sha256[:4]:
        model_ref = ref(f"baseline-{model_id}", "STAGE2_PREDICTOR", digest)
        completed[model_ref.artifact_id] = model_ref
        payloads[model_ref.artifact_id] = {"model_id": model_id}
    for model_id, seed, digest in manifest.neural_checkpoint_sha256:
        checkpoint = ref(
            f"checkpoint-{model_id}-{seed}", "VALIDATED_STAGE2_CHECKPOINT", digest
        )
        completed[checkpoint.artifact_id] = checkpoint
        payloads[checkpoint.artifact_id] = {"model_id": model_id, "seed": seed}

    fake_bundle = type(
        "Bundle",
        (),
        {"manifest_sha256": "6" * 64, "normalizer": object()},
    )()
    monkeypatch.setattr(
        "tarca.e02.jobs._load_stage2_torch",
        lambda root, artifact: payloads[artifact.artifact_id],
    )
    monkeypatch.setattr("tarca.e02.jobs._bundle_from_payload", lambda value: fake_bundle)
    monkeypatch.setattr("tarca.e02.jobs._normalizer_sha256", lambda value: "5" * 64)

    verified = _verify_stage2_artifacts(repository, manifest, completed)

    assert len(verified["itransformer_checkpoints"]) == 3
    development = _formal_bundle()
    grant = create_e02_grant(
        ArtifactRef(
            artifact_id="authorization",
            artifact_type="SEALED_ACCESS_AUTHORIZATION",
            content_hash="a" * 64,
            schema_version="1.0.0",
            relative_path=None,
        ),
        issued_at=datetime.now(UTC),
    )
    grant_task = _task("verified-grant", "GRANT_VERIFY", "VERIFIED_E02_GRANT")
    grant_ref = _publish_json(
        repository, grant_task, {"grant": grant.model_dump(mode="json")}
    )
    stage2_task = _task("verified-stage2", "STAGE2_VERIFY", "VERIFIED_STAGE2_FREEZE")
    stage2_ref = _publish_json(
        repository,
        stage2_task,
        {
            "development_data_ref": data_ref.model_dump(mode="json"),
            "stage2_freeze_receipt": {"receipt_sha256": "a" * 64},
        },
    )
    monkeypatch.setattr(
        "tarca.e02.jobs._load_stage2_torch",
        lambda root, artifact: _bundle_payload(development),
    )
    monkeypatch.setattr(
        "tarca.e02.jobs.load_world_suite",
        lambda path: type("Suite", (), {"world": lambda self, world_id: object()})(),
    )
    monkeypatch.setattr("tarca.e02.jobs.build_world", lambda world: object())
    monkeypatch.setattr(
        "tarca.e02.jobs.open_formal_bundle", lambda *args, **kwargs: development
    )
    formal_task = _task(
        "open-formal",
        "FORMAL_OPEN",
        "E02_FORMAL_DATA",
        inputs=(grant_ref, stage2_ref),
    )
    formal_ref = open_formal_job(
        repository, formal_task, _context(formal_task), _Progress()
    )

    assert _load_torch(repository, formal_ref)["stage2_freeze_receipt_sha256"] == "a" * 64
