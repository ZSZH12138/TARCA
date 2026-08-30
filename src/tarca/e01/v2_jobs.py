from __future__ import annotations

import io
import json
import os
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from tarca.artifacts.store import LocalArtifactStore
from tarca.contracts import ArtifactRef, canonical_json_bytes, canonical_json_hash
from tarca.e01.estimators import EffectSamples, simulate_delayed_effects
from tarca.e01.v2_carry_forward import (
    E01BCarryForwardReceipt,
    verify_e01_b_carry_forward,
)
from tarca.e01.v2_config import E01V2Config, load_e01_v2_config
from tarca.e01.v2_metrics import analyze_e01a_seed, evaluate_e01_v2_gate
from tarca.e01.v2_tasks import compile_e01_v2_graph
from tarca.execution.contracts import ExecutionContext, TaskSpec
from tarca.execution.registry import ExecutorRegistry, ProgressSink

_SCHEMA_VERSION = "2.0.0"


def _config_path(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_E01_V2_CONFIG", "configs/e01/e01_v2.yaml")
    path = Path(raw)
    return (path if path.is_absolute() else repository_root / path).resolve()


def _artifact_root(repository_root: Path) -> Path:
    raw = os.environ.get("TARCA_E01_V2_ARTIFACT_ROOT", "artifacts/e01-v2")
    path = Path(raw)
    resolved = (path if path.is_absolute() else repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("E01-v2 artifact root must stay inside the repository") from error
    return resolved


def e01_v2_artifact_store(
    repository_root: Path,
    task: TaskSpec | None = None,
) -> LocalArtifactStore:
    identity_hash = "0" * 64 if task is None else canonical_json_hash(task.identity)
    relative = (_artifact_root(repository_root) / "runtime/store").relative_to(
        repository_root.resolve()
    )
    return LocalArtifactStore(
        repository_root,
        producer_stage="e01-v2",
        producer_task_id="verifier" if task is None else task.task_id,
        scientific_identity_hash=identity_hash,
        dependencies=() if task is None else task.inputs,
        store_relative_root=relative.as_posix(),
    )


def _publish_torch(repository_root: Path, task: TaskSpec, value: Mapping[str, Any]) -> ArtifactRef:
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    return e01_v2_artifact_store(repository_root, task).publish_bytes(
        buffer.getvalue(),
        task.output_artifact_type,
        "application/x-pytorch-state-dict",
        _SCHEMA_VERSION,
    )


def _load_torch(repository_root: Path, reference: ArtifactRef) -> dict[str, Any]:
    payload = e01_v2_artifact_store(repository_root).load_bytes(reference)
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("E01-v2 effect artifact must contain a mapping")
    return cast(dict[str, Any], value)


def _publish_json(repository_root: Path, task: TaskSpec, value: object) -> ArtifactRef:
    return e01_v2_artifact_store(repository_root, task).publish_bytes(
        canonical_json_bytes(value) + b"\n",
        task.output_artifact_type,
        "application/json",
        _SCHEMA_VERSION,
    )


def _load_json(repository_root: Path, reference: ArtifactRef) -> dict[str, Any]:
    payload = e01_v2_artifact_store(repository_root).load_bytes(reference)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("E01-v2 JSON artifact must contain a mapping")
    return cast(dict[str, Any], value)


def build_v2_seed_effects(
    config: E01V2Config,
    seed: int,
    *,
    device: str,
    batch_size: int,
    simulator: Callable[..., EffectSamples] = simulate_delayed_effects,
) -> dict[str, torch.Tensor]:
    if seed not in config.formal_seeds:
        raise ValueError("generation seed is outside the frozen E01-v2 TEST set")
    world = config.worlds[0]
    effects: dict[str, torch.Tensor] = {}
    for condition in config.conditions:
        sample = simulator(
            seed=seed,
            sample_count=config.sample_sizes[-1],
            horizon=config.horizons[-1],
            true_lag=world.true_lag,
            wrong_lag=world.wrong_lag,
            delta=world.intervention_delta,
            condition=condition,
            device=device,
            batch_size=batch_size,
        )
        effects[condition] = sample.values.detach().to(device="cpu", dtype=torch.float64)
    return effects


def generate_e01a_v2_job(
    repository_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context
    config = load_e01_v2_config(_config_path(repository_root))
    node_ids = {node.task.task_id for node in compile_e01_v2_graph(config).nodes}
    if task.task_id not in node_ids or task.phase != "E01_A_V2_GPU_GENERATE":
        raise ValueError("task is outside the immutable E01-v2 generation graph")
    if not torch.cuda.is_available():
        raise RuntimeError("E01-v2 analytic generation requires the authorized CUDA GPU")
    batch_size = int(os.environ.get("TARCA_E01_V2_GPU_BATCH_SIZE", "8192"))
    effects = build_v2_seed_effects(
        config,
        task.identity.seed,
        device="cuda:0",
        batch_size=batch_size,
    )
    progress.report({"completed_conditions": len(effects), "total_conditions": len(effects)})
    return _publish_torch(
        repository_root,
        task,
        {
            "schema_version": "tarca-e01-v2-effect-block",
            "world_id": config.worlds[0].world_id,
            "seed": task.identity.seed,
            "conditions": effects,
        },
    )


def analyze_e01a_v2_job(
    repository_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context
    if len(task.inputs) != 1:
        raise ValueError("E01-v2 seed analysis requires exactly one effect artifact")
    config = load_e01_v2_config(_config_path(repository_root))
    payload = _load_torch(repository_root, task.inputs[0])
    if payload.get("seed") != task.identity.seed:
        raise ValueError("E01-v2 seed analysis input identity drifted")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("E01-v2 effect conditions are missing")
    report = analyze_e01a_seed(
        config,
        task.identity.seed,
        cast(dict[str, torch.Tensor], conditions),
    )
    progress.report({"completed_steps": 1, "total_steps": 1})
    return _publish_json(repository_root, task, report)


def build_e01_v2_final_report(
    config: E01V2Config,
    reports: Sequence[Mapping[str, Any]],
    carry_forward: E01BCarryForwardReceipt,
) -> dict[str, Any]:
    gate = evaluate_e01_v2_gate(config, reports, carry_forward)
    return {
        "schema_version": "tarca-e01-final-report-v2",
        "experiment_id": config.experiment_id,
        "scientific_config_sha256": config.scientific_hash(),
        "seed_reports": tuple(dict(report) for report in reports),
        "e01_b_carry_forward": carry_forward.model_dump(mode="json"),
        "gate": gate,
        "failed_seed_policy": "NO_SILENT_DELETION",
    }


def aggregate_e01_v2_job(
    repository_root: Path,
    task: TaskSpec,
    context: ExecutionContext,
    progress: ProgressSink,
) -> ArtifactRef:
    del context
    config = load_e01_v2_config(_config_path(repository_root))
    reports = tuple(_load_json(repository_root, reference) for reference in task.inputs)
    if len(reports) != len(config.formal_seeds):
        raise RuntimeError("E01-v2 aggregation requires every formal seed report")
    carry_forward = verify_e01_b_carry_forward(repository_root, config)
    final = build_e01_v2_final_report(config, reports, carry_forward)
    progress.report(
        {"completed_seed_reports": len(reports), "total_seed_reports": len(config.formal_seeds)}
    )
    return _publish_json(repository_root, task, final)


def _bound(
    repository_root: Path,
    function: Callable[..., ArtifactRef],
) -> Callable[..., ArtifactRef]:
    return partial(function, repository_root)


def e01_v2_executor_registry(repository_root: Path) -> ExecutorRegistry:
    return ExecutorRegistry(
        {
            "e01.v2.generate": _bound(repository_root, generate_e01a_v2_job),
            "e01.v2.analyze": _bound(repository_root, analyze_e01a_v2_job),
            "e01.v2.aggregate": _bound(repository_root, aggregate_e01_v2_job),
        }
    )
