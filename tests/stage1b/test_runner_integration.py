from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tarca.contracts import ArtifactRef
from tarca.execution import (
    HostTelemetry,
    ResourceCapacity,
    ResourceRequest,
    ScientificIdentity,
    TaskSpec,
)
from tarca.execution.state import AttemptState, ExecutionStateStore
from tarca.stage1b.compiler import compile_stage1b_graph, repository_v2_inputs
from tarca.stage1b.runner import (
    QualificationBoundaryError,
    _enqueue_ready_tasks,
    _qualification_execution_evidence,
    _runtime_plan_nodes,
    _runtime_scheduler,
    run_hardware_probe,
    run_qualification,
    validate_qualification_receipt_boundaries,
)

from .receipt_helpers import passing_receipt

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _FormalProbe:
    def host_snapshot(self, process_id: int) -> HostTelemetry:
        assert process_id > 0
        return HostTelemetry(
            host_cpu_percent=80.0,
            effective_busy_cores=20.0,
            process_rss_bytes=512 * 1024**2,
            process_pss_bytes=384 * 1024**2,
            process_affinity_cpu_ids=tuple(range(24)),
            host_memory_used_bytes=96 * 1024**3,
            disk_read_bytes_per_second=1024.0,
            disk_write_bytes_per_second=2048.0,
        )

    def gpu_samples(self) -> tuple[object, ...]:
        return ()


class _FormalBackend:
    backend_id = "formal-test"

    def launch(self, task: object, database_path: Path) -> object:
        del task, database_path
        return object()

    def poll(self) -> tuple[object, ...]:
        return ()


def _formal_task(index: int) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id=f"formal-task-{index}",
            model_id="itransformer",
            data_id="world-a",
            seed=104729 + index,
        ),
        phase="NEURAL_TRAIN",
        inputs=(),
        output_artifact_type="TEST_ARTIFACT",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=1,
            gpu_memory_gib=20.0,
            host_memory_gib=16.0,
        ),
    )


def test_runner_receipt_never_exposes_formal_partition_or_experiment() -> None:
    receipt = passing_receipt()
    validated = validate_qualification_receipt_boundaries(receipt)
    assert "TEST" not in validated["partition_names"]
    assert validated["experiment_ids"] == []


def test_runner_rejects_formal_partition_in_receipt() -> None:
    receipt = passing_receipt()
    receipt["partition_names"] = ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "TEST"]
    with pytest.raises(QualificationBoundaryError, match="qualification partitions"):
        validate_qualification_receipt_boundaries(receipt)


def test_runner_rejects_e02_identifier_in_receipt() -> None:
    receipt = passing_receipt()
    receipt["experiment_ids"] = ["E02"]
    with pytest.raises(QualificationBoundaryError, match="formal experiment"):
        validate_qualification_receipt_boundaries(receipt)


def test_runner_rejects_reserved_seed_in_qualification_rows() -> None:
    receipt = passing_receipt()
    receipt["comparisons"] = [{"seed": 201}]

    with pytest.raises(QualificationBoundaryError, match="reserved formal seed"):
        validate_qualification_receipt_boundaries(receipt)


def test_runner_rejects_overlapping_qualification_and_formal_seeds() -> None:
    receipt = passing_receipt()
    receipt["qualification_seeds"] = [101, 201]

    with pytest.raises(QualificationBoundaryError, match="reserved formal seeds"):
        validate_qualification_receipt_boundaries(receipt)


def test_scheduled_evidence_hashes_complete_graph_and_preflight_receipts(
    tmp_path: Path,
) -> None:
    inputs = repository_v2_inputs(REPOSITORY_ROOT)
    graph = compile_stage1b_graph(inputs)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    receipts = {
        "environment_receipt_v2.json": {"cuda_probe_passed": True},
        "precision_receipt_v2.json": {"selected": "FP32"},
        "official_sources_receipt_v2.json": {
            "sources": [
                {"source_id": source.source_id, "commit": source.commit}
                for source in inputs.world_suite.sources
            ]
        },
        "hardware_probe_v2.json": {
            "created_at_utc": "2026-08-26T00:00:00+00:00",
            "decision": {"feasible": True},
        },
    }
    for filename, payload in receipts.items():
        (runtime_root / filename).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    completed = {
        node.node_id: ArtifactRef(
            artifact_id=f"artifact-{index}",
            artifact_type=node.output_artifact_type,
            content_hash=f"{index + 1:064x}",
            schema_version="2.0.0",
            relative_path=f"stage1b/{index}.json",
        )
        for index, node in enumerate(graph.nodes[:-1])
    }
    capacity = ResourceCapacity(
        logical_cpu_count=56,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=1024**4,
    )

    evidence = _qualification_execution_evidence(
        REPOSITORY_ROOT, runtime_root, graph, completed, capacity
    )

    assert evidence["completed_task_count"] == len(graph.nodes)
    assert evidence["expected_task_count"] == len(graph.nodes)
    assert all(
        len(str(evidence[name])) == 64
        for name in (
            "official_source_receipt_sha256",
            "reproduction_receipt_sha256",
            "environment_receipt_sha256",
            "precision_receipt_sha256",
            "run_graph_sha256",
            "task_manifest_sha256",
            "execution_plan_sha256",
            "hardware_receipt_sha256",
        )
    )
    assert (runtime_root / "qualification_execution_evidence_v2.json").is_file()


def test_runtime_plan_contains_all_74_frozen_graph_nodes() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))

    nodes = _runtime_plan_nodes(graph)

    assert len(nodes) == len(graph.nodes) == 74
    assert tuple(node.task_id for node in nodes) == tuple(node.node_id for node in graph.nodes)


def test_ready_child_is_enqueued_while_an_unrelated_root_is_still_running(
    tmp_path: Path,
) -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    run_id = "run-dynamic"
    store = ExecutionStateStore(tmp_path / "execution.sqlite3", artifact_verifier=lambda _: True)
    store.create_run(run_id, graph.graph_id)
    store.register_run_plan(run_id, _runtime_plan_nodes(graph))
    roots = tuple(node for node in graph.nodes if not node.dependency_ids)
    completed_root, running_root = roots[:2]

    completed_task = TaskSpec(
        identity=completed_root.identity,
        phase=completed_root.phase,
        inputs=(),
        output_artifact_type=completed_root.output_artifact_type,
        resource_request=completed_root.resource_request,
    )
    completed_attempt = store.enqueue_task(
        run_id, completed_task, completed_root.executor_key
    )
    store.transition(completed_attempt, AttemptState.READY, AttemptState.RUNNING)
    store.complete_attempt(
        completed_attempt,
        ArtifactRef(
            artifact_id="completed-root-output",
            artifact_type=completed_root.output_artifact_type,
            content_hash="a" * 64,
            schema_version="2.0.0",
            relative_path="stage1b/completed-root.json",
        ),
    )
    running_task = TaskSpec(
        identity=running_root.identity,
        phase=running_root.phase,
        inputs=(),
        output_artifact_type=running_root.output_artifact_type,
        resource_request=running_root.resource_request,
    )
    running_attempt = store.enqueue_task(run_id, running_task, running_root.executor_key)
    store.transition(running_attempt, AttemptState.READY, AttemptState.RUNNING)

    enqueued = _enqueue_ready_tasks(graph, store, run_id)

    child = next(
        node for node in graph.nodes if node.dependency_ids == (completed_root.node_id,)
    )
    assert child.node_id in enqueued
    assert store.attempt_state(f"{child.node_id}-attempt-1") is AttemptState.READY
    assert store.attempt_state(running_attempt) is AttemptState.RUNNING


def test_formal_runtime_scheduler_samples_and_respects_two_gpu_capacity(
    tmp_path: Path,
) -> None:
    store = ExecutionStateStore(tmp_path / "execution.sqlite3", artifact_verifier=lambda _: True)
    store.create_run("run-a", "graph-a")
    for index in range(3):
        store.enqueue_task("run-a", _formal_task(index), "test.execute")
    capacity = ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=500 * 1024**3,
    )
    scheduler = _runtime_scheduler(
        store,
        _FormalBackend(),
        capacity,
        telemetry_probe=_FormalProbe(),
    )

    launches_per_tick = tuple(len(scheduler.tick("run-a")) for _ in range(3))

    assert launches_per_tick == (2, 0, 0)
    assert len(store.resource_samples("run-a", attempt_id=None)) == 1


def _truth() -> dict[str, bool]:
    return {
        "shared_future_noise": True,
        "graph": True,
        "signed_graph": True,
        "causal_lag": True,
        "regime": True,
        "source_pairs": True,
        "negative_controls": True,
    }


def _world(world_id: str, family_id: str) -> dict[str, object]:
    return {
        "world_id": world_id,
        "family_id": family_id,
        "role": "PRIMARY_MECHANISTIC",
        "source_id": "published",
        "adapter": "CORRECTED_CML",
        "dimension": 4,
        "latent_dimension": 0,
        "concepts": ["local_nonlinearity", "propagation"],
        "concept_pairs": [
            {
                "pair_id": "trend_primary",
                "concept": "trend",
                "parameter_family": "alpha",
                "factual_parameter_ref": "official_alpha_2",
                "counterfactual_parameter_ref": "official_alpha_1_45",
                "factual_value": 2.0,
                "counterfactual_value": 1.45,
                "shared_initial_state": True,
                "shared_future_noise": True,
                "evidence_asset_ids": ["test_equation"],
            },
            {
                "pair_id": "scale_primary",
                "concept": "scale",
                "parameter_family": "epsilon",
                "factual_parameter_ref": "official_epsilon_0_3",
                "counterfactual_parameter_ref": "official_epsilon_0_1",
                "factual_value": 0.3,
                "counterfactual_value": 0.1,
                "shared_initial_state": True,
                "shared_future_noise": True,
                "evidence_asset_ids": ["test_equation"],
            },
        ],
        "downstream_mappings": ["network", "spillover"],
        "truth_capabilities": _truth(),
        "graph": {"kind": "RING", "directed": False},
        "generator": {"alpha": 2.0, "sigma": 0.0, "observations_per_step": 1},
        "regimes": [
            {
                "regime_id": "seen",
                "split_role": "SEEN",
                "changed_parameter": "epsilon",
                "parameters": {"epsilon": 0.3},
            },
            {
                "regime_id": "unseen",
                "split_role": "UNSEEN",
                "changed_parameter": "epsilon",
                "parameters": {"epsilon": 0.1},
            },
        ],
    }


def _worlds_payload() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "suite_id": "tiny-stage1b-worlds-v2",
        "sources": [
            {
                "source_id": "published",
                "title": "Published test equation",
                "repository_url": "https://github.com/example/published.git",
                "paper_url": "https://doi.org/10.0000/example",
                "commit": "a" * 40,
                "license_id": "UNDECLARED",
                "code_usage": "DIRECT_OFFICIAL_CODE",
                "authorization_policy": "USER_AUTHORIZED_NO_LICENSE_BLOCK",
                "authorization_id": "stage1b-v2-test-authorization",
                "assets": [
                    {
                        "asset_id": "test_equation",
                        "relative_path": "evidence.py",
                        "sha256": "b" * 64,
                        "required_for": ["REPRODUCTION", "ORACLE"],
                    }
                ],
                "evidence_files": [{"url": "https://example.org/evidence.py", "sha256": "b" * 64}],
            }
        ],
        "worlds": [_world("tiny_cml_a", "family_a"), _world("tiny_cml_b", "family_b")],
    }


def _qualification_payload() -> dict[str, object]:
    common = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 16,
        "dropout": 0.0,
        "batch_size": 64,
        "max_epochs": 2,
        "patience": 1,
        "learning_rate": 0.001,
        "revin": True,
    }
    return {
        "schema_version": "2.0.0",
        "qualification_id": "tiny-qualification-v2",
        "partitions": ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"],
        "qualification_seeds": [101, 103, 107],
        "reserved_formal_seeds": [201, 203],
        "history_length": 8,
        "horizon": 4,
        "horizon_groups": [[1, 2], [3, 4]],
        "trajectory_length": 24,
        "warmup_steps": 4,
        "trajectories_per_partition": {
            "QUAL_TRAIN": 1,
            "QUAL_TUNE": 1,
            "QUAL_SEEN": 1,
            "QUAL_UNSEEN": 1,
        },
        "models": [
            {
                **common,
                "model_id": "patch",
                "adapter": "PATCHTST_REFERENCE",
                "patch_length": 4,
                "patch_stride": 2,
            },
            {**common, "model_id": "inverted", "adapter": "ITRANSFORMER_REFERENCE"},
        ],
        "var_search": {"lag_orders": [1, 2], "ridge": [0.001]},
        "gate": {
            "primary_metric": "CRPS",
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "guardrail_relative_tolerance": 0.05,
            "minimum_primary_families": 1,
            "minimum_comparison_units": 40,
            "minimum_win_rate": 0.65,
            "minimum_skill_score": 0.0,
            "require_seen_and_unseen_majority": True,
        },
    }


def test_tiny_v2_qualification_runs_without_formal_surface(tmp_path: Path) -> None:
    worlds_path = tmp_path / "worlds.yaml"
    qualification_path = tmp_path / "qualification.yaml"
    worlds_path.write_text(yaml.safe_dump(_worlds_payload(), sort_keys=False), encoding="utf-8")
    qualification_path.write_text(
        yaml.safe_dump(_qualification_payload(), sort_keys=False), encoding="utf-8"
    )
    artifact_root = tmp_path / "artifacts"
    probe = run_hardware_probe(worlds_path, qualification_path, artifact_root / "runtime")
    receipt = run_qualification(worlds_path, qualification_path, artifact_root)
    assert probe["decision"]["feasible"] is True
    assert probe["probe_device"] == "cpu"
    assert probe["parallel_work_slots"] == 1
    assert probe["safety_factor"] == 1.25
    assert len(probe["world_observations"]) == 2
    assert len(probe["training_observations"]) == 4
    assert all(observation["device"] == "cpu" for observation in probe["training_observations"])
    assert receipt["partition_names"] == ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"]
    assert receipt["experiment_ids"] == []
    assert len(receipt["world_decisions"]) == 2
    assert len(receipt["training_receipts"]) == 12
    assert (artifact_root / "qualification_v2_summary.json").is_file()
