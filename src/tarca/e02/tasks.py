from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tarca.contracts import ArtifactRef, Sha256Hash, canonical_json_hash
from tarca.e02.config import E02Config
from tarca.execution import ResourceRequest, ScientificIdentity, TaskManifest, TaskSpec


@dataclass(frozen=True, slots=True)
class FrozenStage2Input:
    freeze_receipt: ArtifactRef
    sealed_access_grant: ArtifactRef | None
    frozen: bool


@dataclass(frozen=True, slots=True)
class E02TaskNode:
    identity: ScientificIdentity
    phase: str
    dependency_ids: tuple[str, ...]
    external_inputs: tuple[ArtifactRef, ...]
    output_artifact_type: str
    resource_request: ResourceRequest
    executor_key: str

    @property
    def node_id(self) -> str:
        return self.identity.task_id


@dataclass(frozen=True, slots=True)
class E02Graph:
    graph_id: str
    config_sha256: Sha256Hash
    nodes: tuple[E02TaskNode, ...]


def _resource(phase: str, model_id: str) -> ResourceRequest:
    values = {
        "GRANT_VERIFY": (1, 0, 0.0, 2.0),
        "STAGE2_VERIFY": (2, 0, 0.0, 4.0),
        "FORMAL_OPEN": (24, 0, 0.0, 96.0),
        "FORMAL_PREDICT": (4, 1, 20.0, 32.0),
        "TRAJECTORY_SCORE": (4, 0, 0.0, 24.0),
        "PAIRED_BOOTSTRAP": (8, 0, 0.0, 48.0),
        "E02_DECISION": (2, 0, 0.0, 8.0),
        "E02_RECEIPT": (2, 0, 0.0, 8.0),
    }
    cpu, gpu, vram, ram = values[phase]
    if phase == "FORMAL_PREDICT" and model_id == "STRONGEST_LINEAR":
        cpu, gpu, vram, ram = (8, 0, 0.0, 32.0)
    return ResourceRequest(cpu_threads=cpu, gpu_count=gpu, gpu_memory_gib=vram, host_memory_gib=ram)


def compile_e02_graph(config: E02Config, frozen_stage2: FrozenStage2Input) -> E02Graph:
    if not frozen_stage2.frozen or frozen_stage2.sealed_access_grant is None:
        raise PermissionError("E02 compilation requires frozen Stage 2 and a sealed-access grant")
    nodes: list[E02TaskNode] = []

    def add(
        phase: str,
        model: str,
        *,
        seed: int = 0,
        deps: tuple[E02TaskNode, ...] = (),
        external: tuple[ArtifactRef, ...] = (),
        output: str,
        executor: str,
        tag: str = "",
    ) -> E02TaskNode:
        digest = canonical_json_hash(
            {
                "config": config.scientific_hash(),
                "phase": phase,
                "model": model,
                "seed": seed,
                "deps": [item.node_id for item in deps],
                "external": [item.model_dump(mode="json") for item in external],
                "output": output,
                "tag": tag,
            }
        )
        identity = ScientificIdentity(
            protocol_id=config.protocol_id,
            experiment_id=config.experiment_id,
            task_id=f"e02-{phase.lower().replace('_', '-')}-{digest}",
            model_id=model,
            data_id=config.formal_partition,
            seed=seed,
        )
        node = E02TaskNode(
            identity,
            phase,
            tuple(item.node_id for item in deps),
            external,
            output,
            _resource(phase, model),
            executor,
        )
        nodes.append(node)
        return node

    grant = add(
        "GRANT_VERIFY",
        "AUTHORIZATION",
        external=(frozen_stage2.sealed_access_grant,),
        output="VERIFIED_E02_GRANT",
        executor="e02.verify_grant",
    )
    frozen = add(
        "STAGE2_VERIFY",
        "FROZEN_SUITE",
        external=(frozen_stage2.freeze_receipt,),
        output="VERIFIED_STAGE2_FREEZE",
        executor="e02.verify_stage2",
    )
    formal = add(
        "FORMAL_OPEN",
        "FORMAL_DATA",
        deps=(grant, frozen),
        output="E02_FORMAL_DATA",
        executor="e02.open_formal",
    )
    prediction_models = (
        ("STRONGEST_LINEAR", 0),
        ("ITRANSFORMER_INIT_0", 0),
        ("ITRANSFORMER_INIT_1", 1),
        ("ITRANSFORMER_INIT_2", 2),
    )
    scores: list[E02TaskNode] = []
    for model, seed in prediction_models:
        prediction = add(
            "FORMAL_PREDICT",
            model,
            seed=seed,
            deps=(formal, frozen),
            output="E02_FORMAL_PREDICTION",
            executor="e02.predict_formal",
        )
        scores.append(
            add(
                "TRAJECTORY_SCORE",
                model,
                seed=seed,
                deps=(prediction, formal),
                output="E02_TRAJECTORY_SCORES",
                executor="e02.score_trajectories",
            )
        )
    bootstrap = add(
        "PAIRED_BOOTSTRAP",
        "ITRANSFORMER",
        deps=tuple(scores),
        output="E02_BOOTSTRAP_EVIDENCE",
        executor="e02.bootstrap",
    )
    decision = add(
        "E02_DECISION",
        "ITRANSFORMER",
        deps=(bootstrap, *tuple(scores)),
        output="E02_DECISION",
        executor="e02.decide",
    )
    add(
        "E02_RECEIPT",
        "ITRANSFORMER",
        deps=(decision, bootstrap),
        output="E02_RECEIPT",
        executor="e02.publish_receipt",
    )
    digest = canonical_json_hash(
        {"config": config.scientific_hash(), "nodes": [n.node_id for n in nodes]}
    )
    return E02Graph(f"e02-graph-{digest}", config.scientific_hash(), tuple(nodes))


def compile_e02_ready(graph: E02Graph, completed: Mapping[str, ArtifactRef]) -> TaskManifest:
    known = {node.node_id: node for node in graph.nodes}
    if set(completed) - set(known):
        raise ValueError("completed E02 artifacts contain an unknown task")
    tasks = tuple(
        TaskSpec(
            identity=node.identity,
            phase=node.phase,
            inputs=(*node.external_inputs, *(completed[dep] for dep in node.dependency_ids)),
            output_artifact_type=node.output_artifact_type,
            resource_request=node.resource_request,
        )
        for node in graph.nodes
        if node.node_id not in completed and all(dep in completed for dep in node.dependency_ids)
    )
    digest = canonical_json_hash(
        {"graph": graph.graph_id, "tasks": [t.model_dump(mode="json") for t in tasks]}
    )
    return TaskManifest(
        manifest_id=f"e02-ready-{digest}", tasks=tasks, completed_task_policy="NEVER_RERUN"
    )
