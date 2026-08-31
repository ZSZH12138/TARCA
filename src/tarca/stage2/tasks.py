from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tarca.contracts import ArtifactRef, Sha256Hash, canonical_json_hash
from tarca.execution import ResourceRequest, ScientificIdentity, TaskManifest, TaskSpec
from tarca.stage2.config import Stage2Config


@dataclass(frozen=True, slots=True)
class Stage2GraphInputs:
    stage1b_manifest: ArtifactRef
    e01_receipt: ArtifactRef
    source_capsule: ArtifactRef
    formal_access_event_count: int

    def __post_init__(self) -> None:
        if self.formal_access_event_count != 0:
            raise PermissionError("Stage 2 compilation forbids formal-data access")


@dataclass(frozen=True, slots=True)
class Stage2TaskNode:
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
class Stage2Graph:
    graph_id: str
    config_sha256: Sha256Hash
    nodes: tuple[Stage2TaskNode, ...]


def _resource(phase: str) -> ResourceRequest:
    values = {
        "SOURCE_VERIFY": (2, 0, 0.0, 4.0),
        "UPSTREAM_VERIFY": (2, 0, 0.0, 4.0),
        "DEV_DATA": (16, 0, 0.0, 96.0),
        "BASELINE_FIT": (4, 0, 0.0, 24.0),
        "NEURAL_TRAIN": (4, 1, 20.0, 32.0),
        "CHECKPOINT_VALIDATE": (2, 1, 8.0, 16.0),
        "VALIDATION_PREDICT": (4, 0, 0.0, 24.0),
        "MODEL_SELECT": (2, 0, 0.0, 8.0),
        "FREEZE_CANDIDATE": (4, 0, 0.0, 24.0),
        "STAGE2_RECEIPT": (2, 0, 0.0, 8.0),
    }
    cpu, gpu, vram, ram = values[phase]
    return ResourceRequest(
        cpu_threads=cpu,
        gpu_count=gpu,
        gpu_memory_gib=vram,
        host_memory_gib=ram,
    )


def compile_stage2_graph(config: Stage2Config, inputs: Stage2GraphInputs) -> Stage2Graph:
    if inputs.formal_access_event_count != 0:
        raise PermissionError("Stage 2 compilation forbids formal-data access")
    nodes: list[Stage2TaskNode] = []

    def add(
        phase: str,
        model_id: str,
        data_id: str,
        *,
        seed: int = 0,
        deps: tuple[Stage2TaskNode, ...] = (),
        external: tuple[ArtifactRef, ...] = (),
        output: str,
        executor: str,
        tag: str = "",
    ) -> Stage2TaskNode:
        digest = canonical_json_hash(
            {
                "experiment": config.experiment_id,
                "config": config.scientific_hash(),
                "phase": phase,
                "model": model_id,
                "data": data_id,
                "seed": seed,
                "tag": tag,
                "dependencies": [item.node_id for item in deps],
                "external": [item.model_dump(mode="json") for item in external],
                "output": output,
            }
        )
        identity = ScientificIdentity(
            protocol_id=config.protocol_id,
            experiment_id=config.experiment_id,
            task_id=f"stage2-{phase.lower().replace('_', '-')}-{digest}",
            model_id=model_id,
            data_id=data_id,
            seed=seed,
        )
        node = Stage2TaskNode(
            identity=identity,
            phase=phase,
            dependency_ids=tuple(item.node_id for item in deps),
            external_inputs=external,
            output_artifact_type=output,
            resource_request=_resource(phase),
            executor_key=executor,
        )
        nodes.append(node)
        return node

    stage1b = add(
        "UPSTREAM_VERIFY",
        "STAGE1B",
        config.upstream.world_id,
        external=(inputs.stage1b_manifest,),
        output="VERIFIED_STAGE1B_HANDOFF",
        executor="stage2.verify_upstream",
        tag="stage1b",
    )
    e01 = add(
        "UPSTREAM_VERIFY",
        "E01",
        config.upstream.world_id,
        external=(inputs.e01_receipt,),
        output="VERIFIED_E01_HANDOFF",
        executor="stage2.verify_upstream",
        tag="e01",
    )
    sources = {
        source.source_id: add(
            "SOURCE_VERIFY",
            source.source_id.upper(),
            source.source_id,
            external=(inputs.source_capsule,),
            output="VERIFIED_STAGE2_SOURCE",
            executor="stage2.verify_source",
            tag=source.commit,
        )
        for source in config.sources
    }
    data = add(
        "DEV_DATA",
        "DATA",
        config.upstream.world_id,
        deps=(stage1b, e01, sources["scoring_rules_l96"]),
        output="STAGE2_DEVELOPMENT_DATA",
        executor="stage2.generate_development_data",
    )
    predictor: dict[tuple[str, int], Stage2TaskNode] = {}
    for model_id in ("LAST_VALUE", "SEASONAL_NAIVE", "VAR", "DLINEAR"):
        extra = (sources["dlinear"],) if model_id == "DLINEAR" else ()
        predictor[(model_id, 0)] = add(
            "BASELINE_FIT",
            model_id,
            config.upstream.world_id,
            deps=(data, *extra),
            output="STAGE2_PREDICTOR",
            executor="stage2.fit_baseline",
        )
    checkpoints: list[Stage2TaskNode] = []
    for model_id, source_id in (("PATCHTST", "patchtst"), ("ITRANSFORMER", "itransformer")):
        for index, seed in enumerate(config.training.initialization_seeds):
            trained = add(
                "NEURAL_TRAIN",
                model_id,
                config.upstream.world_id,
                seed=seed,
                deps=(data, sources[source_id]),
                output="STAGE2_NEURAL_CHECKPOINT",
                executor="stage2.train_neural",
                tag=str(index),
            )
            checked = add(
                "CHECKPOINT_VALIDATE",
                model_id,
                config.upstream.world_id,
                seed=seed,
                deps=(trained, data),
                output="VALIDATED_STAGE2_CHECKPOINT",
                executor="stage2.validate_checkpoint",
                tag=str(index),
            )
            checkpoints.append(checked)
            predictor[(model_id, seed)] = checked
    predictions: dict[tuple[str, int], Stage2TaskNode] = {}
    for (model_id, seed), fitted in predictor.items():
        predictions[(model_id, seed)] = add(
            "VALIDATION_PREDICT",
            model_id,
            config.upstream.world_id,
            seed=seed,
            deps=(fitted, data),
            output="STAGE2_VALIDATION_PREDICTION",
            executor="stage2.predict_validation",
        )
    linear = add(
        "MODEL_SELECT",
        "STRONGEST_LINEAR",
        config.upstream.world_id,
        deps=(predictions[("VAR", 0)], predictions[("DLINEAR", 0)]),
        output="STAGE2_MODEL_SELECTION",
        executor="stage2.select_model",
        tag="linear",
    )
    primary = add(
        "MODEL_SELECT",
        "ITRANSFORMER",
        config.upstream.world_id,
        deps=tuple(
            predictions[("ITRANSFORMER", seed)] for seed in config.training.initialization_seeds
        ),
        output="STAGE2_MODEL_SELECTION",
        executor="stage2.select_model",
        tag="itransformer",
    )
    frozen = add(
        "FREEZE_CANDIDATE",
        "SUITE",
        config.upstream.world_id,
        deps=(
            *tuple(predictions.values()),
            *tuple(checkpoints),
            linear,
            primary,
            *tuple(sources.values()),
        ),
        output="STAGE2_FREEZE_CANDIDATE",
        executor="stage2.freeze_candidate",
    )
    add(
        "STAGE2_RECEIPT",
        "SUITE",
        config.upstream.world_id,
        deps=(frozen,),
        output="STAGE2_FREEZE_RECEIPT",
        executor="stage2.publish_receipt",
    )
    graph_hash = canonical_json_hash(
        {"config": config.scientific_hash(), "nodes": [n.node_id for n in nodes]}
    )
    return Stage2Graph(f"stage2-graph-{graph_hash}", config.scientific_hash(), tuple(nodes))


def compile_stage2_ready(graph: Stage2Graph, completed: Mapping[str, ArtifactRef]) -> TaskManifest:
    known = {node.node_id: node for node in graph.nodes}
    if set(completed) - set(known):
        raise ValueError("completed Stage 2 artifacts contain an unknown task")
    ready: list[TaskSpec] = []
    for node in graph.nodes:
        if node.node_id in completed or any(dep not in completed for dep in node.dependency_ids):
            continue
        dependency_inputs = tuple(completed[dep] for dep in node.dependency_ids)
        inputs = (*node.external_inputs, *dependency_inputs)
        ready.append(
            TaskSpec(
                identity=node.identity,
                phase=node.phase,
                inputs=inputs,
                output_artifact_type=node.output_artifact_type,
                resource_request=node.resource_request,
            )
        )
    manifest_hash = canonical_json_hash(
        {"graph": graph.graph_id, "tasks": [t.model_dump(mode="json") for t in ready]}
    )
    return TaskManifest(
        manifest_id=f"stage2-ready-{manifest_hash}",
        tasks=tuple(ready),
        completed_task_policy="NEVER_RERUN",
    )
