from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tarca.contracts import (
    PROTOCOL_ID,
    ArtifactRef,
    Sha256Hash,
    canonical_json_hash,
)
from tarca.execution import (
    ResourceRequest,
    ScientificIdentity,
    TaskManifest,
    TaskSpec,
)
from tarca.stage1b.config import (
    NeuralAdapter,
    QualificationConfig,
    WorldRole,
    WorldSuiteConfig,
    load_qualification_config,
    load_world_suite,
)
from tarca.stage1b.reproduction import ReproductionSuite, load_reproduction_suite


@dataclass(frozen=True, slots=True)
class Stage1BCompilationInputs:
    world_suite: WorldSuiteConfig
    qualification: QualificationConfig
    reproduction_suite: ReproductionSuite
    config_sha256: Sha256Hash
    code_sha256: Sha256Hash

    def __post_init__(self) -> None:
        for value in (self.config_sha256, self.code_sha256):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("compilation hashes must be lowercase SHA-256 values")


@dataclass(frozen=True, slots=True)
class Stage1BJobNode:
    node_id: str
    identity: ScientificIdentity
    phase: str
    dependency_ids: tuple[str, ...]
    expected_input_types: tuple[str, ...]
    output_artifact_type: str
    resource_request: ResourceRequest
    executor_key: str

    def __post_init__(self) -> None:
        if self.node_id != self.identity.task_id:
            raise ValueError("job node ID must equal its scientific task ID")
        if len(self.dependency_ids) != len(self.expected_input_types):
            raise ValueError("job dependencies must align with expected input types")
        if len(self.dependency_ids) != len(set(self.dependency_ids)):
            raise ValueError("job dependencies must be unique")
        if not self.phase.strip() or not self.output_artifact_type.strip():
            raise ValueError("job phase and output artifact type must not be blank")
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", self.executor_key) is None:
            raise ValueError("job executor key must be registry-shaped")


@dataclass(frozen=True, slots=True)
class Stage1BRunGraph:
    graph_id: str
    config_sha256: Sha256Hash
    code_sha256: Sha256Hash
    nodes: tuple[Stage1BJobNode, ...]

    def __post_init__(self) -> None:
        node_ids = tuple(node.node_id for node in self.nodes)
        if not self.nodes or len(node_ids) != len(set(node_ids)):
            raise ValueError("Stage1B graph nodes must be nonempty and unique")
        positions = {node_id: index for index, node_id in enumerate(node_ids)}
        for index, node in enumerate(self.nodes):
            unknown = tuple(
                dependency for dependency in node.dependency_ids if dependency not in positions
            )
            if unknown:
                raise ValueError("Stage1B graph contains an unknown dependency")
            if any(positions[dependency] >= index for dependency in node.dependency_ids):
                raise ValueError("Stage1B graph must be topologically ordered")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _paths_sha256(root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError("compilation input path must stay inside the repository") from error
        if not path.is_file():
            raise ValueError(f"compilation input file is missing: {relative.as_posix()}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def repository_v2_inputs(repository_root: Path | None = None) -> Stage1BCompilationInputs:
    root = (repository_root or _repository_root()).resolve()
    world_path = root / "configs/stage1b/worlds_v2.yaml"
    qualification_path = root / "configs/stage1b/qualification_v2.yaml"
    reproduction_path = root / "configs/stage1b/official_reproduction_v2.yaml"
    config_paths = (world_path, qualification_path, reproduction_path)
    scientific_roots = (
        root / "src/tarca/artifacts",
        root / "src/tarca/contracts",
        root / "src/tarca/stage1b",
    )
    code_paths = (
        *(path for scientific_root in scientific_roots for path in scientific_root.rglob("*.py")),
        root / "src/tarca/py.typed",
    )
    return Stage1BCompilationInputs(
        world_suite=load_world_suite(world_path),
        qualification=load_qualification_config(qualification_path),
        reproduction_suite=load_reproduction_suite(reproduction_path),
        config_sha256=_paths_sha256(root, config_paths),
        code_sha256=_paths_sha256(root, code_paths),
    )


def _resources(phase: str) -> ResourceRequest:
    values = {
        "SOURCE_MATERIALIZE": (2, 0, 0.0, 4.0),
        "OFFICIAL_REPRODUCTION": (2, 0, 0.0, 8.0),
        "WORLD_HEALTH": (4, 0, 0.0, 16.0),
        "DATA_GENERATE": (24, 0, 0.0, 96.0),
        "DATA_VALIDATE": (8, 0, 0.0, 32.0),
        "VAR_SCORE": (8, 0, 0.0, 32.0),
        "NEURAL_TRAIN": (4, 1, 20.0, 32.0),
        "MODEL_FREEZE_CHECK": (2, 1, 8.0, 16.0),
        "SCORE_BOOTSTRAP": (8, 0, 0.0, 48.0),
        "QUALIFICATION_AGGREGATE": (8, 0, 0.0, 32.0),
        "QUALIFICATION_RECEIPT": (2, 0, 0.0, 8.0),
    }
    try:
        cpu, gpu, gpu_memory, host_memory = values[phase]
    except KeyError as error:
        raise ValueError(f"unregistered Stage1B phase: {phase}") from error
    return ResourceRequest(
        cpu_threads=cpu,
        gpu_count=gpu,
        gpu_memory_gib=gpu_memory,
        host_memory_gib=host_memory,
    )


def compile_stage1b_graph(inputs: Stage1BCompilationInputs) -> Stage1BRunGraph:
    nodes: list[Stage1BJobNode] = []
    by_id: dict[str, Stage1BJobNode] = {}

    def add_node(
        *,
        phase: str,
        data_id: str,
        model_id: str = "model-none",
        seed: int = 0,
        dependencies: tuple[Stage1BJobNode, ...] = (),
        output_artifact_type: str,
        executor_key: str,
        discriminator: str = "",
    ) -> Stage1BJobNode:
        dependency_ids = tuple(dependency.node_id for dependency in dependencies)
        task_hash = canonical_json_hash(
            {
                "protocol_id": PROTOCOL_ID,
                "experiment_id": inputs.qualification.qualification_id,
                "phase": phase,
                "data_id": data_id,
                "model_id": model_id,
                "seed": seed,
                "discriminator": discriminator,
                "config_sha256": inputs.config_sha256,
                "code_sha256": inputs.code_sha256,
                "dependency_ids": list(dependency_ids),
                "output_artifact_type": output_artifact_type,
            }
        )
        node_id = f"stage1b-{phase.lower().replace('_', '-')}-{task_hash}"
        identity = ScientificIdentity(
            protocol_id=PROTOCOL_ID,
            experiment_id=inputs.qualification.qualification_id,
            task_id=node_id,
            model_id=model_id,
            data_id=data_id,
            seed=seed,
        )
        node = Stage1BJobNode(
            node_id=node_id,
            identity=identity,
            phase=phase,
            dependency_ids=dependency_ids,
            expected_input_types=tuple(
                dependency.output_artifact_type for dependency in dependencies
            ),
            output_artifact_type=output_artifact_type,
            resource_request=_resources(phase),
            executor_key=executor_key,
        )
        if node.node_id in by_id:
            raise ValueError("Stage1B compilation produced a duplicate scientific task")
        nodes.append(node)
        by_id[node.node_id] = node
        return node

    source_nodes = {
        source.source_id: add_node(
            phase="SOURCE_MATERIALIZE",
            data_id=source.source_id,
            output_artifact_type="OFFICIAL_SOURCE_RECEIPT",
            executor_key="stage1b.materialize_source",
        )
        for source in inputs.world_suite.sources
    }
    reproduction_nodes = {
        case.source_id: add_node(
            phase="OFFICIAL_REPRODUCTION",
            data_id=case.source_id,
            dependencies=(source_nodes[case.source_id],),
            output_artifact_type="OFFICIAL_REPRODUCTION_RECEIPT",
            executor_key="stage1b.reproduce_official_case",
            discriminator=case.case_id,
        )
        for case in inputs.reproduction_suite.cases
    }
    health_nodes: dict[str, Stage1BJobNode] = {}
    for world in inputs.world_suite.worlds:
        if world.role is WorldRole.REFERENCE_ONLY:
            continue
        source_ids = (world.source_id, *world.supporting_source_ids)
        dependencies = tuple(
            dependency
            for source_id in source_ids
            for dependency in (
                source_nodes[source_id],
                *((reproduction_nodes[source_id],) if source_id in reproduction_nodes else ()),
            )
        )
        health_nodes[world.world_id] = add_node(
            phase="WORLD_HEALTH",
            data_id=world.world_id,
            dependencies=dependencies,
            output_artifact_type="WORLD_HEALTH_RECEIPT",
            executor_key="stage1b.check_world_health",
        )

    primary_worlds = tuple(
        world for world in inputs.world_suite.worlds if world.role is WorldRole.PRIMARY_MECHANISTIC
    )
    score_nodes: list[Stage1BJobNode] = []
    for world in primary_worlds:
        for seed in inputs.qualification.qualification_seeds:
            generated = add_node(
                phase="DATA_GENERATE",
                data_id=world.world_id,
                seed=seed,
                dependencies=(health_nodes[world.world_id],),
                output_artifact_type="QUALIFICATION_DATASET",
                executor_key="stage1b.generate_dataset",
            )
            validated = add_node(
                phase="DATA_VALIDATE",
                data_id=world.world_id,
                seed=seed,
                dependencies=(generated,),
                output_artifact_type="VALIDATED_QUALIFICATION_DATASET",
                executor_key="stage1b.validate_dataset",
            )
            var_score = add_node(
                phase="VAR_SCORE",
                data_id=world.world_id,
                model_id="tuned_var",
                seed=seed,
                dependencies=(validated,),
                output_artifact_type="VAR_EVALUATION",
                executor_key="stage1b.score_var",
            )
            for model in inputs.qualification.models:
                model_source = {
                    NeuralAdapter.PATCHTST_REFERENCE: "patchtst",
                    NeuralAdapter.ITRANSFORMER_REFERENCE: "itransformer",
                }[model.adapter]
                trained = add_node(
                    phase="NEURAL_TRAIN",
                    data_id=world.world_id,
                    model_id=model.model_id,
                    seed=seed,
                    dependencies=(validated, reproduction_nodes[model_source]),
                    output_artifact_type="TRAINED_NEURAL_CHECKPOINT",
                    executor_key="stage1b.train_neural",
                )
                frozen = add_node(
                    phase="MODEL_FREEZE_CHECK",
                    data_id=world.world_id,
                    model_id=model.model_id,
                    seed=seed,
                    dependencies=(trained, validated),
                    output_artifact_type="FROZEN_MODEL_RECEIPT",
                    executor_key="stage1b.freeze_check_model",
                )
                score_nodes.append(
                    add_node(
                        phase="SCORE_BOOTSTRAP",
                        data_id=world.world_id,
                        model_id=model.model_id,
                        seed=seed,
                        dependencies=(validated, var_score, frozen),
                        output_artifact_type="QUALIFICATION_COMPARISON",
                        executor_key="stage1b.score_bootstrap",
                    )
                )

    aggregate = add_node(
        phase="QUALIFICATION_AGGREGATE",
        data_id=inputs.world_suite.suite_id,
        dependencies=(*tuple(health_nodes.values()), *tuple(score_nodes)),
        output_artifact_type="QUALIFICATION_GATE_EVIDENCE",
        executor_key="stage1b.aggregate_qualification",
    )
    add_node(
        phase="QUALIFICATION_RECEIPT",
        data_id=inputs.world_suite.suite_id,
        dependencies=(aggregate, *tuple(reproduction_nodes.values())),
        output_artifact_type="STAGE1B_QUALIFICATION_RECEIPT",
        executor_key="stage1b.publish_qualification_receipt",
    )
    graph_id = f"stage1b-graph-{canonical_json_hash({'nodes': [node.node_id for node in nodes]})}"
    return Stage1BRunGraph(
        graph_id=graph_id,
        config_sha256=inputs.config_sha256,
        code_sha256=inputs.code_sha256,
        nodes=tuple(nodes),
    )


def compile_ready_manifest(
    graph: Stage1BRunGraph,
    completed: Mapping[str, ArtifactRef],
) -> TaskManifest:
    nodes = {node.node_id: node for node in graph.nodes}
    unknown = tuple(sorted(set(completed) - set(nodes)))
    if unknown:
        raise ValueError(f"completed outputs contain unknown task IDs: {', '.join(unknown)}")
    for task_id, artifact in completed.items():
        node = nodes[task_id]
        if artifact.artifact_type != node.output_artifact_type:
            raise ValueError(f"completed task {task_id} artifact type does not match")
        if any(dependency not in completed for dependency in node.dependency_ids):
            raise ValueError(f"completed task {task_id} is missing dependency artifacts")

    ready: list[TaskSpec] = []
    for node in graph.nodes:
        if node.node_id in completed or any(
            dependency not in completed for dependency in node.dependency_ids
        ):
            continue
        inputs = tuple(completed[dependency] for dependency in node.dependency_ids)
        actual_types = tuple(artifact.artifact_type for artifact in inputs)
        if actual_types != node.expected_input_types:
            raise ValueError(f"ready task {node.node_id} dependency artifact types do not match")
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
        {
            "graph_id": graph.graph_id,
            "tasks": [task.model_dump(mode="json") for task in ready],
        }
    )
    manifest_id = f"stage1b-ready-{manifest_hash}"
    return TaskManifest(
        manifest_id=manifest_id,
        tasks=tuple(ready),
        completed_task_policy="NEVER_RERUN",
    )
