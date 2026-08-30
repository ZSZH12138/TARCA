from __future__ import annotations

from dataclasses import dataclass

from tarca.contracts import canonical_json_hash
from tarca.e01.v2_config import E01V2Config
from tarca.execution import ResourceRequest, ScientificIdentity, TaskSpec


@dataclass(frozen=True, slots=True)
class E01V2GraphNode:
    task: TaskSpec
    executor_key: str
    dependency_task_ids: tuple[str, ...]

    @property
    def phase(self) -> str:
        return self.task.phase


@dataclass(frozen=True, slots=True)
class E01V2Graph:
    graph_id: str
    scientific_config_hash: str
    nodes: tuple[E01V2GraphNode, ...]
    ready_nodes: tuple[E01V2GraphNode, ...]


def _task(
    config: E01V2Config,
    *,
    task_id: str,
    model_id: str,
    data_id: str,
    seed: int,
    phase: str,
    output_type: str,
    resources: ResourceRequest,
) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id=config.protocol_id,
            experiment_id=config.experiment_id,
            task_id=task_id,
            model_id=model_id,
            data_id=data_id,
            seed=seed,
        ),
        phase=phase,
        inputs=(),
        output_artifact_type=output_type,
        resource_request=resources,
    )


def compile_e01_v2_graph(
    config: E01V2Config,
    *,
    completed_task_ids: frozenset[str] = frozenset(),
) -> E01V2Graph:
    nodes: list[E01V2GraphNode] = []
    analysis_ids: list[str] = []
    world = config.worlds[0]
    for index, seed in enumerate(config.formal_seeds):
        generation_id = f"e01v2-a-generate-{index:03d}-{seed}"
        analysis_id = f"e01v2-a-analyze-{index:03d}-{seed}"
        generation = _task(
            config,
            task_id=generation_id,
            model_id="analytic-delayed-control-v2-generator",
            data_id=world.world_id,
            seed=seed,
            phase="E01_A_V2_GPU_GENERATE",
            output_type="E01_V2_EFFECT_BLOCK",
            resources=ResourceRequest(
                cpu_threads=1,
                gpu_count=1,
                gpu_memory_gib=4.0,
                host_memory_gib=2.0,
            ),
        )
        analysis = _task(
            config,
            task_id=analysis_id,
            model_id="e01-v2-calibrated-seed-analysis",
            data_id=world.world_id,
            seed=seed,
            phase="E01_A_V2_CPU_ANALYZE",
            output_type="E01_V2_SEED_REPORT",
            resources=ResourceRequest(
                cpu_threads=1,
                gpu_count=0,
                gpu_memory_gib=0.0,
                host_memory_gib=1.0,
            ),
        )
        nodes.extend(
            (
                E01V2GraphNode(generation, "e01.v2.generate", ()),
                E01V2GraphNode(analysis, "e01.v2.analyze", (generation_id,)),
            )
        )
        analysis_ids.append(analysis_id)
    aggregate = _task(
        config,
        task_id="e01v2-aggregate",
        model_id="e01-v2-final-aggregation",
        data_id=config.scientific_hash(),
        seed=0,
        phase="E01_V2_AGGREGATE",
        output_type="E01_V2_FINAL_REPORT",
        resources=ResourceRequest(
            cpu_threads=2,
            gpu_count=0,
            gpu_memory_gib=0.0,
            host_memory_gib=4.0,
        ),
    )
    nodes.append(E01V2GraphNode(aggregate, "e01.v2.aggregate", tuple(analysis_ids)))
    node_tuple = tuple(nodes)
    node_ids = {node.task.task_id for node in node_tuple}
    if set(completed_task_ids) - node_ids:
        raise ValueError("completed task IDs are outside the frozen E01-v2 graph")
    ready = tuple(
        node
        for node in node_tuple
        if node.task.task_id not in completed_task_ids
        and set(node.dependency_task_ids).issubset(completed_task_ids)
    )
    scientific_hash = config.scientific_hash()
    graph_hash = canonical_json_hash(
        {
            "scientific_config_hash": scientific_hash,
            "nodes": tuple(
                {
                    "task": node.task.model_dump(mode="json"),
                    "executor_key": node.executor_key,
                    "dependencies": node.dependency_task_ids,
                }
                for node in node_tuple
            ),
        }
    )
    return E01V2Graph(
        graph_id=f"e01v2-graph-{graph_hash}",
        scientific_config_hash=scientific_hash,
        nodes=node_tuple,
        ready_nodes=ready,
    )
