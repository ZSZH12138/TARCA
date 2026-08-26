from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

import torch

from tarca.stage1b.config import WorldAdapter, WorldConfig

STATE_DEPENDENT_SIGN = 2


@dataclass(frozen=True, slots=True)
class WorldTruth:
    world_id: str
    adjacency: torch.Tensor
    signed_adjacency: torch.Tensor
    shortest_path_lags: tuple[tuple[int, ...], ...]
    graph_sha256: str
    concepts: tuple[str, ...]
    latent_dimension: int


def ring_adjacency(dimension: int, *, directed: bool, include_self: bool = False) -> torch.Tensor:
    adjacency = torch.zeros((dimension, dimension), dtype=torch.int8)
    for source in range(dimension):
        adjacency[(source + 1) % dimension, source] = 1
        if not directed:
            adjacency[(source - 1) % dimension, source] = 1
        if include_self:
            adjacency[source, source] = 1
    return adjacency


def lorenz96_adjacency(dimension: int) -> torch.Tensor:
    adjacency = torch.zeros((dimension, dimension), dtype=torch.int8)
    for target in range(dimension):
        for source in (
            target,
            (target + 1) % dimension,
            (target - 1) % dimension,
            (target - 2) % dimension,
        ):
            adjacency[target, source] = 1
    return adjacency


def predator_prey_adjacency(
    dimension: int, parents_per_node: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if dimension % 2:
        raise ValueError("predator-prey dimension must be even")
    species = dimension // 2
    if parents_per_node <= 0 or species % parents_per_node:
        raise ValueError("predator-prey groups must divide the species count")
    adjacency = torch.zeros((dimension, dimension), dtype=torch.int8)
    signed = torch.zeros_like(adjacency)
    for index in range(species):
        start = (index // parents_per_node) * parents_per_node
        prey_target = index
        predator_target = species + index
        adjacency[prey_target, prey_target] = 1
        adjacency[predator_target, predator_target] = 1
        signed[prey_target, prey_target] = 1
        signed[predator_target, predator_target] = -1
        for peer in range(start, start + parents_per_node):
            adjacency[prey_target, species + peer] = 1
            adjacency[predator_target, peer] = 1
            signed[prey_target, species + peer] = -1
            signed[predator_target, peer] = 1
    return adjacency, signed


def shortest_path_lags(adjacency: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    rows: list[tuple[int, ...]] = []
    for source in range(adjacency.shape[0]):
        distances = [-1] * adjacency.shape[0]
        distances[source] = 0
        pending: deque[int] = deque([source])
        while pending:
            current = pending.popleft()
            for target in torch.where(adjacency[:, current] != 0)[0].tolist():
                if distances[target] < 0:
                    distances[target] = distances[current] + 1
                    pending.append(target)
        rows.append(tuple(distances))
    return tuple(rows)


def build_world_truth(config: WorldConfig) -> WorldTruth:
    adapter = config.adapter
    if adapter in {WorldAdapter.LORENZ96, WorldAdapter.LORENZ96_TWO_SCALE}:
        adjacency = lorenz96_adjacency(config.dimension)
        signed = torch.full_like(adjacency, STATE_DEPENDENT_SIGN) * adjacency
        signed.fill_diagonal_(-1)
    elif adapter is WorldAdapter.GVAR_PREDATOR_PREY:
        parents = int(config.generator_map()["parents_per_node"])
        adjacency, signed = predator_prey_adjacency(config.dimension, parents)
    else:
        adjacency = ring_adjacency(
            config.dimension,
            directed=config.graph.directed,
            include_self=True,
        )
        signed = torch.full_like(adjacency, STATE_DEPENDENT_SIGN) * adjacency
    graph_bytes = b"".join(
        (
            adjacency.contiguous().numpy().tobytes(),
            signed.contiguous().numpy().tobytes(),
            config.latent_dimension.to_bytes(8, "little", signed=False),
        )
    )
    return WorldTruth(
        world_id=config.world_id,
        adjacency=adjacency,
        signed_adjacency=signed,
        shortest_path_lags=shortest_path_lags(adjacency),
        graph_sha256=hashlib.sha256(graph_bytes).hexdigest(),
        concepts=config.concepts,
        latent_dimension=config.latent_dimension,
    )
