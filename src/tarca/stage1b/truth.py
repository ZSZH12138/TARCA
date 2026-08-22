from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WorldTruth:
    world_id: str
    adjacency: torch.Tensor
    shortest_path_lags: tuple[tuple[int, ...], ...]
    graph_sha256: str
    concepts: tuple[str, ...]


def ring_adjacency(dimension: int, *, directed: bool) -> torch.Tensor:
    """Return target-by-source adjacency for a directed or undirected ring."""
    adjacency = torch.zeros((dimension, dimension), dtype=torch.int8)
    for source in range(dimension):
        adjacency[(source + 1) % dimension, source] = 1
        if not directed:
            adjacency[(source - 1) % dimension, source] = 1
    return adjacency


def shortest_path_lags(adjacency: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    dimension = adjacency.shape[0]
    rows: list[tuple[int, ...]] = []
    for source in range(dimension):
        distances = [-1] * dimension
        distances[source] = 0
        pending: deque[int] = deque([source])
        while pending:
            current = pending.popleft()
            targets = torch.where(adjacency[:, current] != 0)[0].tolist()
            for target in targets:
                if distances[target] >= 0:
                    continue
                distances[target] = distances[current] + 1
                pending.append(target)
        rows.append(tuple(distances))
    return tuple(rows)


def build_world_truth(
    world_id: str,
    dimension: int,
    directed: bool,
    concepts: tuple[str, ...],
) -> WorldTruth:
    adjacency = ring_adjacency(dimension, directed=directed)
    graph_bytes = adjacency.contiguous().numpy().tobytes()
    return WorldTruth(
        world_id=world_id,
        adjacency=adjacency,
        shortest_path_lags=shortest_path_lags(adjacency),
        graph_sha256=hashlib.sha256(graph_bytes).hexdigest(),
        concepts=concepts,
    )
