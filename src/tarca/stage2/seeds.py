from __future__ import annotations

import hashlib


def derive_namespaced_seed(namespace: str) -> int:
    """Derive the frozen positive 31-bit seed for a UTF-8 namespace."""

    if not namespace.strip():
        raise ValueError("seed namespace must not be blank")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return 1 + value % 2_147_483_646


def validate_seed_isolation(
    *,
    development_seeds: tuple[int, ...],
    initialization_seeds: tuple[int, ...],
    excluded_seeds: tuple[int, ...],
) -> None:
    """Reject duplicates or overlap across development, initialization, and upstream seeds."""

    groups = {
        "development seeds": development_seeds,
        "initialization seeds": initialization_seeds,
        "excluded seeds": excluded_seeds,
    }
    for label, seeds in groups.items():
        if not seeds or any(type(seed) is not int or seed <= 0 for seed in seeds):
            raise ValueError(f"{label} must contain positive integer seeds")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{label} must be unique")
    development = set(development_seeds)
    initialization = set(initialization_seeds)
    excluded = set(excluded_seeds)
    if development & initialization or development & excluded or initialization & excluded:
        raise ValueError("seed groups must not overlap")
