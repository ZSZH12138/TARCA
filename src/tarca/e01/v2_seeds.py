from __future__ import annotations

import hashlib


def derive_v2_seeds(
    namespace: str,
    count: int,
    excluded: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if not namespace.strip():
        raise ValueError("seed namespace must be nonblank")
    if type(count) is not int or count <= 0:
        raise ValueError("seed count must be a positive integer")
    if any(type(seed) is not int or seed <= 0 or seed >= 2**31 for seed in excluded):
        raise ValueError("excluded seeds must be positive 31-bit integers")
    blocked = frozenset(excluded)
    seeds: list[int] = []
    for index in range(count):
        label = f"{namespace}/{index:04d}".encode()
        candidate = int.from_bytes(hashlib.sha256(label).digest()[:4], "big") & 0x7FFF_FFFF
        if candidate == 0 or candidate in blocked or candidate in seeds:
            raise ValueError("SHA-256 seed derivation produced a blocked or duplicate seed")
        seeds.append(candidate)
    return tuple(seeds)
