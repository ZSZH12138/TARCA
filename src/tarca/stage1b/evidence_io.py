from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from tarca.contracts import canonical_json_bytes


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 identity of exact evidence bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a large evidence artifact into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(
    path: Path,
    value: Mapping[str, object],
    *,
    replace: bool,
) -> bytes:
    """Atomically write newline-terminated canonical JSON in the target directory."""

    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload
