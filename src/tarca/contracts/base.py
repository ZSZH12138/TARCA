from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

CONTRACT_SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "TARCA-E2E-STAGE-PROTOCOL-2.0"

Sha256Hash: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
GitCommit: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$"),
]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be timezone-aware UTC")
    return value


UtcDatetime: TypeAlias = Annotated[datetime, AfterValidator(_require_utc)]


class StrictContractModel(BaseModel):
    """Common configuration for every persisted Stage contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def canonical_json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> Sha256Hash:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> Sha256Hash:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> Sha256Hash:
    return sha256_bytes(canonical_json_bytes(value))
