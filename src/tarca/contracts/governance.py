"""Governance contracts for authorization and sealed-access boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from .common import _require_hash, _require_text


@dataclass(frozen=True, slots=True)
class SealedAccessGrant:
    """Explicit grant required before any sealed materialization is attempted."""

    grant_id: str
    scope: str
    protocol_hash: str

    def __post_init__(self) -> None:
        _require_text(self.grant_id, "grant_id")
        _require_text(self.scope, "scope")
        _require_hash(self.protocol_hash, "protocol_hash")
