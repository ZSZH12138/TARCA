"""Authorization skeleton with fail-closed sealed boundary."""

from __future__ import annotations

from tarca.contracts.errors import AuthorizationBlockedError, UnimplementedCapabilityError
from tarca.contracts.governance import SealedAccessGrant


def require_sealed_grant(grant: SealedAccessGrant | None) -> SealedAccessGrant:
    if grant is None:
        raise AuthorizationBlockedError()
    if not isinstance(grant, SealedAccessGrant):
        raise TypeError("grant must be SealedAccessGrant or None")
    return grant


def materialize_sealed(grant: SealedAccessGrant | None) -> None:
    require_sealed_grant(grant)
    raise UnimplementedCapabilityError("governance.materialize_sealed")


def verify_gate(decision: object) -> None:
    raise UnimplementedCapabilityError("governance.verify_gate")
