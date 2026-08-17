"""Stable error taxonomy for architecture and execution boundaries."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    CONTRACT_ERROR = "CONTRACT_ERROR"
    DATA_ERROR = "DATA_ERROR"
    SCIENTIFIC_FAIL = "SCIENTIFIC_FAIL"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    AUTHORIZATION_BLOCKED = "AUTHORIZATION_BLOCKED"
    SEALED_ACCESS_VIOLATION = "SEALED_ACCESS_VIOLATION"
    ARCHITECTURE_VIOLATION = "ARCHITECTURE_VIOLATION"
    UNIMPLEMENTED_CAPABILITY = "UNIMPLEMENTED_CAPABILITY"


class TarcaArchitectureError(RuntimeError):
    """Base error carrying a stable machine-readable code."""

    code: ErrorCode

    def __init__(self, message: str, *, code: ErrorCode) -> None:
        super().__init__(message)
        self.code = code


class UnimplementedCapabilityError(TarcaArchitectureError):
    """Raised when a future capability is intentionally not implemented."""

    capability: str

    def __init__(self, capability: str) -> None:
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError("capability must be a non-empty string")
        self.capability = capability
        super().__init__(
            f"Capability is intentionally unimplemented: {capability}",
            code=ErrorCode.UNIMPLEMENTED_CAPABILITY,
        )


class AuthorizationBlockedError(TarcaArchitectureError):
    """Raised when sealed access is requested without an explicit grant."""

    def __init__(self, message: str = "Sealed access requires an explicit grant") -> None:
        super().__init__(message, code=ErrorCode.AUTHORIZATION_BLOCKED)
