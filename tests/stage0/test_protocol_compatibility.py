from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from tarca.contracts import (
    AccessScope,
    ArtifactRef,
    DatasetSpec,
    DatasetWindowPartition,
    SealedAccessGrant,
    Sha256Hash,
    validate_sealed_access,
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
DATASET = DatasetSpec(name="synthetic_easy", version="1.0.0")


def _authorization_ref(
    artifact_type: str = "SEALED_ACCESS_AUTHORIZATION",
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id="sealed-access-authorization-1",
        artifact_type=artifact_type,
        content_hash="a" * 64,
        schema_version="1.0.0",
        relative_path=None,
    )


def _grant(**overrides: object) -> SealedAccessGrant:
    values: dict[str, object] = {
        "grant_id": "grant-1",
        "dataset": DATASET,
        "scope_name": "stage1a-test-read",
        "allowed_partitions": (DatasetWindowPartition.TEST,),
        "authorization_ref": _authorization_ref(),
        "issued_at": NOW - timedelta(minutes=5),
        "expires_at": NOW + timedelta(minutes=5),
    }
    return SealedAccessGrant(**(values | overrides))


def test_sha256_wire_format_remains_raw_lowercase_hex() -> None:
    adapter = TypeAdapter(Sha256Hash)

    assert adapter.validate_python("a" * 64) == "a" * 64
    with pytest.raises(ValidationError):
        adapter.validate_python("sha256:" + ("a" * 64))


@pytest.mark.parametrize("value", ["../weather", "weather/data", r"C:\weather", ".", ".."])
def test_dataset_spec_rejects_path_semantics(value: str) -> None:
    with pytest.raises(ValidationError):
        DatasetSpec(name=value, version="1.0.0")


def test_sealed_access_grant_is_strict_and_frozen() -> None:
    grant = _grant()

    with pytest.raises(ValidationError):
        grant.scope_name = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SealedAccessGrant.model_validate({**grant.model_dump(), "unexpected": True})


def test_sealed_access_grant_rejects_ambiguous_authorization() -> None:
    with pytest.raises(ValidationError):
        _grant(
            allowed_partitions=(
                DatasetWindowPartition.TEST,
                DatasetWindowPartition.TEST,
            )
        )
    with pytest.raises(ValidationError):
        _grant(expires_at=NOW - timedelta(minutes=5))
    with pytest.raises(ValidationError):
        _grant(authorization_ref=_authorization_ref("OTHER"))


def test_sealed_access_fails_closed_without_an_exact_current_grant() -> None:
    access = AccessScope(sealed=True, scope_name="stage1a-test-read")

    with pytest.raises(PermissionError):
        validate_sealed_access(DATASET, DatasetWindowPartition.TEST, access, None, NOW)
    with pytest.raises(PermissionError):
        validate_sealed_access(
            DATASET,
            DatasetWindowPartition.TEST,
            access,
            _grant(dataset=DatasetSpec(name="other", version="1.0.0")),
            NOW,
        )
    with pytest.raises(PermissionError):
        validate_sealed_access(
            DATASET,
            DatasetWindowPartition.TEST,
            access,
            _grant(scope_name="other-scope"),
            NOW,
        )
    with pytest.raises(PermissionError):
        validate_sealed_access(
            DATASET,
            DatasetWindowPartition.VALIDATION,
            access,
            _grant(),
            NOW,
        )
    with pytest.raises(PermissionError):
        validate_sealed_access(
            DATASET,
            DatasetWindowPartition.TEST,
            access,
            _grant(),
            NOW + timedelta(hours=1),
        )


def test_sealed_access_accepts_an_exact_current_grant() -> None:
    validate_sealed_access(
        DATASET,
        DatasetWindowPartition.TEST,
        AccessScope(sealed=True, scope_name="stage1a-test-read"),
        _grant(),
        NOW,
    )


def test_unsealed_access_does_not_require_a_grant() -> None:
    validate_sealed_access(
        DATASET,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="public-train-read"),
        None,
        NOW,
    )


def test_protocol_documents_the_approved_compatibility_repair() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = (
        repo_root / "docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"
    ).read_text(encoding="utf-8")
    ccp = (repo_root / "docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0001.md").read_text(
        encoding="utf-8"
    )

    assert "**协议版本**" in protocol and "v2.0.1" in protocol
    assert "**稳定协议身份**" in protocol
    assert "`TARCA-E2E-STAGE-PROTOCOL-2.0`" in protocol
    assert "64 lowercase hex" in protocol
    assert "`sha256:` + 64 lowercase hex" not in protocol
    assert "class SealedAccessGrant(StrictContractModel):" in protocol
    assert protocol.count("grant: SealedAccessGrant | None = None") >= 2
    assert "CCP-0001" in protocol
    assert "`APPROVED`" in ccp
    assert "不重新生成 Stage 0 artifact" in ccp
