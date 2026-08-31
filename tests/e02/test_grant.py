from datetime import UTC, datetime, timedelta

from tarca.contracts import ArtifactRef, DatasetWindowPartition, canonical_json_hash
from tarca.e02.grant import create_e02_grant


def test_e02_grant_is_exactly_scoped_to_two_formal_partitions() -> None:
    issued = datetime(2026, 8, 31, tzinfo=UTC)
    authorization = ArtifactRef(
        artifact_id="authorization",
        artifact_type="SEALED_ACCESS_AUTHORIZATION",
        content_hash=canonical_json_hash("authorization"),
        schema_version="1.0.0",
        relative_path="artifacts/e02/authorization.json",
    )
    grant = create_e02_grant(authorization, issued_at=issued)
    assert grant.allowed_partitions == (
        DatasetWindowPartition.TEST_SEEN_REGIME,
        DatasetWindowPartition.TEST_UNSEEN_REGIME,
    )
    assert grant.expires_at == issued + timedelta(hours=24)
