from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tarca.contracts import (
    ArtifactRef,
    DatasetSpec,
    DatasetWindowPartition,
    SealedAccessGrant,
    canonical_json_hash,
)


def create_e02_grant(
    authorization_ref: ArtifactRef,
    *,
    issued_at: datetime | None = None,
    lifetime: timedelta = timedelta(hours=24),
) -> SealedAccessGrant:
    issued = issued_at or datetime.now(UTC)
    if issued.tzinfo is None or issued.utcoffset() != timedelta(0):
        raise ValueError("E02 grant issue time must be timezone-aware UTC")
    if lifetime <= timedelta(0) or lifetime > timedelta(hours=24):
        raise ValueError("E02 grant lifetime must be within zero and 24 hours")
    grant_id = canonical_json_hash(
        {
            "authorization": authorization_ref.model_dump(mode="json"),
            "issued_at": issued.isoformat(),
            "expires_at": (issued + lifetime).isoformat(),
        }
    )
    return SealedAccessGrant(
        grant_id=f"e02-grant-{grant_id}",
        dataset=DatasetSpec(name="lorenz96_twoscale_v2", version="e02-v1"),
        scope_name="e02_predictor_validity_v1-formal",
        allowed_partitions=(
            DatasetWindowPartition.TEST_SEEN_REGIME,
            DatasetWindowPartition.TEST_UNSEEN_REGIME,
        ),
        authorization_ref=authorization_ref,
        issued_at=issued,
        expires_at=issued + lifetime,
    )
