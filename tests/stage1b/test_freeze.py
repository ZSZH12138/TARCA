from __future__ import annotations

import json
from pathlib import Path

import pytest

from tarca.stage1b.freeze import (
    FreezeRejected,
    OverrideAuthorization,
    freeze_suite,
    load_active_pointer,
    revision_root,
    verify_frozen_suite,
)

from .receipt_helpers import passing_receipt


def test_freeze_fails_when_suite_gate_does_not_pass(tmp_path: Path) -> None:
    receipt = passing_receipt()
    receipt["suite_decision"] = {
        "status": "FAIL",
        "passed_world_ids": [],
        "failed_world_ids": ["lorenz96_f10_v2"],
        "primary_families": [],
        "failed_checks": ["independent_primary_families"],
    }

    with pytest.raises(FreezeRejected, match="suite gate"):
        freeze_suite(receipt, tmp_path, series="v2", revision_id="v2-r1")

    assert not (tmp_path / "active.json").exists()


def test_authorized_override_keeps_v2_series(tmp_path: Path) -> None:
    first_receipt = passing_receipt()
    first = freeze_suite(first_receipt, tmp_path, series="v2", revision_id="v2-r1")
    second = freeze_suite(
        passing_receipt(),
        tmp_path,
        series="v2",
        revision_id="v2-r2",
        authorization=OverrideAuthorization(
            authorized_by="user",
            reason="approved v2 override",
            prior_revision_id="v2-r1",
        ),
    )

    assert (tmp_path / "versions/v2/revisions/r1/manifest.json").is_file()
    assert (tmp_path / "versions/v2/revisions/r2/manifest.json").is_file()
    assert load_active_pointer(tmp_path)["series"] == "v2"
    assert load_active_pointer(tmp_path)["revision_id"] == "v2-r2"
    assert first["revision_id"] == "v2-r1"
    assert second["revision_id"] == "v2-r2"
    stored_receipt = json.loads(
        (tmp_path / "versions/v2/revisions/r1/qualification_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored_receipt["comparisons"] == first_receipt["comparisons"]
    assert stored_receipt["failure_ledger"] == first_receipt["failure_ledger"]
    assert verify_frozen_suite(tmp_path)["status"] == "PASS"


def test_override_without_authorization_is_rejected(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path, series="v2", revision_id="v2-r1")

    with pytest.raises(FreezeRejected, match="authorization"):
        freeze_suite(passing_receipt(), tmp_path, series="v2", revision_id="v2-r2")


def test_freeze_rejects_partial_execution_evidence(tmp_path: Path) -> None:
    receipt = passing_receipt()
    evidence = dict(receipt["qualification_evidence"])
    evidence["completed_task_count"] = 73
    receipt["qualification_evidence"] = evidence

    with pytest.raises(FreezeRejected, match="partial qualification"):
        freeze_suite(receipt, tmp_path)


@pytest.mark.parametrize("drift_field", ["source_drift_detected", "identity_drift_detected"])
def test_freeze_rejects_source_or_identity_drift(tmp_path: Path, drift_field: str) -> None:
    receipt = passing_receipt()
    evidence = dict(receipt["qualification_evidence"])
    evidence[drift_field] = True
    receipt["qualification_evidence"] = evidence

    with pytest.raises(FreezeRejected, match="drift"):
        freeze_suite(receipt, tmp_path)


def test_freeze_rejects_reserved_formal_seed(tmp_path: Path) -> None:
    receipt = passing_receipt()
    receipt["qualification_seeds"] = [101, 201]

    with pytest.raises(FreezeRejected, match="reserved formal seed"):
        freeze_suite(receipt, tmp_path)


def test_frozen_manifest_detects_tampering(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path, series="v2", revision_id="v2-r1")
    manifest_path = tmp_path / "versions/v2/revisions/r1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualification_id"] = "tampered"
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FreezeRejected, match="manifest hash"):
        verify_frozen_suite(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_evidence_verified", False, "source evidence"),
        ("source_manifest_sha256", "bad", "source_manifest_sha256"),
        ("source_commits", {}, "source commits"),
        ("source_commits", {"source": "bad"}, "source commits"),
        ("world_decisions", None, "world_decisions"),
        ("failure_ledger", None, "failure_ledger"),
    ],
)
def test_freeze_rejects_incomplete_receipt_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    receipt = passing_receipt()
    receipt[field] = value

    with pytest.raises(FreezeRejected, match=message):
        freeze_suite(receipt, tmp_path)


def test_freeze_rejects_hardware_identity_and_undeclared_row_seed(tmp_path: Path) -> None:
    identity_drift = passing_receipt()
    evidence = dict(identity_drift["qualification_evidence"])
    evidence["hardware_receipt_sha256"] = "f" * 64
    identity_drift["qualification_evidence"] = evidence
    with pytest.raises(FreezeRejected, match="hardware receipt identity drift"):
        freeze_suite(identity_drift, tmp_path)

    undeclared_seed = passing_receipt()
    undeclared_seed["training_receipts"] = [{"qualification_seed": 999}]
    with pytest.raises(FreezeRejected, match="undeclared"):
        freeze_suite(undeclared_seed, tmp_path)


def test_freeze_rejects_formal_identifier_and_bad_seed_declarations(tmp_path: Path) -> None:
    formal = passing_receipt("E01")
    with pytest.raises(FreezeRejected, match="formal experiment"):
        freeze_suite(formal, tmp_path)

    duplicate = passing_receipt()
    duplicate["qualification_seeds"] = [101, 101]
    with pytest.raises(FreezeRejected, match="duplicates"):
        freeze_suite(duplicate, tmp_path)

    malformed = passing_receipt()
    malformed["reserved_formal_seeds"] = [True]
    with pytest.raises(FreezeRejected, match="invalid"):
        freeze_suite(malformed, tmp_path)


def test_revision_identity_and_initial_override_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v2-rN"):
        revision_root(tmp_path, "v2")
    with pytest.raises(ValueError, match="must not be blank"):
        OverrideAuthorization(" ", "reason", "v2-r1")
    with pytest.raises(ValueError, match="v2-rN"):
        OverrideAuthorization("user", "reason", "v2")
    with pytest.raises(FreezeRejected, match="fixed to v2"):
        freeze_suite(passing_receipt(), tmp_path, series="v3", revision_id="v3-r1")
    with pytest.raises(FreezeRejected, match="v2-rN"):
        freeze_suite(passing_receipt(), tmp_path, revision_id="v2")
    with pytest.raises(FreezeRejected, match="initial v2 freeze"):
        freeze_suite(passing_receipt(), tmp_path, revision_id="v2-r2")
    with pytest.raises(FreezeRejected, match="initial freeze"):
        freeze_suite(
            passing_receipt(),
            tmp_path,
            authorization=OverrideAuthorization("user", "reason", "v2-r1"),
        )


def test_override_must_match_active_and_create_exact_next_revision(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path)
    with pytest.raises(FreezeRejected, match="cannot be overwritten"):
        freeze_suite(passing_receipt(), tmp_path)
    with pytest.raises(FreezeRejected, match="does not match"):
        freeze_suite(
            passing_receipt(),
            tmp_path,
            revision_id="v2-r2",
            authorization=OverrideAuthorization("user", "reason", "v2-r9"),
        )
    with pytest.raises(FreezeRejected, match="next v2 revision"):
        freeze_suite(
            passing_receipt(),
            tmp_path,
            revision_id="v2-r3",
            authorization=OverrideAuthorization("user", "reason", "v2-r1"),
        )


def test_verify_reports_unfrozen_missing_revision_and_receipt_tampering(
    tmp_path: Path,
) -> None:
    assert verify_frozen_suite(tmp_path, allow_unfrozen=True) == {
        "status": "UNFROZEN",
        "active_revision_id": None,
    }
    with pytest.raises(FreezeRejected, match="not frozen"):
        verify_frozen_suite(tmp_path)
    freeze_suite(passing_receipt(), tmp_path)
    with pytest.raises(FreezeRejected, match="scientific series"):
        verify_frozen_suite(tmp_path, series="v3")
    with pytest.raises(FreezeRejected, match="files are missing"):
        verify_frozen_suite(tmp_path, revision_id="v2-r2")

    receipt_path = tmp_path / "versions/v2/revisions/r1/qualification_receipt.json"
    receipt_path.chmod(0o644)
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FreezeRejected, match="qualification receipt hash"):
        verify_frozen_suite(tmp_path)
