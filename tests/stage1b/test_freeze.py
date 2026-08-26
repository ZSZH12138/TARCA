from __future__ import annotations

import json
from pathlib import Path

import pytest

from tarca.stage1b.freeze import (
    FreezeRejected,
    OverrideAuthorization,
    freeze_suite,
    load_active_pointer,
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
        freeze_suite(receipt, tmp_path, version="v2")

    assert not (tmp_path / "active.json").exists()


def test_authorized_override_keeps_v2_and_moves_active_pointer(tmp_path: Path) -> None:
    first = freeze_suite(passing_receipt(), tmp_path, version="v2")
    second = freeze_suite(
        passing_receipt("stage1b-qualification-v3"),
        tmp_path,
        version="v3",
        authorization=OverrideAuthorization(
            authorized_by="user",
            reason="User approved a new world-suite version",
            prior_version="v2",
        ),
    )

    assert (tmp_path / "versions/v2/manifest.json").is_file()
    assert (tmp_path / "versions/v3/manifest.json").is_file()
    assert load_active_pointer(tmp_path)["version"] == "v3"
    assert first["version"] == "v2"
    assert second["version"] == "v3"
    assert verify_frozen_suite(tmp_path)["status"] == "PASS"


def test_override_without_authorization_is_rejected(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path, version="v2")

    with pytest.raises(FreezeRejected, match="authorization"):
        freeze_suite(passing_receipt("stage1b-qualification-v3"), tmp_path, version="v3")


def test_frozen_manifest_detects_tampering(tmp_path: Path) -> None:
    freeze_suite(passing_receipt(), tmp_path, version="v2")
    manifest_path = tmp_path / "versions/v2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualification_id"] = "tampered"
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FreezeRejected, match="manifest hash"):
        verify_frozen_suite(tmp_path)
