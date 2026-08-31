from __future__ import annotations

import json
from pathlib import Path

import pytest

from tarca.stage2.freeze import (
    Stage2FreezeRejected,
    freeze_stage2_suite,
    verify_frozen_stage2_suite,
)
from tarca.stage2.manifest import compile_stage2_manifest
from tests.stage2.test_manifest import compilation_inputs


def test_complete_stage2_manifest_freezes_and_reloads(tmp_path: Path) -> None:
    manifest = compile_stage2_manifest(compilation_inputs())

    receipt = freeze_stage2_suite(tmp_path, manifest)

    assert receipt.status == "FROZEN"
    assert receipt.strongest_linear_model_id == "DLINEAR"
    assert receipt.formal_access_event_count == 0
    assert verify_frozen_stage2_suite(tmp_path) == receipt


def test_stage2_freeze_refuses_unapproved_overwrite(tmp_path: Path) -> None:
    manifest = compile_stage2_manifest(compilation_inputs())
    freeze_stage2_suite(tmp_path, manifest)

    with pytest.raises(Stage2FreezeRejected, match="already exists"):
        freeze_stage2_suite(tmp_path, manifest)


def test_stage2_freeze_rejects_runtime_failures(tmp_path: Path) -> None:
    inputs = compilation_inputs()
    failed = inputs.with_runtime_failures(("training-attempt-2",))

    with pytest.raises(Stage2FreezeRejected, match="runtime failures"):
        freeze_stage2_suite(tmp_path, compile_stage2_manifest(failed))


def test_frozen_stage2_tamper_is_detected(tmp_path: Path) -> None:
    freeze_stage2_suite(tmp_path, compile_stage2_manifest(compilation_inputs()))
    manifest_path = tmp_path / "frozen" / "v1" / "stage2_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["normalizer_sha256"] = "0" * 64
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Stage2FreezeRejected, match="manifest"):
        verify_frozen_stage2_suite(tmp_path)
