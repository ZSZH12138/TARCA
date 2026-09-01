import json
import subprocess
from hashlib import sha256
from pathlib import Path

from tarca.stage2.freeze import Stage2FreezeReceipt
from tarca.stage2.manifest import stage2_manifest_from_payload

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/research/stage2_e02_local_implementation_report_v1.md"
HANDOFF = ROOT / "docs/research/stage2_e02_server_handoff_v1.md"
FREEZE_ROOT = ROOT / "artifacts/stage2/frozen/v1"


def test_stage2_frozen_evidence_is_small_valid_and_publishable() -> None:
    receipt_path = FREEZE_ROOT / "stage2_freeze_receipt.json"
    manifest_path = FREEZE_ROOT / "stage2_manifest.json"
    receipt_bytes = receipt_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()

    assert len(receipt_bytes) < 10_000
    assert len(manifest_bytes) < 20_000
    assert sha256(receipt_bytes).hexdigest() == (
        "5ec77ab844ef0bc793bf8543db57f01856ab603718a21fd1a19c42bf0947d8e5"
    )
    assert sha256(manifest_bytes).hexdigest() == (
        "6d9ef496956a714e956c57800f0c1cf479a042624f757f54f4882a99f8d132d4"
    )
    receipt = Stage2FreezeReceipt.model_validate_json(receipt_bytes)
    manifest = stage2_manifest_from_payload(json.loads(manifest_bytes))
    assert receipt.status == "FROZEN"
    assert receipt.formal_access_event_count == 0
    assert receipt.scientific_sha256 == manifest.scientific_sha256


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_stage2_git_boundary_exposes_only_small_frozen_evidence() -> None:
    assert _is_ignored(
        "artifacts/stage2/server-results/final/tarca-stage2-v1-complete.tar.gz"
    )
    assert not _is_ignored(
        "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json"
    )
    assert not _is_ignored("artifacts/stage2/frozen/v1/stage2_manifest.json")


def test_handoff_contains_exact_first_open_boundary() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "bash deploy/stage2/recovery_bootstrap.sh" in text
    assert "tarca-stage2-recovery-20260831T102151Z.tar.gz" in text
    assert "I_ACKNOWLEDGE_STAGE2_DEVICE_MISMATCH_RECOVERY_V1" in (
        ROOT / "deploy/stage2/recovery_bootstrap.sh"
    ).read_text(encoding="utf-8")
    assert "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN" in text
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" in text
    assert "LOCAL_RECOVERY_KIT_READY" in text
    assert "禁止使用 `launch`" in text
    assert "127.0.0.1:8765" in text
    assert "TARCA_24H_RESET" in text
    assert "TARCA_RESET_MARGIN" in text


def test_local_report_records_evidence_without_claiming_remote_success() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "Python 3.11.15" in text
    assert "2.13.0+cpu" in text
    assert "PREVIOUS_REMOTE_RUN_RECOVERED_LOCALLY" in text
    assert "NOT_RUN_E02_FORMAL" in text
    assert "LOCAL_RECOVERY_IMPLEMENTATION_COMPLETE" in text
    assert "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c" in text
