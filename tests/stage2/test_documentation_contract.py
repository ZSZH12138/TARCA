from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/research/stage2_e02_local_implementation_report_v1.md"
HANDOFF = ROOT / "docs/research/stage2_e02_server_handoff_v1.md"


def test_handoff_contains_exact_first_open_boundary() -> None:
    text = HANDOFF.read_text(encoding="utf-8")
    assert "bash deploy/stage2/bootstrap.sh --mode preflight" in text
    assert "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN" in text
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" in text
    assert "尚未执行完整 Stage 2/E02" in text
    assert "127.0.0.1:8765" in text
    assert "TARCA_24H_RESET" in text
    assert "TARCA_RESET_MARGIN" in text


def test_local_report_records_evidence_without_claiming_remote_success() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "Python 3.11.15" in text
    assert "2.13.0+cpu" in text
    assert "80.55%" in text
    assert "NOT_RUN_FULL_STAGE2_E02" in text
    assert "REMOTE_SERVER_NOT_CONNECTED" in text
    assert "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c" in text
