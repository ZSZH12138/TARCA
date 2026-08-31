from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/stage2"


def test_bootstrap_stops_before_training_and_checks_two_cards() -> None:
    text = (DEPLOY / "bootstrap.sh").read_text(encoding="utf-8")
    assert "torch.cuda.device_count() == 2" in text
    assert "source_hashes_verified" in text
    assert "checkpoint_roundtrip_passed" in text
    assert "PREFLIGHT_PASS: no training or formal task was started" in text
    assert "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN" not in text
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" not in text


def test_shell_scripts_use_strict_mode_and_no_embedded_secret() -> None:
    for name in ("entrypoint.sh", "bootstrap.sh", "supervisor.sh"):
        text = (DEPLOY / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail")
        assert "PRIVATE KEY" not in text
