from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/stage2/recovery_bootstrap.sh"


def test_recovery_bootstrap_is_fail_closed_and_stops_before_user_resume() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    restore = script.index("restore-input")
    preflight = script.index("bootstrap --mode preflight")
    repair = script.index("stage2 repair")
    monitor = script.index("up -d tarca-stage2")
    ready = script.index("RECOVERY_READY_FOR_USER_RESUME")
    assert restore < preflight < repair < monitor < ready
    assert "sha256sum" in script
    assert 'recovery_container_path="/recovery/$(basename -- "$recovery_archive")"' in script
    assert '--recovery-archive "${recovery_container_path}"' in script
    assert "--user 0:0" in script
    assert "chown -R tarca:tarca /opt/tarca/artifacts" in script
    assert "I_ACKNOWLEDGE_STAGE2_DEVICE_MISMATCH_RECOVERY_V1" in script
    assert "I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN" in script
    assert "stage2 resume" in script
    assert "codex" not in script.lower()


def test_recovery_bootstrap_does_not_execute_resume_implicitly() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'printf \'%s\\n\' "${resume_command}"' in script
    assert '${compose[@]} run --rm tarca-stage2 stage2 resume' not in script
