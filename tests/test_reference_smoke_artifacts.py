from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.reference_smoke_artifacts import (  # noqa: E402
    _status_payload,
    redact_text,
    truncate_log,
    write_artifacts,
)
from tarca.stage0.reference_smoke_policy import (  # noqa: E402
    MAX_LOG_BYTES,
    SmokeCheck,
    SmokeResult,
    SmokeStatus,
)


def _result() -> SmokeResult:
    return SmokeResult(
        name="plot",
        status=SmokeStatus.PARTIAL,
        command=(("python", "-m", "compileall", "-q", "experiments"),),
        commit="a" * 40,
        exit_code=0,
        duration_seconds=0.25,
        used_gpu=False,
        expected_commit="a" * 40,
        observed_commit="a" * 40,
        phase="COMPONENT",
        reason="fixture reason",
        checks=(SmokeCheck("fixture", "PASS", "safe detail"),),
        environment={"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
    )


def test_redact_text_masks_env_secret_values_and_secret_assignments() -> None:
    original = os.environ.get("TEST_API_TOKEN")
    os.environ["TEST_API_TOKEN"] = "secret-12345"
    try:
        redacted = redact_text("token=secret-12345 password=secret-12345 visible")
    finally:
        if original is None:
            os.environ.pop("TEST_API_TOKEN", None)
        else:
            os.environ["TEST_API_TOKEN"] = original

    assert "secret-12345" not in redacted
    assert "[REDACTED]" in redacted


def test_truncate_log_redacts_and_caps_large_logs() -> None:
    original = os.environ.get("TEST_API_TOKEN")
    os.environ["TEST_API_TOKEN"] = "secret-12345"
    try:
        truncated = truncate_log("TEST_API_TOKEN=secret-12345\n" + ("x" * (MAX_LOG_BYTES * 2)))
    finally:
        if original is None:
            os.environ.pop("TEST_API_TOKEN", None)
        else:
            os.environ["TEST_API_TOKEN"] = original

    assert "secret-12345" not in truncated
    assert "[REDACTED]" in truncated
    assert "[TRUNCATED]" in truncated


def test_status_payload_is_json_ready_and_redacts_reason_and_checks() -> None:
    original = os.environ.get("TEST_API_TOKEN")
    os.environ["TEST_API_TOKEN"] = "secret-12345"
    try:
        payload = _status_payload(
            SmokeResult(
                name="plot",
                status=SmokeStatus.FAILED,
                command=(("python", "safe.py"),),
                commit="b" * 40,
                exit_code=1,
                duration_seconds=0.5,
                used_gpu=False,
                expected_commit="b" * 40,
                observed_commit="b" * 40,
                phase="EXCEPTION",
                reason="token=secret-12345",
                checks=(SmokeCheck("fixture", "FAIL", "password=secret-12345"),),
            )
        )
    finally:
        if original is None:
            os.environ.pop("TEST_API_TOKEN", None)
        else:
            os.environ["TEST_API_TOKEN"] = original

    assert payload["status"] == "FAILED"
    assert payload["used_gpu"] is False
    assert "[REDACTED]" in str(payload["reason"])
    assert "[REDACTED]" in str(payload["checks"][0]["detail"])
    json.dumps(payload, ensure_ascii=False)


def test_write_artifacts_creates_expected_files_without_temp_leaks(tmp_path: Path) -> None:
    result = _result()

    write_artifacts(tmp_path, result)

    expected = {
        "command.txt",
        "commit.txt",
        "environment.txt",
        "result_summary.md",
        "status.json",
        "stderr.log",
        "stdout.log",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "commit.txt").read_text(encoding="utf-8").strip() == "a" * 40
    assert "compileall" in (tmp_path / "command.txt").read_text(encoding="utf-8")
    assert '"HF_HUB_OFFLINE": "1"' in (tmp_path / "environment.txt").read_text(encoding="utf-8")
    assert "not an end-to-end PLOT reproduction" not in (tmp_path / "result_summary.md").read_text(
        encoding="utf-8"
    )
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PARTIAL"
