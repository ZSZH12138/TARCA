from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_qualification_cli_exposes_only_probe_qualify_and_freeze_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_stage1b_qualification.py", "--help"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "probe" in completed.stdout
    assert "qualify" in completed.stdout
    assert "freeze" in completed.stdout
    assert "E01" not in completed.stdout
    assert "E02" not in completed.stdout


def test_check_cli_truthfully_allows_unfrozen_state() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_stage1b.py", "--allow-unfrozen", "--json"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["status"] in {"PASS", "UNFROZEN"}
