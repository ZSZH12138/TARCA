from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/run_stage1b_runtime.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(SCRIPT), *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_runtime_cli_has_no_formal_experiment_commands() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    assert all(name in completed.stdout for name in ("preflight", "launch", "resume", "status"))
    assert "E01" not in completed.stdout and "E02" not in completed.stdout


def test_empty_status_is_safe_json_and_requires_explicit_empty_permission(tmp_path: Path) -> None:
    blocked = _run("--artifact-root", str(tmp_path), "status")
    allowed = _run("--artifact-root", str(tmp_path), "status", "--empty-ok")

    assert blocked.returncode == 1
    payload = json.loads(allowed.stdout)
    assert payload == {"status": "EMPTY"}
    assert "crps" not in allowed.stdout.lower()
    assert "truth" not in allowed.stdout.lower()


def test_resume_requires_an_existing_execution_database(tmp_path: Path) -> None:
    completed = _run("--artifact-root", str(tmp_path), "resume")

    assert completed.returncode == 1
    assert "execution database" in completed.stdout.lower()
