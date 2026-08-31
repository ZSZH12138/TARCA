import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_e02_cli_help_lists_exact_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_e02_v1.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for command in (
        "prepare",
        "dry-run",
        "preflight",
        "launch",
        "resume",
        "status",
        "finalize",
        "recover",
    ):
        assert command in result.stdout
