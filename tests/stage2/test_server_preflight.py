from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_server_preflight_has_an_importable_spawn_safe_module_entrypoint() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "deploy/stage2/py310"), str(ROOT / "src"))),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "tarca.stage2.server_preflight", "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "remaining-rental-hours" in completed.stdout
