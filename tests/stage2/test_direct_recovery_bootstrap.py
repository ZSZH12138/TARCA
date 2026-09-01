from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/stage2/recovery_bootstrap_direct.sh"


def _bash_command(script: Path, *arguments: str) -> list[str]:
    if os.name != "nt":
        return ["bash", str(script), *arguments]
    drive = script.drive[0].lower()
    posix_path = script.as_posix()[2:]
    return ["wsl", "bash", f"/mnt/{drive}{posix_path}", *arguments]


def test_container_native_recovery_entrypoint_has_a_read_only_help_path() -> None:
    result = subprocess.run(
        _bash_command(SCRIPT, "--help"),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "TARCA Stage 2 container-native recovery",
        "Restores, preflights, repairs, starts the read-only monitor, and stops before resume.",
    ]


def test_container_native_recovery_reports_only_a_fixed_failure_stage() -> None:
    result = subprocess.run(
        _bash_command(
            SCRIPT,
            "--recovery-archive",
            "/definitely/missing/recovery.tar.gz",
            "--server-bundle",
            "/definitely/missing/server.tar.gz",
            "--remaining-rental-hours",
            "22",
            "--repository-root",
            "/definitely/missing/repository",
        ),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )

    assert result.returncode != 0
    assert "TARCA_DIRECT_BOOTSTRAP_FAILED_STAGE=validate-inputs" in result.stderr
    assert "/definitely/missing" not in result.stderr


def test_container_native_recovery_does_not_require_ensurepip() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert '"$base_python" -m venv --system-site-packages --without-pip' in script


def test_container_native_recovery_selects_a_compatible_image_python() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "for python_candidate in python python3 /opt/conda/bin/python; do" in script
    assert '"$python_candidate" -c' in script
