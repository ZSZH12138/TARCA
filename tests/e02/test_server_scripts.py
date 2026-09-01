from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _bash_syntax_command(script: Path) -> list[str]:
    if os.name != "nt":
        return ["bash", "-n", str(script)]
    drive = script.drive[0].lower()
    posix_path = script.as_posix()[2:]
    return ["wsl", "bash", "-n", f"/mnt/{drive}{posix_path}"]


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()
    return f"/mnt/{resolved.drive[0].lower()}{resolved.as_posix()[2:]}"


def test_e02_fresh_server_scripts_stop_before_formal_launch() -> None:
    host = (ROOT / "deploy/stage2/e02_bootstrap.sh").read_text(encoding="utf-8")
    direct = (ROOT / "deploy/stage2/e02_bootstrap_direct.sh").read_text(encoding="utf-8")

    assert "sha256sum --check" in host
    assert "docker compose" in host
    assert "e02-bootstrap" in host
    assert "--stage2-archive" in host and "--server-bundle" in host

    for marker in (
        "--system-site-packages",
        "--no-index",
        "--require-hashes",
        "tarca.e02.server_handoff verify-bundle",
        "tarca.e02.server_handoff restore",
        "run_e02_v1.py\" prepare",
        "run_e02_v1.py\" dry-run",
        "tarca.e02.server_preflight",
        "run_e02_v1.py\" preflight",
        "E02_READY_FOR_USER_LAUNCH",
    ):
        assert marker in direct
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" not in host
    assert "I_ACKNOWLEDGE_E02_V1_FORMAL_RUN" not in direct
    assert "run_e02_v1.py\" launch" not in direct


def test_e02_fresh_server_scripts_have_valid_bash_syntax() -> None:
    for relative in (
        "deploy/stage2/e02_bootstrap.sh",
        "deploy/stage2/e02_bootstrap_direct.sh",
    ):
        completed = subprocess.run(
            _bash_syntax_command(ROOT / relative),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_e02_direct_bootstrap_materializes_sources_before_restore(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    capsule = repository / (
        "artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz"
    )
    capsule.parent.mkdir(parents=True)
    capsule.write_bytes(b"offline-source-capsule")
    (capsule.parent / "stage2-v1-official-sources.tar.gz.receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    inputs = (
        handoff / "tarca-stage2-v1-server.tar.gz",
        handoff / "tarca-stage2-v1-complete-20260901T011423Z.tar.gz",
    )
    for path in inputs:
        path.write_bytes(path.name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_bytes(
            f"{digest}  {path.name}\n".encode()
        )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_bytes(
        b"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$TARCA_TEST_LOG\"\n"
    )
    log = tmp_path / "python-calls.log"
    if os.name == "nt":
        subprocess.run(
            ["wsl", "chmod", "+x", _wsl_path(fake_python)], check=True
        )
        command = [
            "wsl",
            "env",
            f"PATH={_wsl_path(fake_bin)}:/usr/bin:/bin",
            f"TARCA_TEST_LOG={_wsl_path(log)}",
            "bash",
            _wsl_path(ROOT / "deploy/stage2/e02_bootstrap_direct.sh"),
        ]
    else:
        fake_python.chmod(0o755)
        command = [
            "env",
            f"PATH={fake_bin}:/usr/bin:/bin",
            f"TARCA_TEST_LOG={log}",
            "bash",
            str(ROOT / "deploy/stage2/e02_bootstrap_direct.sh"),
        ]
    completed = subprocess.run(
        [
            *command,
            "--repository-root",
            _wsl_path(repository),
            "--stage2-archive",
            _wsl_path(inputs[1]),
            "--server-bundle",
            _wsl_path(inputs[0]),
            "--remaining-rental-hours",
            "22",
            "--use-current-python",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    verify_index = next(
        index for index, call in enumerate(calls)
        if "tarca.e02.server_handoff verify-bundle" in call
    )
    assert any("scripts/import_stage2_source_capsule.py" in call for call in calls)
    import_index = next(
        index for index, call in enumerate(calls)
        if "scripts/import_stage2_source_capsule.py" in call
    )
    restore_index = next(
        index for index, call in enumerate(calls)
        if "tarca.e02.server_handoff restore" in call
    )
    assert verify_index < import_index < restore_index
