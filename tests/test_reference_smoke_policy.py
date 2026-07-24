from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0 import reference_smoke as smoke  # noqa: E402
from tarca.stage0.reference_smoke import (  # noqa: E402
    MAX_LOG_BYTES,
    CommandOutcome,
    ResourceEstimate,
    SmokeCheck,
    SmokeResult,
    SmokeStatus,
)

PLOT_COMMIT = "96dbec5f04bc03aea6e55c430eeafd5c9be27fb2"
DIROCA_COMMIT = "7002947b4954abea1f3d11fcb6f36e7f3c43e8bd"


def _result(status: SmokeStatus, *, phase: str = "COMPONENT") -> SmokeResult:
    return SmokeResult(
        name="plot",
        status=status,
        command=(("python", "-m", "compileall", "-q", "experiments"),),
        commit=PLOT_COMMIT,
        exit_code=0 if status in {SmokeStatus.IMPORT_ONLY, SmokeStatus.PARTIAL} else None,
        duration_seconds=0.25,
        used_gpu=False,
        expected_commit=PLOT_COMMIT,
        observed_commit=PLOT_COMMIT,
        phase=phase,
        reason="fixture result",
        checks=(SmokeCheck(name="fixture", status="PASS", detail="safe"),),
        policy_error=phase == "POLICY",
    )


@pytest.mark.parametrize(
    ("estimate", "total", "available", "disk_free", "expected_status"),
    [
        (
            ResourceEstimate(80, 10, False, 30),
            100,
            100,
            100,
            SmokeStatus.BLOCKED_BY_HARDWARE,
        ),
        (
            ResourceEstimate(10, 70, False, 30),
            100,
            100,
            100,
            SmokeStatus.BLOCKED_BY_HARDWARE,
        ),
        (
            ResourceEstimate(10, 10, True, 30),
            100,
            100,
            100,
            SmokeStatus.BLOCKED_BY_HARDWARE,
        ),
        (
            ResourceEstimate(10, 10, False, 30),
            100,
            100,
            100,
            None,
        ),
    ],
)
def test_resource_gate_enforces_memory_disk_and_gpu_limits(
    estimate: ResourceEstimate,
    total: int,
    available: int,
    disk_free: int,
    expected_status: SmokeStatus | None,
) -> None:
    block = smoke.evaluate_resource_gate(
        estimate,
        total_memory_bytes=total,
        available_memory_bytes=available,
        free_disk_bytes=disk_free,
    )
    if expected_status is None:
        assert block is None
    else:
        assert block is not None
        assert block.status is expected_status


def test_low_current_available_memory_is_blocked() -> None:
    block = smoke.evaluate_resource_gate(
        ResourceEstimate(50, 10, False, 30),
        total_memory_bytes=100,
        available_memory_bytes=20,
        free_disk_bytes=100,
    )
    assert block is not None
    assert block.status is SmokeStatus.BLOCKED_BY_HARDWARE


@pytest.mark.parametrize("status", list(SmokeStatus))
def test_every_terminal_status_writes_complete_atomic_artifact_set(
    tmp_path: Path,
    status: SmokeStatus,
) -> None:
    result = _result(status)
    target = tmp_path / status.value.lower()

    smoke.write_artifacts(target, result)

    _assert_seven_artifacts(target)
    payload = json.loads((target / "status.json").read_text(encoding="utf-8"))
    assert payload["status"] == status.value
    assert payload["command"]
    assert payload["commit"] == PLOT_COMMIT
    assert payload["expected_commit"] == PLOT_COMMIT
    assert payload["observed_commit"] == PLOT_COMMIT
    assert payload["used_gpu"] is False
    assert isinstance(payload["duration_seconds"], float)
    assert payload["phase"]
    assert payload["reason"]
    assert payload["checks"]
    assert payload["generated_at"].endswith("Z")
    assert not list(target.glob("*.tmp"))


def test_logs_are_truncated_and_environment_secrets_are_redacted(tmp_path: Path) -> None:
    secret = "do-not-leak-12345"
    original = os.environ.get("TEST_API_TOKEN")
    os.environ["TEST_API_TOKEN"] = secret
    try:
        noisy = f"TEST_API_TOKEN={secret}\n" + ("x" * (MAX_LOG_BYTES * 2))
        result = replace(
            _result(SmokeStatus.PARTIAL),
            stdout=noisy,
            stderr=f"password={secret}",
        )
        smoke.write_artifacts(tmp_path, result)
    finally:
        if original is None:
            os.environ.pop("TEST_API_TOKEN", None)
        else:
            os.environ["TEST_API_TOKEN"] = original

    stdout = (tmp_path / "stdout.log").read_text(encoding="utf-8")
    stderr = (tmp_path / "stderr.log").read_text(encoding="utf-8")
    environment = (tmp_path / "environment.txt").read_text(encoding="utf-8")
    assert secret not in stdout
    assert secret not in stderr
    assert secret not in environment
    assert "[REDACTED]" in stdout
    assert "[TRUNCATED]" in stdout
    assert len(stdout.encode("utf-8")) <= MAX_LOG_BYTES + 256


def test_smoke_results_and_nested_checks_are_immutable() -> None:
    result = _result(SmokeStatus.PARTIAL)

    with pytest.raises((AttributeError, TypeError)):
        result.reason = "mutated"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        result.checks[0].detail = "mutated"  # type: ignore[misc]
    assert isinstance(result.environment, MappingProxyType)
    with pytest.raises(TypeError):
        result.environment["TOKEN"] = "bad"  # type: ignore[index]


def test_candidate_commands_use_current_python_and_contain_no_mutation_or_download() -> None:
    all_commands = smoke.PLOT_COMMANDS | smoke.DIROCA_COMMANDS
    rendered = "\n".join(" ".join(command) for command in all_commands).lower()

    assert all(command[0] == smoke.SAFE_PYTHON for command in all_commands)
    assert "pip install" not in rendered
    assert "conda install" not in rendered
    assert "uv add" not in rendered
    assert "download" not in rendered
    assert "mcqa" not in rendered
    assert "gemma" not in rendered
    assert "cuda" not in rendered
    assert "gpu" not in rendered
    assert "sweep" not in rendered


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import torch\n\ndef sinkhorn_uniform_ot(c, r, n):\n    return c\n", "PASS"),
        (
            "from dataclasses import dataclass\n"
            "from typing import Sequence\n"
            "import torch\n"
            "from .data import CarryPairRecord\n"
            "from .features import abstract_effect_signature\n"
            "from .interventions import RunCache\n"
            "from .model import GRUAdder\n"
            "from .sites import Site\n\n"
            "def sinkhorn_uniform_ot(c, r, n):\n"
            "    return c\n",
            "PASS",
        ),
        ("import requests\n\ndef sinkhorn_uniform_ot(c, r, n):\n    return c\n", "FAIL"),
        ("import torch\n\ndef other(c):\n    return c\n", "FAIL"),
        (
            "import torch\n\ndef sinkhorn_uniform_ot(c, r, n):\n    return eval('c')\n",
            "FAIL",
        ),
        ("this is not valid python !", "FAIL"),
    ],
)
def test_plot_component_ast_audit_rejects_unexpected_code(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    component = tmp_path / "experiments" / "binary_addition" / "transport.py"
    component.parent.mkdir(parents=True)
    component.write_text(source, encoding="utf-8")
    assert smoke._audit_plot_component(tmp_path).status == expected


def test_plot_component_audit_skips_absent_file(tmp_path: Path) -> None:
    assert smoke._audit_plot_component(tmp_path).status == "SKIP"


def test_plot_component_audit_rejects_dangerous_transitive_local_import(
    tmp_path: Path,
) -> None:
    component_root = tmp_path / "experiments" / "binary_addition"
    component_root.mkdir(parents=True)
    (component_root / "transport.py").write_text(
        "from .features import helper\n\ndef sinkhorn_uniform_ot(c, r, n):\n    return helper(c)\n",
        encoding="utf-8",
    )
    (component_root / "features.py").write_text(
        "import socket\n\ndef helper(value):\n    return value\n",
        encoding="utf-8",
    )

    result = smoke._audit_plot_component(tmp_path)

    assert result.status == "FAIL"
    assert "features.py" in result.detail
    assert "socket" in result.detail


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ((), SmokeStatus.FAILED),
        (
            (CommandOutcome(("compile",), 1, "", "syntax", 0.1),),
            SmokeStatus.FAILED,
        ),
        (
            (
                CommandOutcome(("compile",), 0, "", "", 0.1),
                CommandOutcome(
                    ("component",),
                    1,
                    "",
                    "ModuleNotFoundError: torch",
                    0.1,
                ),
            ),
            SmokeStatus.BLOCKED_BY_DEPENDENCY,
        ),
        (
            (
                CommandOutcome(("compile",), 0, "", "", 0.1),
                CommandOutcome(("component",), 1, "", "assertion", 0.1),
            ),
            SmokeStatus.FAILED,
        ),
    ],
)
def test_plot_outcome_failure_classification(
    outcomes: tuple[CommandOutcome, ...],
    expected: SmokeStatus,
) -> None:
    assert smoke.classify_plot_outcomes(outcomes, PLOT_COMMIT, ()).status is expected


def test_plot_outcome_audit_failure_does_not_masquerade_as_import_only() -> None:
    compile_only = (CommandOutcome(("compile",), 0, "", "", 0.1),)
    checks = (SmokeCheck("plot_component_audit", "FAIL", "unexpected imports"),)

    result = smoke.classify_plot_outcomes(compile_only, PLOT_COMMIT, checks)

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "COMPONENT"
    assert "audit" in result.reason.lower()


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ((), SmokeStatus.FAILED),
        (
            (CommandOutcome(("compile",), 1, "", "syntax", 0.1),),
            SmokeStatus.FAILED,
        ),
        (
            (
                CommandOutcome(("compile",), 0, "", "", 0.1),
                CommandOutcome(("help",), 1, "", "ModuleNotFoundError: cvxpy", 0.1),
            ),
            SmokeStatus.BLOCKED_BY_DEPENDENCY,
        ),
        (
            (
                CommandOutcome(("compile",), 0, "", "", 0.1),
                CommandOutcome(("help",), 1, "", "bad arguments", 0.1),
            ),
            SmokeStatus.FAILED,
        ),
        (
            (CommandOutcome(("compile",), 0, "", "", 0.1),),
            SmokeStatus.IMPORT_ONLY,
        ),
    ],
)
def test_diroca_outcome_failure_classification(
    outcomes: tuple[CommandOutcome, ...],
    expected: SmokeStatus,
) -> None:
    assert smoke.classify_diroca_outcomes(outcomes, DIROCA_COMMIT, ()).status is expected


def test_smoke_result_rejects_gpu_claim() -> None:
    with pytest.raises(ValueError, match="GPU"):
        replace(_result(SmokeStatus.PARTIAL), used_gpu=True)


def _assert_seven_artifacts(directory: Path) -> None:
    expected = {
        "commit.txt",
        "environment.txt",
        "command.txt",
        "stdout.log",
        "stderr.log",
        "result_summary.md",
        "status.json",
    }
    assert {path.name for path in directory.iterdir()} == expected
