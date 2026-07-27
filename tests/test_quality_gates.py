"""Structural and behavioral contracts for the Stage 0 quality gates."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_COMMIT_PATH = PROJECT_ROOT / ".pre-commit-config.yaml"
CI_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
MAKEFILE_PATH = PROJECT_ROOT / "Makefile"
PRE_COMMIT_HOOKS_SHA = "cef0300fd0fc4d2a87a85fa2093c6b283ea36f4b"
RUFF_PRE_COMMIT_SHA = "60ef368a6f48dfb4317651017f66dbb055241a6c"
CHECKOUT_ACTION_SHA = "11d5960a326750d5838078e36cf38b85af677262"
SETUP_PYTHON_ACTION_SHA = "a26af69be951a213d495a4c3e4e4022e16d87065"
SETUP_UV_ACTION_SHA = "d0cc045d04ccac9d8b7881df0226f9e82c39688e"
MINIMUM_SECURE_PYTEST_VERSION = "9.0.3"
MINIMUM_SECURE_UV_VERSION = "0.11.15"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _repo_by_url(config: dict[str, Any], url: str) -> dict[str, Any]:
    return next(repo for repo in config["repos"] if repo["repo"] == url)


def _job_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert len(jobs) == 1
    job = next(iter(jobs.values()))
    return job["steps"]


def _make_recipe(text: str, target: str) -> str:
    lines = text.splitlines()
    start = next(
        index + 1
        for index, line in enumerate(lines)
        if line.startswith(f"{target}:") and not line.startswith(("\t", " "))
    )
    recipe_lines: list[str] = []
    for line in lines[start:]:
        if line and not line.startswith(("\t", " ")):
            break
        if line.strip():
            recipe_lines.append(line.strip())
    return "\n".join(recipe_lines)


def test_pre_commit_uses_pinned_official_hooks() -> None:
    config = _load_yaml(PRE_COMMIT_PATH)

    standard = _repo_by_url(config, "https://github.com/pre-commit/pre-commit-hooks")
    assert standard["rev"] == PRE_COMMIT_HOOKS_SHA
    assert {hook["id"] for hook in standard["hooks"]} >= {
        "trailing-whitespace",
        "end-of-file-fixer",
        "check-yaml",
        "check-added-large-files",
        "check-merge-conflict",
        "detect-private-key",
        "check-case-conflict",
    }

    ruff = _repo_by_url(config, "https://github.com/astral-sh/ruff-pre-commit")
    assert ruff["rev"] == RUFF_PRE_COMMIT_SHA
    assert {hook["id"] for hook in ruff["hooks"]} == {"ruff-check", "ruff-format"}


def test_repository_hygiene_hook_is_cross_platform_and_shell_free() -> None:
    config = _load_yaml(PRE_COMMIT_PATH)
    local = _repo_by_url(config, "local")
    hook = next(item for item in local["hooks"] if item["id"] == "repository-hygiene")

    assert hook["language"] == "python"
    assert hook["entry"] == "python -I"
    assert hook["args"][0] == "-c"
    assert hook.get("pass_filenames", True) is True
    code = hook["args"][1]
    assert "subprocess" not in code
    assert "shell=" not in code

    sentinel = hook["args"][2]

    def run_for(*paths: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code, sentinel, *paths],
            check=False,
            capture_output=True,
            text=True,
        )

    for allowed in (
        ".env.example",
        "data/raw/README.md",
        "data/interim/README.md",
        "data/processed/README.md",
        "src/tarca/stage0/models.py",
        "artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md",
        "artifacts/stage0/third_party_commits.json",
        "artifacts/stage0/reference_smoke/plot/result_summary.md",
        "artifacts/stage0/reference_smoke/diroca/result_summary.md",
    ):
        assert run_for(allowed).returncode == 0, allowed

    for forbidden in (
        ".env",
        "config/.env.local",
        "config/.env.local/secret.txt",
        "config/.env.example/secret.txt",
        "weights/model.pt",
        "weights/model.pth",
        "weights/model.ckpt",
        "weights/model.onnx",
        "weights/model.safetensors",
        "weights/model.bin",
        ".cache/third_party/plot/file.py",
        r".cache\third_party\diroca\file.py",
        "data/raw/observations.csv",
        "data/interim/features.parquet",
        "data/processed/train.npz",
        "data/raw/nested/README.md",
        ".superpowers/sdd/2026-07-24-stage0-public-release-design.md",
        "docs/superpowers/plans/2026-07-23-stage0-implementation.md",
        "docs/MANUAL_VERIFICATION_STAGE0.md",
        "artifacts/stage0/command_log.json",
        "artifacts/stage0/doctor_report.json",
        "artifacts/stage0/reference_smoke/plot/status.json",
        "artifacts/stage0/reference_smoke/diroca/stderr.log",
    ):
        result = run_for(forbidden)
        assert result.returncode != 0, forbidden
        assert forbidden in result.stdout


def test_ci_is_cpu_only_offline_and_frozen() -> None:
    workflow = _load_yaml(CI_PATH)

    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}

    job = next(iter(workflow["jobs"].values()))
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 30
    required_environment = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    assert required_environment.items() <= job["env"].items()

    steps = _job_steps(workflow)
    uses = [step["uses"] for step in steps if "uses" in step]
    assert f"actions/checkout@{CHECKOUT_ACTION_SHA}" in uses
    assert f"actions/setup-python@{SETUP_PYTHON_ACTION_SHA}" in uses
    assert f"astral-sh/setup-uv@{SETUP_UV_ACTION_SHA}" in uses

    setup_python_uses = f"actions/setup-python@{SETUP_PYTHON_ACTION_SHA}"
    python_step = next(step for step in steps if step.get("uses") == setup_python_uses)
    assert python_step["with"]["python-version"] == "3.11"

    setup_uv_uses = f"astral-sh/setup-uv@{SETUP_UV_ACTION_SHA}"
    uv_step = next(step for step in steps if step.get("uses") == setup_uv_uses)
    assert uv_step["with"]["version"] == MINIMUM_SECURE_UV_VERSION

    commands = [step["run"] for step in steps if "run" in step]
    assert commands == [
        "uv sync --frozen --extra research --group dev",
        "uv run python -m compileall -q src scripts tests third_party_manifest",
        "uv run pytest -q",
        "uv run pre-commit run --all-files",
    ]
    all_commands = "\n".join(commands).lower()
    assert not any(
        token in all_commands for token in ("plot", "diroca", "gemma", "mcqa", "slurm", "sweep")
    )


def test_pyproject_enforces_stage1_coverage_and_ruff() -> None:
    config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

    dev_dependencies = set(config["dependency-groups"]["dev"])
    assert f"pytest>={MINIMUM_SECURE_PYTEST_VERSION},<10" in dev_dependencies
    assert f"uv>={MINIMUM_SECURE_UV_VERSION},<0.12" in dev_dependencies

    pytest_options = config["tool"]["pytest"]["ini_options"]
    assert pytest_options["testpaths"] == ["tests"]
    addopts = pytest_options["addopts"]
    assert "--cov" in addopts
    assert not any(option.startswith("--cov=") for option in addopts)
    assert "--cov-fail-under=80" in addopts

    coverage = config["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["run"]["source"] == [
        "tarca.stage0",
        "tarca.contracts",
        "tarca.data.synthetic",
    ]
    assert coverage["report"]["fail_under"] == 80

    ruff = config["tool"]["ruff"]
    assert ruff["target-version"] == "py311"
    assert {"src", "scripts", "tests", "third_party_manifest"} <= set(ruff["src"])
    assert {"E", "F", "I", "B", "UP", "RUF"} <= set(ruff["lint"]["select"])


def test_makefile_targets_use_overrideable_conda_uv_and_bounded_smoke() -> None:
    text = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "TARCA_CONDA_PREFIX ?= $(UV_PROJECT_ENVIRONMENT)" in text, (
        "Makefile must derive its optional Windows prefix from the environment."
    )
    assert "ifeq ($(OS),Windows_NT)" in text, "Makefile must branch for Windows."
    assert "UV_CMD ?= $(TARCA_CONDA_PREFIX)/Scripts/uv.exe" in text, (
        "The Windows branch must derive its collision-free uv command from the "
        "caller-provided prefix."
    )
    assert "UV_CMD ?= uv" in text, "The non-Windows branch must retain plain uv."
    assert re.search(r"\b[A-Za-z]:[\\/]", text) is None, (
        "Makefile must not embed an absolute maintainer-machine path."
    )
    assert ".PHONY: lock sync doctor smoke test lint stage0-check" in text

    assert '"$(UV_CMD)" lock' in _make_recipe(text, "lock")
    assert '"$(UV_CMD)" sync --frozen --extra research --group dev' in _make_recipe(text, "sync")
    assert "scripts/doctor.py" in _make_recipe(text, "doctor")

    smoke = _make_recipe(text, "smoke").lower()
    assert "tests/test_doctor.py" in smoke
    assert "tests/test_pot_smoke.py" in smoke
    assert "tests/test_torch_hook_smoke.py" in smoke
    assert not any(
        token in smoke for token in ("plot", "diroca", "gemma", "mcqa", "slurm", "sweep")
    )

    assert '"$(UV_CMD)" run pytest -q' in _make_recipe(text, "test")
    lint = _make_recipe(text, "lint")
    assert '"$(UV_CMD)" run ruff check .' in lint
    assert '"$(UV_CMD)" run ruff format --check .' in lint

    stage0 = _make_recipe(text, "stage0-check")
    assert "python -m compileall -q src scripts tests third_party_manifest" in stage0
    assert "pytest -q" in stage0
    assert "pre-commit run --all-files" in stage0
    assert "scripts/doctor.py" in stage0


def test_makefile_expands_portable_uv_for_each_platform_branch() -> None:
    make = shutil.which("make")
    if make is None:
        pytest.skip("GNU make is unavailable locally; CI must exercise Makefile expansion.")

    inherited_environment = {**os.environ, "UV": "/opt/hostedtoolcache/uv/ci/uv"}
    windows_prefix = "C:/portable/tarca-conda"
    windows = subprocess.run(
        [
            make,
            "-pn",
            "-f",
            str(MAKEFILE_PATH),
            "OS=Windows_NT",
            f"TARCA_CONDA_PREFIX={windows_prefix}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=inherited_environment,
    )
    assert windows.returncode == 0, windows.stderr
    assert f'"{windows_prefix}/Scripts/uv.exe" lock' in windows.stdout, (
        "Windows Makefile expansion must use the explicit caller-provided prefix."
    )

    non_windows = subprocess.run(
        [make, "-pn", "-f", str(MAKEFILE_PATH), "OS=Linux"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=inherited_environment,
    )
    assert non_windows.returncode == 0, non_windows.stderr
    assert '"uv" lock' in non_windows.stdout, "Non-Windows Makefile expansion must use plain uv."
