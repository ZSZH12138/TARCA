from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0 import reference_smoke as smoke  # noqa: E402
from tarca.stage0.reference_smoke import (  # noqa: E402
    MAX_TIMEOUT_SECONDS,
    CommandOutcome,
    RepositoryOutcome,
    SmokeCheck,
    SmokeResult,
    SmokeStatus,
    run_reference_smoke,
)

_CLI_SPEC = importlib.util.spec_from_file_location(
    "tarca_stage0_reference_smoke_cli",
    PROJECT_ROOT / "scripts" / "run_reference_smoke.py",
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
smoke_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(smoke_cli)

EXPECTED_COMMITS = {
    "plot": "96dbec5f04bc03aea6e55c430eeafd5c9be27fb2",
    "diroca": "7002947b4954abea1f3d11fcb6f36e7f3c43e8bd",
}

EXPECTED_REPOSITORIES = {
    "plot": "https://github.com/jchang153/causal-abstractions-ot",
    "diroca": "https://github.com/yfelekis/DiRoCA",
}


def _manifest_entry(name: str) -> dict[str, str]:
    return {
        "name": name,
        "paper_title": f"{name} paper",
        "paper_url": "https://example.org/paper",
        "repository_url": EXPECTED_REPOSITORIES[name],
        "role": "reference",
        "license": "UNVERIFIED",
        "default_branch": "main",
        "verified_commit": EXPECTED_COMMITS[name],
        "verified_at": "2026-07-23",
        "local_reference_path": f".cache/third_party/{name}",
        "notes": "fixture",
    }


@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest_dir = tmp_path / "third_party_manifest"
    manifest_dir.mkdir()
    (manifest_dir / "sources.yaml").write_text(
        yaml.safe_dump(
            [_manifest_entry("plot"), _manifest_entry("diroca")],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _result(status: SmokeStatus, *, phase: str = "COMPONENT") -> SmokeResult:
    return SmokeResult(
        name="plot",
        status=status,
        command=(("python", "-m", "compileall", "-q", "experiments"),),
        commit=EXPECTED_COMMITS["plot"],
        exit_code=0 if status in {SmokeStatus.IMPORT_ONLY, SmokeStatus.PARTIAL} else None,
        duration_seconds=0.25,
        used_gpu=False,
        expected_commit=EXPECTED_COMMITS["plot"],
        observed_commit=EXPECTED_COMMITS["plot"],
        phase=phase,
        reason="fixture result",
        checks=(SmokeCheck(name="fixture", status="PASS", detail="safe"),),
        policy_error=phase == "POLICY",
    )


@pytest.mark.parametrize("name", ["unknown", "", "PLOT", "../plot"])
def test_only_exact_lowercase_reference_names_are_accepted(name: str) -> None:
    with pytest.raises(ValueError, match=r"plot.*diroca"):
        run_reference_smoke(name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "command",
    [
        ("python", "experiments/mcqa/run.py"),
        ("python", "-c", "load Gemma-2-2B"),
        ("bash", "submit.SLURM"),
        ("python", "run_sweep.py"),
        ("python", "--device", "CUDA"),
        ("python", "--accelerator", "gpu"),
        ("pip", "install", "package"),
        ("python", "download_model.py"),
        ("python", "train.py"),
    ],
)
def test_dangerous_or_dependency_mutating_commands_are_rejected(
    command: tuple[str, ...],
) -> None:
    with pytest.raises(smoke.SmokePolicyError):
        smoke.validate_candidate_command(command, {command})


def test_command_must_match_internal_allowlist_exactly() -> None:
    allowed = {("python", "generate_data.py", "--help")}

    assert smoke.validate_candidate_command(
        ("python", "generate_data.py", "--help"),
        allowed,
    ) == ("python", "generate_data.py", "--help")
    with pytest.raises(smoke.SmokePolicyError, match="allowlist"):
        smoke.validate_candidate_command(
            ("python", "generate_data.py", "--config", "user.yaml"),
            allowed,
        )


@pytest.mark.parametrize("timeout", [0, -1, MAX_TIMEOUT_SECONDS + 1, 7_200])
def test_timeout_must_be_positive_and_capped_well_below_two_hours(timeout: int) -> None:
    with pytest.raises(ValueError, match="timeout"):
        smoke.validate_timeout(timeout)


def test_timeout_accepts_hard_upper_bound() -> None:
    assert smoke.validate_timeout(MAX_TIMEOUT_SECONDS) == MAX_TIMEOUT_SECONDS
    assert MAX_TIMEOUT_SECONDS <= 300


def test_manifest_commit_mismatch_fails_before_subprocess_and_writes_artifacts(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = isolated_project / "third_party_manifest" / "sources.yaml"
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries[0]["verified_commit"] = "0" * 40
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run after manifest mismatch")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "POLICY"
    assert result.policy_error is True
    _assert_seven_artifacts(isolated_project / "artifacts" / "stage0" / "reference_smoke" / "plot")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("repository_url", "https://github.com/attacker/fork"),
        ("default_branch", "develop"),
        ("local_reference_path", ".cache/third_party/other"),
    ],
)
def test_manifest_identity_fields_must_match_built_in_policy(
    isolated_project: Path,
    field: str,
    bad_value: str,
) -> None:
    manifest_path = isolated_project / "third_party_manifest" / "sources.yaml"
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries[0][field] = bad_value
    manifest_path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "POLICY"
    assert field in result.reason


@pytest.mark.parametrize(
    "cache_root",
    [
        Path("../third_party"),
        Path(".cache/../third_party"),
        Path("C:/absolute/third_party"),
    ],
)
def test_cache_path_traversal_and_absolute_paths_are_rejected(
    isolated_project: Path,
    cache_root: Path,
) -> None:
    result = run_reference_smoke("plot", cache_root=cache_root)

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "POLICY"
    assert result.policy_error is True


def test_reparse_point_or_symlink_cache_is_rejected(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_detector = smoke._path_has_reparse_point

    def fake_detector(path: Path, stop: Path) -> bool:
        if path.name == "third_party":
            return True
        return real_detector(path, stop)

    monkeypatch.setattr(smoke, "_path_has_reparse_point", fake_detector)

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "POLICY"
    assert "symbolic link" in result.reason or "reparse" in result.reason


def test_existing_repository_requires_exact_origin_and_head(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = isolated_project / ".cache" / "third_party" / "plot"
    (repo / ".git").mkdir(parents=True)
    responses = iter(
        [
            CommandOutcome(
                command=("git", "remote", "get-url", "origin"),
                exit_code=0,
                stdout="https://github.com/attacker/fork\n",
                stderr="",
                duration_seconds=0.01,
            ),
            CommandOutcome(
                command=("git", "rev-parse", "HEAD"),
                exit_code=0,
                stdout=f"{EXPECTED_COMMITS['plot']}\n",
                stderr="",
                duration_seconds=0.01,
            ),
        ]
    )
    monkeypatch.setattr(smoke, "_run_process", lambda *args, **kwargs: next(responses))

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "REPOSITORY"
    assert "origin" in result.reason
    assert repo.exists()


def test_git_fetch_failure_preserves_commands_logs_exit_code_and_duration(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fail_fetch(
        command: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        commands.append(command)
        if command[:2] == ("git", "init"):
            (cwd / ".git").mkdir()
        if command[:2] == ("git", "fetch"):
            return CommandOutcome(
                command,
                128,
                "partial fetch output",
                "fatal: unable to access repository: network timed out",
                1.25,
            )
        return CommandOutcome(command, 0, "", "", 0.1)

    monkeypatch.setattr(smoke, "_resource_block_for", lambda name: None)
    monkeypatch.setattr(smoke, "_run_process", fail_fetch)

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "REPOSITORY_NETWORK"
    assert result.policy_error is False
    assert result.exit_code == 128
    assert any(command[:2] == ("git", "fetch") for command in result.command)
    assert "partial fetch output" in result.stdout
    assert "network timed out" in result.stderr
    assert result.duration_seconds >= 1.25
    artifact = isolated_project / "artifacts" / "stage0" / "reference_smoke" / "plot"
    assert "git fetch" in (artifact / "command.txt").read_text(encoding="utf-8")
    assert "network timed out" in (artifact / "stderr.log").read_text(encoding="utf-8")


def test_existing_repository_git_command_failure_preserves_both_attempts(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = isolated_project / ".cache" / "third_party" / "plot"
    (repo / ".git").mkdir(parents=True)
    responses = iter(
        [
            CommandOutcome(
                ("git", "remote", "get-url", "origin"),
                0,
                EXPECTED_REPOSITORIES["plot"],
                "",
                0.2,
            ),
            CommandOutcome(
                ("git", "rev-parse", "HEAD"),
                128,
                "",
                "fatal: bad revision",
                0.3,
            ),
        ]
    )
    monkeypatch.setattr(smoke, "_resource_block_for", lambda name: None)
    monkeypatch.setattr(smoke, "_run_process", lambda *args, **kwargs: next(responses))

    result = run_reference_smoke("plot")

    assert result.phase == "REPOSITORY_GIT"
    assert result.policy_error is False
    assert result.exit_code == 128
    assert len(result.command) == 2
    assert "bad revision" in result.stderr
    assert result.duration_seconds == pytest.approx(0.5)


def test_process_runner_uses_argv_shell_false_timeout_and_offline_cpu_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    outcome = smoke._run_process(
        ("python", "-m", "compileall", "-q", "experiments"),
        cwd=tmp_path,
        timeout_seconds=10,
        env=smoke._offline_environment(),
    )

    assert outcome.exit_code == 0
    assert captured["shell"] is False
    assert captured["timeout"] == 10
    assert captured["cwd"] == tmp_path
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_clone_commands_use_fixed_commit_and_never_push_install_or_download(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_process(
        command: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        commands.append(command)
        if command[:2] == ("git", "init"):
            (cwd / ".git").mkdir()
        stdout = ""
        if command[-3:] == ("remote", "get-url", "origin"):
            stdout = EXPECTED_REPOSITORIES["plot"]
        if command[-2:] == ("rev-parse", "HEAD"):
            stdout = EXPECTED_COMMITS["plot"]
        if command[-3:] == ("rev-parse", "--abbrev-ref", "HEAD"):
            stdout = "HEAD"
        return CommandOutcome(command, 0, stdout, "", 0.01)

    monkeypatch.setattr(smoke, "_run_process", fake_process)
    policy = smoke.load_reference_policy("plot")
    outcome = smoke._ensure_repository(policy, Path(".cache/third_party"), 15)

    flattened = "\n".join(" ".join(command) for command in commands).lower()
    assert outcome.observed_commit == EXPECTED_COMMITS["plot"]
    assert EXPECTED_COMMITS["plot"] in flattened
    assert " push " not in f" {flattened} "
    assert " install " not in f" {flattened} "
    assert "download" not in flattened
    assert all(command[0] == "git" for command in commands)


def test_exception_path_still_writes_all_artifacts(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke,
        "_ensure_repository",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fixture failure")),
    )

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.FAILED
    assert result.phase == "EXCEPTION"
    assert "OSError" in result.reason
    _assert_seven_artifacts(isolated_project / "artifacts" / "stage0" / "reference_smoke" / "plot")


def test_dependency_failure_is_classified_without_claiming_smoke_passed(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = isolated_project / ".cache" / "third_party" / "diroca"
    repo.mkdir(parents=True)
    monkeypatch.setattr(
        smoke,
        "_ensure_repository",
        lambda *args, **kwargs: RepositoryOutcome(
            repository_path=repo,
            observed_commit=EXPECTED_COMMITS["diroca"],
            commands=(),
            checks=(SmokeCheck("repository", "PASS", "fixture"),),
        ),
    )
    monkeypatch.setattr(smoke, "_inspect_reference_files", lambda *args: ())
    monkeypatch.setattr(smoke, "_resource_block_for", lambda *args: None)
    monkeypatch.setattr(
        smoke,
        "_run_diroca",
        lambda *args, **kwargs: replace(
            _result(SmokeStatus.BLOCKED_BY_DEPENDENCY, phase="HELP"),
            name="diroca",
            expected_commit=EXPECTED_COMMITS["diroca"],
            observed_commit=EXPECTED_COMMITS["diroca"],
            commit=EXPECTED_COMMITS["diroca"],
            reason="ModuleNotFoundError: cvxpy",
        ),
    )

    result = run_reference_smoke("diroca")

    assert result.status is SmokeStatus.BLOCKED_BY_DEPENDENCY
    assert result.status is not SmokeStatus.SMOKE_PASSED


def test_static_compile_and_component_only_can_never_claim_smoke_passed() -> None:
    outcomes = (
        CommandOutcome(("python", "-m", "compileall"), 0, "", "", 0.1),
        CommandOutcome(("python", "-c", "safe component"), 0, "ok", "", 0.1),
    )

    result = smoke.classify_plot_outcomes(
        outcomes,
        EXPECTED_COMMITS["plot"],
        (),
    )

    assert result.status is SmokeStatus.PARTIAL
    assert result.status is not SmokeStatus.SMOKE_PASSED


def test_help_only_can_never_claim_diroca_smoke_passed() -> None:
    outcomes = (
        CommandOutcome(("python", "-m", "compileall"), 0, "", "", 0.1),
        CommandOutcome(("python", "generate_data.py", "--help"), 0, "usage", "", 0.1),
        CommandOutcome(
            ("python", "gauss_optimization.py", "--help"),
            0,
            "usage",
            "",
            0.1,
        ),
    )

    result = smoke.classify_diroca_outcomes(
        outcomes,
        EXPECTED_COMMITS["diroca"],
        (),
    )

    assert result.status is SmokeStatus.IMPORT_ONLY
    assert result.status is not SmokeStatus.SMOKE_PASSED


def test_process_runner_records_timeout_and_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=("python", "--help"),
            timeout=1,
            output="partial",
            stderr="late",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    timed_out = smoke._run_process(
        ("python", "--help"),
        cwd=tmp_path,
        timeout_seconds=1,
    )
    assert timed_out.exit_code == 124
    assert "Timed out" in timed_out.stderr
    assert timed_out.stdout == "partial"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot start")),
    )
    failed = smoke._run_process(
        ("python", "--help"),
        cwd=tmp_path,
        timeout_seconds=1,
    )
    assert failed.exit_code == 127
    assert "OSError" in failed.stderr


def test_reference_file_inspection_hashes_readme_requirements_and_configs(
    isolated_project: Path,
) -> None:
    plot_repo = isolated_project / "plot-fixture"
    plot_repo.mkdir()
    (plot_repo / "README.md").write_text("# PLOT\n", encoding="utf-8")
    (plot_repo / "requirements.txt").write_text("torch\n", encoding="utf-8")
    plot_checks = smoke._inspect_reference_files(
        smoke.load_reference_policy("plot"),
        plot_repo,
    )
    assert {check.name: check.status for check in plot_checks} == {
        "README": "PASS",
        "requirements": "PASS",
    }
    assert all("sha256=" in check.detail for check in plot_checks)

    diroca_repo = isolated_project / "diroca-fixture"
    (diroca_repo / "configs").mkdir(parents=True)
    (diroca_repo / "README.rst").write_text("DiRoCA\n", encoding="utf-8")
    (diroca_repo / "configs" / "tiny.yaml").write_text("seed: 1\n", encoding="utf-8")
    diroca_checks = smoke._inspect_reference_files(
        smoke.load_reference_policy("diroca"),
        diroca_repo,
    )
    statuses = {check.name: check.status for check in diroca_checks}
    assert statuses["README"] == "PASS"
    assert statuses["requirements"] == "WARN"
    assert statuses["configs"] == "PASS"
    assert statuses["yaml_execution"] == "SKIP"


def test_plot_and_diroca_runners_execute_only_allowlisted_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = tmp_path / "experiments" / "binary_addition" / "transport.py"
    component.parent.mkdir(parents=True)
    component.write_text(
        "import torch\n\ndef sinkhorn_uniform_ot(c, r, n):\n    return c\n",
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    def passing(
        command: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        commands.append(command)
        return CommandOutcome(command, 0, "usage" if "--help" in command else "", "", 0.1)

    monkeypatch.setattr(smoke, "_run_process", passing)
    plot_result = smoke._run_plot(
        tmp_path,
        5,
        EXPECTED_COMMITS["plot"],
        (),
    )
    diroca_result = smoke._run_diroca(
        tmp_path,
        5,
        EXPECTED_COMMITS["diroca"],
        (),
    )

    assert plot_result.status is SmokeStatus.PARTIAL
    assert diroca_result.status is SmokeStatus.IMPORT_ONLY
    assert set(commands).issubset(smoke.PLOT_COMMANDS | smoke.DIROCA_COMMANDS)
    assert not any(
        "generate_data.py" in command or "gauss_optimization.py" in command for command in commands
    )
    diroca_checks = {check.name: check for check in diroca_result.checks}
    assert diroca_checks["runtime_help"].status == "SKIP"
    assert "sandbox" in diroca_checks["runtime_help"].detail.lower()


def test_public_runner_records_resource_block_without_repository_access(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke,
        "_resource_block_for",
        lambda name: smoke.ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "fixture hardware gate",
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_ensure_repository",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository must not be accessed")
        ),
    )

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.BLOCKED_BY_HARDWARE
    assert result.phase == "RESOURCE_GATE"
    _assert_seven_artifacts(isolated_project / "artifacts" / "stage0" / "reference_smoke" / "plot")


def test_public_plot_success_combines_repository_and_execution_evidence(
    isolated_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = isolated_project / ".cache" / "third_party" / "plot"
    repo.mkdir(parents=True)
    monkeypatch.setattr(smoke, "_resource_block_for", lambda name: None)
    monkeypatch.setattr(
        smoke,
        "_ensure_repository",
        lambda *args, **kwargs: RepositoryOutcome(
            repository_path=repo,
            observed_commit=EXPECTED_COMMITS["plot"],
            commands=(("git", "rev-parse", "HEAD"),),
            checks=(SmokeCheck("repository", "PASS", "fixed"),),
            stdout="repo",
            duration_seconds=0.1,
        ),
    )
    monkeypatch.setattr(smoke, "_inspect_reference_files", lambda *args: ())
    monkeypatch.setattr(
        smoke,
        "_run_plot",
        lambda *args, **kwargs: _result(SmokeStatus.PARTIAL),
    )

    result = run_reference_smoke("plot")

    assert result.status is SmokeStatus.PARTIAL
    assert result.observed_commit == EXPECTED_COMMITS["plot"]
    assert result.command[0] == ("git", "rev-parse", "HEAD")
    assert result.duration_seconds == pytest.approx(0.35)
    assert result.environment["working_directory"] == str(repo.resolve())
    environment_artifact = (
        isolated_project / "artifacts" / "stage0" / "reference_smoke" / "plot" / "environment.txt"
    )
    saved_environment = json.loads(environment_artifact.read_text(encoding="utf-8"))
    assert saved_environment["working_directory"] == str(repo.resolve())


@pytest.mark.parametrize(
    ("result", "expected_exit"),
    [
        (_result(SmokeStatus.PARTIAL), 0),
        (_result(SmokeStatus.BLOCKED_BY_DEPENDENCY), 0),
        (_result(SmokeStatus.FAILED), 1),
        (_result(SmokeStatus.FAILED, phase="POLICY"), 2),
    ],
)
def test_cli_exit_code_distinguishes_completed_non_smoke_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    result: SmokeResult,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(smoke_cli, "run_reference_smoke", lambda *args, **kwargs: result)

    assert smoke_cli.main(["plot", "--timeout", "10"]) == expected_exit


def test_cli_returns_input_error_for_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_cli,
        "run_reference_smoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad timeout")),
    )

    assert smoke_cli.main(["plot"]) == 2


def test_cli_bootstraps_src_without_installed_project_or_pythonpath() -> None:
    environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    completed = subprocess.run(
        [
            smoke.SAFE_PYTHON,
            str(Path(__file__).resolve().parents[1] / "scripts" / "run_reference_smoke.py"),
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
        timeout=20,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "plot" in completed.stdout
    assert "diroca" in completed.stdout


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
