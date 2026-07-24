from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import psutil
import yaml

from tarca.stage0 import reference_smoke_policy as smoke_policy
from tarca.stage0.executables import resolve_external_executable
from tarca.stage0.reference_smoke_artifacts import write_artifacts
from tarca.stage0.reference_smoke_policy import (
    BUILTIN_POLICIES,
    DIROCA_COMMANDS,
    PLOT_COMMANDS,
    PLOT_COMPONENT_CODE,
    SAFE_PYTHON,
    SHA_PATTERN,
    CommandOutcome,
    ReferenceName,
    ReferencePolicy,
    RepositoryExecutionError,
    RepositoryOutcome,
    RepositoryPolicyError,
    ResourceBlock,
    ResourceEstimate,
    SmokeCheck,
    SmokePolicyError,
    SmokeResult,
    SmokeStatus,
    _aggregate_output,
    classify_diroca_outcomes,
    classify_plot_outcomes,
    evaluate_resource_gate,
    recorded_environment,
    validate_candidate_command,
    validate_public_repository_url,
    validate_relative_parts,
    validate_timeout,
)
from tarca.stage0.reference_smoke_policy import (
    audit_plot_component as _audit_plot_component,
)
from tarca.stage0.reference_smoke_policy import (
    inspect_reference_files as _inspect_reference_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WINDOWS_REPARSE_POINT = 0x400
MAX_LOG_BYTES = smoke_policy.MAX_LOG_BYTES
MAX_TIMEOUT_SECONDS = smoke_policy.MAX_TIMEOUT_SECONDS


def load_reference_policy(name: ReferenceName) -> ReferencePolicy:
    if name not in BUILTIN_POLICIES:
        raise ValueError("name must be exactly one of: plot, diroca.")
    expected = BUILTIN_POLICIES[name]
    manifest_path = PROJECT_ROOT / "third_party_manifest" / "sources.yaml"
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SmokePolicyError(f"Unable to read fixed source manifest: {error}") from error
    if not isinstance(raw, list):
        raise SmokePolicyError("Source manifest must be a YAML list.")
    matches = [entry for entry in raw if isinstance(entry, dict) and entry.get("name") == name]
    if len(matches) != 1:
        raise SmokePolicyError(f"Manifest must contain exactly one {name!r} entry.")
    entry = matches[0]
    comparisons = {
        "repository_url": expected.repository_url,
        "default_branch": expected.default_branch,
        "verified_commit": expected.verified_commit,
        "local_reference_path": expected.local_reference_path,
    }
    for field_name, expected_value in comparisons.items():
        if entry.get(field_name) != expected_value:
            raise SmokePolicyError(f"Manifest {field_name} does not match the fixed {name} policy.")
    validate_public_repository_url(expected.repository_url)
    if SHA_PATTERN.fullmatch(expected.verified_commit) is None:
        raise SmokePolicyError("verified_commit must be exactly 40 lowercase hex characters.")
    validate_relative_parts(
        Path(expected.local_reference_path),
        (".cache", "third_party", name),
        "local_reference_path",
    )
    return expected


def _path_has_reparse_point(path: Path, stop: Path) -> bool:
    stop_resolved = stop.resolve()
    current = path
    while True:
        if current.exists():
            if current.is_symlink():
                return True
            try:
                attributes = getattr(current.lstat(), "st_file_attributes", 0)
            except OSError:
                return True
            if attributes & _WINDOWS_REPARSE_POINT:
                return True
        if current == stop or current.resolve(strict=False) == stop_resolved:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _resolve_strict_root(
    value: Path,
    expected_parts: tuple[str, ...],
    label: str,
) -> Path:
    validate_relative_parts(value, expected_parts, label)
    project_root = PROJECT_ROOT.resolve()
    candidate = PROJECT_ROOT.joinpath(*expected_parts)
    if _path_has_reparse_point(candidate, PROJECT_ROOT):
        raise SmokePolicyError(
            f"{label} contains a symbolic link or reparse point and is not trusted."
        )
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(project_root):
        raise SmokePolicyError(f"{label} escapes PROJECT_ROOT.")
    expected = project_root.joinpath(*expected_parts)
    if resolved != expected:
        raise SmokePolicyError(f"{label} does not resolve to its fixed PROJECT_ROOT path.")
    return candidate


def _validate_cache_root(cache_root: Path) -> Path:
    return _resolve_strict_root(
        cache_root,
        (".cache", "third_party"),
        "cache_root",
    )


def _validate_artifact_root(artifact_root: Path) -> Path:
    return _resolve_strict_root(
        artifact_root,
        ("artifacts", "stage0", "reference_smoke"),
        "artifact_root",
    )


def _resource_block_for(name: ReferenceName) -> ResourceBlock | None:
    estimates = {
        "plot": ResourceEstimate(
            peak_memory_bytes=256 * 1024**2,
            additional_disk_bytes=512 * 1024**2,
            requires_gpu=False,
            expected_runtime_seconds=120,
        ),
        "diroca": ResourceEstimate(
            peak_memory_bytes=384 * 1024**2,
            additional_disk_bytes=512 * 1024**2,
            requires_gpu=False,
            expected_runtime_seconds=120,
        ),
    }
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    return evaluate_resource_gate(
        estimates[name],
        total_memory_bytes=memory.total,
        available_memory_bytes=memory.available,
        free_disk_bytes=disk.free,
    )


def _safe_base_environment() -> dict[str, str]:
    allowed_keys = (
        "PATH",
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    return {key: os.environ[key] for key in allowed_keys if key in os.environ}


def _offline_environment(repository_path: Path | None = None) -> dict[str, str]:
    environment = {
        **_safe_base_environment(),
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
    }
    if repository_path is not None:
        environment["PYTHONPATH"] = str(repository_path)
    return environment


def _run_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> CommandOutcome:
    started = time.monotonic()
    executed_command = command
    try:
        if command and command[0] == "git":
            executed_command = (
                resolve_external_executable("git", PROJECT_ROOT),
                *command[1:],
            )
        completed = subprocess.run(
            executed_command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
            timeout=timeout_seconds,
            env=env,
        )
        return CommandOutcome(
            command=executed_command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return CommandOutcome(
            command=executed_command,
            exit_code=124,
            stdout=stdout,
            stderr=f"{stderr}\nTimed out after {timeout_seconds} seconds.".strip(),
            duration_seconds=time.monotonic() - started,
        )
    except OSError as error:
        return CommandOutcome(
            command=executed_command,
            exit_code=127,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
            duration_seconds=time.monotonic() - started,
        )


def _verify_repo_links(repository_path: Path) -> None:
    root = repository_path.resolve()
    for current_root, directory_names, file_names in os.walk(repository_path):
        current = Path(current_root)
        for name in (*directory_names, *file_names):
            candidate = current / name
            if not candidate.is_symlink() and not (
                getattr(candidate.lstat(), "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                continue
            if not candidate.resolve(strict=False).is_relative_to(root):
                raise RepositoryPolicyError(
                    "Repository contains a symbolic link or junction escaping its root."
                )


def _require_success(
    outcomes: tuple[CommandOutcome, ...],
    label: str,
) -> None:
    failed = next((outcome for outcome in outcomes if outcome.exit_code != 0), None)
    if failed is not None:
        detail = failed.stderr.strip() or failed.stdout.strip() or "no diagnostic output"
        raise RepositoryExecutionError(f"{label} failed: {detail}", outcomes)


def _verify_existing_repository(
    repository_path: Path,
    policy: ReferencePolicy,
    timeout_seconds: int,
) -> RepositoryOutcome:
    _verify_repo_links(repository_path)
    environment = _offline_environment()
    origin = _run_process(
        ("git", "remote", "get-url", "origin"),
        cwd=repository_path,
        timeout_seconds=timeout_seconds,
        env=environment,
    )
    head = _run_process(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_path,
        timeout_seconds=timeout_seconds,
        env=environment,
    )
    _require_success((origin, head), "existing repository verification")
    observed_origin = origin.stdout.strip()
    observed_head = head.stdout.strip().lower()
    if observed_origin != policy.repository_url:
        raise RepositoryPolicyError(
            "Existing repository origin does not exactly match the official URL.",
            (origin, head),
        )
    if observed_head != policy.verified_commit:
        raise RepositoryPolicyError(
            "Existing repository HEAD does not exactly match verified_commit.",
            (origin, head),
        )
    return RepositoryOutcome(
        repository_path=repository_path,
        observed_commit=observed_head,
        commands=(origin.command, head.command),
        checks=(
            SmokeCheck("repository_origin", "PASS", observed_origin),
            SmokeCheck("repository_head", "PASS", observed_head),
        ),
        stdout=origin.stdout + head.stdout,
        stderr=origin.stderr + head.stderr,
        duration_seconds=origin.duration_seconds + head.duration_seconds,
    )


def _validated_temp_clone_path(cache_root: Path, name: str) -> Path:
    temporary = Path(tempfile.mkdtemp(prefix=f".tmp-{name}-", dir=cache_root))
    if temporary.parent != cache_root or not temporary.name.startswith(f".tmp-{name}-"):
        raise RepositoryPolicyError("Temporary clone path escaped the fixed cache root.")
    return temporary


def _remove_owned_temporary_clone(temporary: Path, cache_root: Path, name: str) -> None:
    if (
        temporary.parent == cache_root
        and temporary.name.startswith(f".tmp-{name}-")
        and temporary.exists()
    ):
        shutil.rmtree(temporary)


def _clone_fixed_repository(
    repository_path: Path,
    cache_root: Path,
    policy: ReferencePolicy,
    timeout_seconds: int,
) -> RepositoryOutcome:
    temporary = _validated_temp_clone_path(cache_root, policy.name)
    commands: list[tuple[str, ...]] = []
    outcomes: list[CommandOutcome] = []
    environment = _offline_environment()
    try:
        clone_commands = (
            ("git", "init", "--initial-branch", policy.default_branch),
            ("git", "remote", "add", "origin", policy.repository_url),
            (
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                policy.verified_commit,
            ),
            ("git", "checkout", "--detach", policy.verified_commit),
            ("git", "remote", "get-url", "origin"),
            ("git", "rev-parse", "HEAD"),
        )
        for command in clone_commands:
            outcome = _run_process(
                command,
                cwd=temporary,
                timeout_seconds=timeout_seconds,
                env=environment,
            )
            commands.append(command)
            outcomes.append(outcome)
            _require_success(tuple(outcomes), "fixed-commit clone")
        observed_origin = outcomes[-2].stdout.strip()
        observed_head = outcomes[-1].stdout.strip().lower()
        if observed_origin != policy.repository_url:
            raise RepositoryPolicyError(
                "Fetched repository origin verification failed.",
                tuple(outcomes),
            )
        if observed_head != policy.verified_commit:
            raise RepositoryPolicyError(
                "Fetched repository commit verification failed.",
                tuple(outcomes),
            )
        _verify_repo_links(temporary)
        if repository_path.exists():
            raise RepositoryPolicyError(
                "Destination appeared during clone; refusing to overwrite local content."
            )
        temporary.replace(repository_path)
        return RepositoryOutcome(
            repository_path=repository_path,
            observed_commit=observed_head,
            commands=tuple(commands),
            checks=(
                SmokeCheck("repository_origin", "PASS", observed_origin),
                SmokeCheck("repository_head", "PASS", observed_head),
                SmokeCheck(
                    "clone_policy",
                    "PASS",
                    "Fetched one fixed commit without dependency installation.",
                ),
            ),
            stdout="".join(outcome.stdout for outcome in outcomes),
            stderr="".join(outcome.stderr for outcome in outcomes),
            duration_seconds=sum(outcome.duration_seconds for outcome in outcomes),
        )
    finally:
        _remove_owned_temporary_clone(temporary, cache_root, policy.name)


def _ensure_repository(
    policy: ReferencePolicy,
    cache_root: Path,
    timeout_seconds: int,
) -> RepositoryOutcome:
    validated_cache = _validate_cache_root(cache_root)
    validated_cache.mkdir(parents=True, exist_ok=True)
    if _path_has_reparse_point(validated_cache, PROJECT_ROOT):
        raise RepositoryPolicyError("Cache root became a symbolic link or reparse point.")
    repository_path = validated_cache / policy.name
    if repository_path.exists():
        if (
            repository_path.is_symlink()
            or getattr(repository_path.lstat(), "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
        ):
            raise RepositoryPolicyError("Repository path is a symbolic link or junction.")
        if not repository_path.is_dir() or not (repository_path / ".git").exists():
            raise RepositoryPolicyError(
                "Existing cache path is not a Git repository; refusing to overwrite it."
            )
        return _verify_existing_repository(repository_path, policy, timeout_seconds)
    return _clone_fixed_repository(
        repository_path,
        validated_cache,
        policy,
        timeout_seconds,
    )


def _run_plot(
    repository_path: Path,
    timeout_seconds: int,
    commit: str,
    checks: tuple[SmokeCheck, ...],
) -> SmokeResult:
    compile_command = validate_candidate_command(
        (SAFE_PYTHON, "-m", "compileall", "-q", "experiments"),
        PLOT_COMMANDS,
    )
    compile_outcome = _run_process(
        compile_command,
        cwd=repository_path,
        timeout_seconds=timeout_seconds,
        env=_offline_environment(repository_path),
    )
    updated_checks = (
        *checks,
        SmokeCheck(
            "static_compile",
            "PASS" if compile_outcome.exit_code == 0 else "FAIL",
            f"exit_code={compile_outcome.exit_code}",
        ),
    )
    if compile_outcome.exit_code != 0:
        return classify_plot_outcomes((compile_outcome,), commit, updated_checks)
    audit = _audit_plot_component(repository_path)
    updated_checks += (audit,)
    if audit.status != "PASS":
        return classify_plot_outcomes((compile_outcome,), commit, updated_checks)
    component_command = validate_candidate_command(
        (SAFE_PYTHON, "-c", PLOT_COMPONENT_CODE),
        PLOT_COMMANDS,
    )
    component_outcome = _run_process(
        component_command,
        cwd=repository_path,
        timeout_seconds=timeout_seconds,
        env=_offline_environment(repository_path),
    )
    updated_checks += (
        SmokeCheck(
            "binary_addition_transport",
            "PASS" if component_outcome.exit_code == 0 else "FAIL",
            f"exit_code={component_outcome.exit_code}; CPU-only audited primitive",
        ),
        SmokeCheck(
            "end_to_end_plot",
            "SKIP",
            "HEQ training, MCQA, Gemma, Slurm, sweeps, and full experiments are prohibited.",
        ),
    )
    return classify_plot_outcomes(
        (compile_outcome, component_outcome),
        commit,
        updated_checks,
    )


def _run_diroca(
    repository_path: Path,
    timeout_seconds: int,
    commit: str,
    checks: tuple[SmokeCheck, ...],
) -> SmokeResult:
    command = validate_candidate_command(
        (SAFE_PYTHON, "-m", "compileall", "-q", "."),
        DIROCA_COMMANDS,
    )
    outcome = _run_process(
        command,
        cwd=repository_path,
        timeout_seconds=timeout_seconds,
        env=_offline_environment(repository_path),
    )
    updated_checks = (
        *checks,
        SmokeCheck(
            "static_compile",
            "PASS" if outcome.exit_code == 0 else "FAIL",
            f"exit_code={outcome.exit_code}",
        ),
        SmokeCheck(
            "runtime_help",
            "SKIP",
            "Third-party scripts have unaudited top-level imports and no enforced OS/network "
            "sandbox is available; --help execution is not considered safe.",
        ),
        SmokeCheck(
            "tiny_optimization",
            "SKIP",
            "Official YAML uses eval and no safely verified expected-output path was run.",
        ),
        SmokeCheck(
            "documentation_consistency",
            "WARN",
            "README CLI and script arguments differ; requirements names ot instead of POT.",
        ),
    )
    return classify_diroca_outcomes((outcome,), commit, updated_checks)


def _failure_result(
    name: ReferenceName,
    *,
    status: SmokeStatus,
    expected_commit: str,
    observed_commit: str = "",
    phase: str,
    reason: str,
    policy_error: bool = False,
    command: tuple[tuple[str, ...], ...] = (),
    checks: tuple[SmokeCheck, ...] = (),
    stdout: str = "",
    stderr: str = "",
    duration_seconds: float = 0.0,
    exit_code: int | None = None,
) -> SmokeResult:
    return SmokeResult(
        name=name,
        status=status,
        command=command,
        commit=observed_commit or expected_commit,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        used_gpu=False,
        expected_commit=expected_commit,
        observed_commit=observed_commit,
        phase=phase,
        reason=reason,
        checks=checks,
        policy_error=policy_error,
        stdout=stdout,
        stderr=stderr,
        environment=recorded_environment(),
    )


def _repository_error_evidence(
    outcomes: tuple[CommandOutcome, ...],
) -> dict[str, object]:
    failed = next(
        (outcome for outcome in reversed(outcomes) if outcome.exit_code != 0),
        None,
    )
    return {
        "command": tuple(outcome.command for outcome in outcomes),
        "stdout": _aggregate_output(outcomes, "stdout"),
        "stderr": _aggregate_output(outcomes, "stderr"),
        "duration_seconds": sum(outcome.duration_seconds for outcome in outcomes),
        "exit_code": failed.exit_code if failed is not None else None,
    }


def _is_network_git_failure(outcomes: tuple[CommandOutcome, ...]) -> bool:
    diagnostic = "\n".join(f"{outcome.stdout}\n{outcome.stderr}".lower() for outcome in outcomes)
    return any(
        marker in diagnostic
        for marker in (
            "network",
            "timed out",
            "could not resolve",
            "failed to connect",
            "unable to access",
            "connection reset",
        )
    )


def _canonical_artifact_directory(name: ReferenceName) -> Path:
    return PROJECT_ROOT / "artifacts" / "stage0" / "reference_smoke" / name


def run_reference_smoke(
    name: ReferenceName,
    cache_root: Path = Path(".cache/third_party"),
    artifact_root: Path = Path("artifacts/stage0/reference_smoke"),
    timeout_seconds: int = 60,
) -> SmokeResult:
    if name not in BUILTIN_POLICIES:
        raise ValueError("name must be exactly one of: plot, diroca.")
    validate_timeout(timeout_seconds)
    expected = BUILTIN_POLICIES[name]
    target = _canonical_artifact_directory(name)
    try:
        validated_artifact_root = _validate_artifact_root(artifact_root)
        target = validated_artifact_root / name
        _validate_cache_root(cache_root)
        policy = load_reference_policy(name)
    except SmokePolicyError as error:
        result = _failure_result(
            name,
            status=SmokeStatus.FAILED,
            expected_commit=expected.verified_commit,
            phase="POLICY",
            reason=str(error),
            policy_error=True,
            checks=(SmokeCheck("policy", "FAIL", str(error)),),
        )
        write_artifacts(target, result)
        return result

    try:
        resource_block = _resource_block_for(name)
        if resource_block is not None:
            result = _failure_result(
                name,
                status=resource_block.status,
                expected_commit=policy.verified_commit,
                phase="RESOURCE_GATE",
                reason=resource_block.reason,
                checks=(SmokeCheck("resource_gate", "BLOCKED", resource_block.reason),),
            )
            write_artifacts(target, result)
            return result

        repository = _ensure_repository(policy, cache_root, timeout_seconds)
        inspection_checks = _inspect_reference_files(
            policy,
            repository.repository_path,
        )
        if name == "plot":
            execution = _run_plot(
                repository.repository_path,
                timeout_seconds,
                policy.verified_commit,
                inspection_checks,
            )
        else:
            execution = _run_diroca(
                repository.repository_path,
                timeout_seconds,
                policy.verified_commit,
                inspection_checks,
            )
        result = replace(
            execution,
            command=repository.commands + execution.command,
            expected_commit=policy.verified_commit,
            observed_commit=repository.observed_commit,
            commit=repository.observed_commit,
            checks=repository.checks + execution.checks,
            stdout="\n".join(part for part in (repository.stdout, execution.stdout) if part),
            stderr="\n".join(part for part in (repository.stderr, execution.stderr) if part),
            duration_seconds=repository.duration_seconds + execution.duration_seconds,
            environment={
                **execution.environment,
                "working_directory": str(repository.repository_path.resolve()),
            },
        )
    except RepositoryExecutionError as error:
        evidence = _repository_error_evidence(error.outcomes)
        is_network = _is_network_git_failure(error.outcomes)
        failure_kind = "NETWORK_FAILURE" if is_network else "GIT_FAILURE"
        result = _failure_result(
            name,
            status=SmokeStatus.FAILED,
            expected_commit=policy.verified_commit,
            phase="REPOSITORY_NETWORK" if is_network else "REPOSITORY_GIT",
            reason=f"{failure_kind}: {error}",
            checks=(SmokeCheck("repository", "FAIL", f"{failure_kind}: {error}"),),
            **evidence,
        )
    except RepositoryPolicyError as error:
        evidence = _repository_error_evidence(error.outcomes)
        result = _failure_result(
            name,
            status=SmokeStatus.FAILED,
            expected_commit=policy.verified_commit,
            phase="REPOSITORY",
            reason=str(error),
            policy_error=True,
            checks=(SmokeCheck("repository", "FAIL", str(error)),),
            **evidence,
        )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        result = _failure_result(
            name,
            status=SmokeStatus.FAILED,
            expected_commit=policy.verified_commit,
            phase="EXCEPTION",
            reason=reason,
            checks=(SmokeCheck("unexpected_exception", "FAIL", reason),),
        )
    write_artifacts(target, result)
    return result
