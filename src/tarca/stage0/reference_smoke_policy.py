from __future__ import annotations

import ast
import hashlib
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

ReferenceName = Literal["plot", "diroca"]
SAFE_PYTHON = sys.executable
MAX_TIMEOUT_SECONDS = 300
MAX_LOG_BYTES = 65_536
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_COMMAND_PATTERN = re.compile(
    r"(?i)(?<![a-z])(?:mcqa|gemma|slurm|sweep|cuda|gpu|pip|conda|uv|"
    r"install|download|train(?:ing)?)(?![a-z])"
)

PLOT_COMPONENT_CODE = (
    "import torch; "
    "from experiments.binary_addition.transport import sinkhorn_uniform_ot; "
    "c=torch.tensor([[0.,1.],[1.,0.]], device='cpu'); "
    "p=sinkhorn_uniform_ot(c,0.1,30); "
    "assert p.shape==(2,2) and torch.isfinite(p).all(); "
    "print(p)"
)
PLOT_COMMANDS = frozenset(
    {
        (SAFE_PYTHON, "-m", "compileall", "-q", "experiments"),
        (SAFE_PYTHON, "-c", PLOT_COMPONENT_CODE),
    }
)
DIROCA_COMMANDS = frozenset(
    {
        (SAFE_PYTHON, "-m", "compileall", "-q", "."),
        (SAFE_PYTHON, "generate_data.py", "--help"),
        (SAFE_PYTHON, "gauss_optimization.py", "--help"),
    }
)


class SmokeStatus(StrEnum):
    IMPORT_ONLY = "IMPORT_ONLY"
    SMOKE_PASSED = "SMOKE_PASSED"
    PARTIAL = "PARTIAL"
    BLOCKED_BY_HARDWARE = "BLOCKED_BY_HARDWARE"
    BLOCKED_BY_DEPENDENCY = "BLOCKED_BY_DEPENDENCY"
    FAILED = "FAILED"


class SmokePolicyError(ValueError):
    """An input violated the fixed Stage 0 execution policy."""


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class CommandOutcome:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


class RepositoryPolicyError(SmokePolicyError):
    """Existing or fetched repository state is not trustworthy."""

    def __init__(
        self,
        message: str,
        outcomes: tuple[CommandOutcome, ...] = (),
    ) -> None:
        super().__init__(message)
        self.outcomes = outcomes


class RepositoryExecutionError(RuntimeError):
    """A bounded Git command failed; executed evidence remains attached."""

    def __init__(
        self,
        message: str,
        outcomes: tuple[CommandOutcome, ...],
    ) -> None:
        super().__init__(message)
        self.outcomes = outcomes


@dataclass(frozen=True)
class ReferencePolicy:
    name: ReferenceName
    repository_url: str
    default_branch: str
    verified_commit: str
    local_reference_path: str


@dataclass(frozen=True)
class RepositoryOutcome:
    repository_path: Path
    observed_commit: str
    commands: tuple[tuple[str, ...], ...]
    checks: tuple[SmokeCheck, ...]
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class ResourceEstimate:
    peak_memory_bytes: int
    additional_disk_bytes: int
    requires_gpu: bool
    expected_runtime_seconds: int


@dataclass(frozen=True)
class ResourceBlock:
    status: SmokeStatus
    reason: str


def _empty_environment() -> MappingProxyType[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class SmokeResult:
    name: ReferenceName
    status: SmokeStatus
    command: tuple[tuple[str, ...], ...]
    commit: str
    exit_code: int | None
    duration_seconds: float
    used_gpu: bool
    expected_commit: str
    observed_commit: str
    phase: str
    reason: str
    checks: tuple[SmokeCheck, ...]
    policy_error: bool = False
    stdout: str = ""
    stderr: str = ""
    environment: Mapping[str, str] = field(default_factory=_empty_environment)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(tuple(item) for item in self.command))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        if self.used_gpu:
            raise ValueError("Stage 0 reference smoke must never use a GPU.")
        if self.status not in SmokeStatus:
            raise ValueError("Unknown smoke status.")


BUILTIN_POLICIES: Mapping[str, ReferencePolicy] = MappingProxyType(
    {
        "plot": ReferencePolicy(
            name="plot",
            repository_url="https://github.com/jchang153/causal-abstractions-ot",
            default_branch="main",
            verified_commit="96dbec5f04bc03aea6e55c430eeafd5c9be27fb2",
            local_reference_path=".cache/third_party/plot",
        ),
        "diroca": ReferencePolicy(
            name="diroca",
            repository_url="https://github.com/yfelekis/DiRoCA",
            default_branch="main",
            verified_commit="7002947b4954abea1f3d11fcb6f36e7f3c43e8bd",
            local_reference_path=".cache/third_party/diroca",
        ),
    }
)


def validate_timeout(timeout_seconds: int) -> int:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or timeout_seconds > MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(f"timeout_seconds must be an integer from 1 to {MAX_TIMEOUT_SECONDS}.")
    return timeout_seconds


def validate_candidate_command(
    command: tuple[str, ...],
    allowlist: set[tuple[str, ...]] | frozenset[tuple[str, ...]],
) -> tuple[str, ...]:
    normalized = tuple(str(part) for part in command)
    if normalized not in allowlist:
        raise SmokePolicyError("Command is not in the exact internal allowlist.")
    if FORBIDDEN_COMMAND_PATTERN.search(" ".join(normalized)) is not None:
        raise SmokePolicyError("Command contains a forbidden Stage 0 operation.")
    return normalized


def validate_public_repository_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or len([part for part in parsed.path.split("/") if part]) != 2
    ):
        raise SmokePolicyError("repository_url is not a credential-free official GitHub URL.")


def validate_relative_parts(
    value: Path,
    expected_parts: tuple[str, ...],
    label: str,
) -> None:
    if value.is_absolute() or ".." in value.parts or value.parts != expected_parts:
        raise SmokePolicyError(
            f"{label} must be the exact project-relative path {'/'.join(expected_parts)}."
        )


def evaluate_resource_gate(
    estimate: ResourceEstimate,
    *,
    total_memory_bytes: int,
    available_memory_bytes: int,
    free_disk_bytes: int,
) -> ResourceBlock | None:
    if estimate.requires_gpu:
        return ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "The candidate requires a GPU, which Stage 0 forbids.",
        )
    if estimate.peak_memory_bytes > total_memory_bytes * 0.70:
        return ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "Estimated peak memory exceeds 70% of physical memory.",
        )
    if estimate.peak_memory_bytes > available_memory_bytes:
        return ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "Current available memory is below the bounded smoke estimate.",
        )
    if estimate.additional_disk_bytes > free_disk_bytes * 0.60:
        return ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "Estimated disk use exceeds 60% of currently free space.",
        )
    if estimate.expected_runtime_seconds > 7_200:
        return ResourceBlock(
            SmokeStatus.BLOCKED_BY_HARDWARE,
            "Estimated runtime exceeds the two-hour hard gate.",
        )
    return None


def recorded_environment() -> MappingProxyType[str, str]:
    return MappingProxyType(
        {
            "python_executable": SAFE_PYTHON,
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "used_gpu": "false",
            "dependency_installation": "prohibited",
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file_summary(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"sha256={_sha256_file(path)}; first_nonempty_line={first_line[:160]}"


def inspect_reference_files(
    policy: ReferencePolicy,
    repository_path: Path,
) -> tuple[SmokeCheck, ...]:
    checks: list[SmokeCheck] = []
    readme = next(
        (
            candidate
            for candidate in (
                repository_path / "README.md",
                repository_path / "README.rst",
                repository_path / "readme.md",
            )
            if candidate.is_file()
        ),
        None,
    )
    requirements = repository_path / "requirements.txt"
    if readme is None:
        checks.append(SmokeCheck("README", "FAIL", "No top-level README found."))
    else:
        checks.append(SmokeCheck("README", "PASS", _safe_file_summary(readme)))
    if requirements.is_file():
        checks.append(SmokeCheck("requirements", "PASS", _safe_file_summary(requirements)))
    else:
        checks.append(SmokeCheck("requirements", "WARN", "No top-level requirements.txt found."))
    if policy.name == "diroca":
        config_files = sorted((repository_path / "configs").glob("*.yaml"))
        if config_files:
            joined = "; ".join(f"{path.name}:{_sha256_file(path)}" for path in config_files[:20])
            checks.append(SmokeCheck("configs", "PASS", joined))
        else:
            checks.append(SmokeCheck("configs", "WARN", "No YAML config found."))
        checks.append(
            SmokeCheck(
                "yaml_execution",
                "SKIP",
                "Untrusted YAML/eval execution is prohibited in Stage 0.",
            )
        )
    return tuple(checks)


_PLOT_ALLOWED_IMPORTS = {
    "__future__",
    "data",
    "dataclasses",
    "features",
    "interventions",
    "math",
    "model",
    "pca_basis",
    "random",
    "scm",
    "sites",
    "torch",
    "typing",
}
_PLOT_LOCAL_MODULES = {
    "data",
    "features",
    "interventions",
    "model",
    "pca_basis",
    "scm",
    "sites",
    "transport",
}
_PLOT_FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "__import__", "input"}


def _audit_plot_module(
    path: Path,
) -> tuple[set[str], bool, str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as error:
        return set(), False, f"{path.name}: unable to parse: {type(error).__name__}: {error}"

    imported_roots: set[str] = set()
    local_imports: set[str] = set()
    function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                imported_roots.add(root)
                if node.level > 0 and root in _PLOT_LOCAL_MODULES:
                    local_imports.add(root)
            elif node.level > 0:
                local_imports.update(
                    alias.name for alias in node.names if alias.name in _PLOT_LOCAL_MODULES
                )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function_found = function_found or node.name == "sinkhorn_uniform_ot"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _PLOT_FORBIDDEN_CALLS
        ):
            return (
                set(),
                function_found,
                f"{path.name}: forbidden dynamic call {node.func.id!r} found.",
            )

    unexpected_imports = imported_roots - _PLOT_ALLOWED_IMPORTS
    if unexpected_imports:
        return (
            set(),
            function_found,
            f"{path.name}: unexpected imports: {', '.join(sorted(unexpected_imports))}.",
        )
    return local_imports, function_found, None


def audit_plot_component(repository_path: Path) -> SmokeCheck:
    component_root = repository_path / "experiments" / "binary_addition"
    component = component_root / "transport.py"
    if not component.is_file():
        return SmokeCheck(
            "plot_component_audit",
            "SKIP",
            "Expected binary_addition/transport.py was not found.",
        )

    pending = ["transport"]
    audited: dict[str, str] = {}
    function_found = False
    while pending:
        module_name = pending.pop()
        if module_name in audited:
            continue
        path = component_root / f"{module_name}.py"
        if not path.is_file():
            continue
        local_imports, found_here, error = _audit_plot_module(path)
        if error is not None:
            return SmokeCheck(
                "plot_component_audit",
                "FAIL",
                error,
            )
        audited[module_name] = _sha256_file(path)
        if module_name == "transport":
            function_found = found_here
        pending.extend(sorted(local_imports - audited.keys()))

    if not function_found:
        return SmokeCheck(
            "plot_component_audit",
            "FAIL",
            "sinkhorn_uniform_ot was not found.",
        )
    return SmokeCheck(
        "plot_component_audit",
        "PASS",
        "Recursive AST import-closure audit passed; "
        + "; ".join(f"{name}.py:{digest}" for name, digest in sorted(audited.items())),
    )


def _dependency_failure(outcome: CommandOutcome) -> bool:
    diagnostic = f"{outcome.stdout}\n{outcome.stderr}".lower()
    return any(
        marker in diagnostic
        for marker in (
            "modulenotfounderror",
            "no module named",
            "importerror",
            "missing dependency",
        )
    )


def _aggregate_output(outcomes: tuple[CommandOutcome, ...], field_name: str) -> str:
    sections: list[str] = []
    for outcome in outcomes:
        value = getattr(outcome, field_name)
        if value:
            sections.append(f"$ {subprocess.list2cmdline(outcome.command)}\n{value.rstrip()}")
    return "\n\n".join(sections)


def classify_plot_outcomes(
    outcomes: tuple[CommandOutcome, ...],
    commit: str,
    checks: tuple[SmokeCheck, ...],
) -> SmokeResult:
    audit_failed = any(check.status == "FAIL" for check in checks)
    if not outcomes:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "STATIC",
            "No PLOT checks were executed.",
            None,
        )
    elif outcomes[0].exit_code != 0:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "STATIC",
            "PLOT static compilation failed.",
            outcomes[0].exit_code,
        )
    elif audit_failed:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "COMPONENT",
            "PLOT component audit failed before any runtime smoke was allowed to proceed.",
            outcomes[0].exit_code,
        )
    elif len(outcomes) == 1:
        status, phase, reason, exit_code = (
            SmokeStatus.IMPORT_ONLY,
            "STATIC",
            "Static compilation passed; no audited runtime component was executed.",
            outcomes[0].exit_code,
        )
    elif outcomes[-1].exit_code == 0:
        status, phase, reason, exit_code = (
            SmokeStatus.PARTIAL,
            "COMPONENT",
            "Static compilation and one audited pure-CPU transport primitive passed; "
            "this is not an end-to-end PLOT reproduction.",
            outcomes[-1].exit_code,
        )
    elif _dependency_failure(outcomes[-1]):
        status, phase, reason, exit_code = (
            SmokeStatus.BLOCKED_BY_DEPENDENCY,
            "COMPONENT",
            "The audited component could not import a required dependency.",
            outcomes[-1].exit_code,
        )
    else:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "COMPONENT",
            "The audited PLOT component failed.",
            outcomes[-1].exit_code,
        )
    return _classified_result("plot", status, phase, reason, exit_code, outcomes, commit, checks)


def classify_diroca_outcomes(
    outcomes: tuple[CommandOutcome, ...],
    commit: str,
    checks: tuple[SmokeCheck, ...],
) -> SmokeResult:
    failed = next((outcome for outcome in outcomes if outcome.exit_code != 0), None)
    if not outcomes:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "STATIC",
            "No DiRoCA checks were executed.",
            None,
        )
    elif outcomes[0].exit_code != 0:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "STATIC",
            "DiRoCA static compilation failed.",
            outcomes[0].exit_code,
        )
    elif failed is not None and _dependency_failure(failed):
        status, phase, reason, exit_code = (
            SmokeStatus.BLOCKED_BY_DEPENDENCY,
            "HELP",
            "DiRoCA CLI import was blocked by an unavailable dependency; "
            "the TARCA environment was not modified.",
            failed.exit_code,
        )
    elif failed is not None:
        status, phase, reason, exit_code = (
            SmokeStatus.FAILED,
            "HELP",
            "A bounded DiRoCA --help command failed.",
            failed.exit_code,
        )
    elif len(outcomes) >= 3:
        status, phase, reason, exit_code = (
            SmokeStatus.IMPORT_ONLY,
            "HELP",
            "Static compilation and two --help checks passed. No YAML, optimization, "
            "or expected scientific output was executed.",
            outcomes[-1].exit_code,
        )
    else:
        status, phase, reason, exit_code = (
            SmokeStatus.IMPORT_ONLY,
            "STATIC",
            "Only static/import-level DiRoCA checks completed.",
            outcomes[-1].exit_code,
        )
    return _classified_result("diroca", status, phase, reason, exit_code, outcomes, commit, checks)


def _classified_result(
    name: ReferenceName,
    status: SmokeStatus,
    phase: str,
    reason: str,
    exit_code: int | None,
    outcomes: tuple[CommandOutcome, ...],
    commit: str,
    checks: tuple[SmokeCheck, ...],
) -> SmokeResult:
    return SmokeResult(
        name=name,
        status=status,
        command=tuple(outcome.command for outcome in outcomes),
        commit=commit,
        exit_code=exit_code,
        duration_seconds=sum(outcome.duration_seconds for outcome in outcomes),
        used_gpu=False,
        expected_commit=commit,
        observed_commit=commit,
        phase=phase,
        reason=reason,
        checks=checks,
        stdout=_aggregate_output(outcomes, "stdout"),
        stderr=_aggregate_output(outcomes, "stderr"),
        environment=recorded_environment(),
    )
