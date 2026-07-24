"""CPU-only diagnostics for the TARCA Stage 0 research environment."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import ot
import psutil
import torch

from tarca.stage0.models import CheckResult, DoctorReport

_REMEDIATIONS = {
    "system": "Verify the isolated Conda environment and rerun python scripts/doctor.py.",
    "system.os": "Run the doctor on a supported local operating system.",
    "system.python": (
        "Activate D:\\software\\MyAnaconda\\envs\\tarca-stage0 and verify Python is >=3.11,<3.12."
    ),
    "system.cpu": "Verify the operating system exposes at least one CPU core.",
    "system.memory": "Close memory-heavy programs and rerun the doctor.",
    "system.gpu": "No action is required for the CPU-only Stage 0 workflow.",
    "system.cuda": "No action is required; Stage 0 intentionally uses CPU PyTorch.",
    "system.disk": "Free disk space on the project volume and rerun the doctor.",
    "system.project_write": (
        "Grant write permission to the project directory and rerun the doctor."
    ),
    "system.git": "Install Git, add it to PATH, and rerun git --version.",
    "system.uv": (
        "Install uv in D:\\software\\MyAnaconda\\envs\\tarca-stage0 and rerun uv --version."
    ),
    "numeric": "Run pytest tests/test_doctor.py -q in the isolated Conda environment.",
    "numeric.reproducibility": (
        "Run pytest tests/test_reproducibility.py -q in the isolated Conda environment."
    ),
    "numeric.pot": (
        "Run pytest tests/test_pot_smoke.py -q and verify the locked POT installation."
    ),
    "numeric.torch_hook": ("Run pytest tests/test_torch_hook_smoke.py -q and verify CPU PyTorch."),
    "numeric.pyvene": (
        "Verify pyvene imports in the isolated Conda environment without downloading a model."
    ),
}


def _pass_or_fail(
    *,
    name: str,
    passed: bool,
    details: dict[str, Any],
    remediation: str,
) -> CheckResult:
    return CheckResult(
        name=name,
        status="PASS" if passed else "FAIL",
        details=details,
        remediation=None if passed else remediation,
    )


def _exception_result(name: str, exc: Exception, remediation: str) -> CheckResult:
    return CheckResult(
        name=name,
        status="FAIL",
        details={
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
        remediation=remediation,
    )


def _safe_result(
    name: str,
    remediation: str,
    checker: Callable[[], CheckResult],
) -> CheckResult:
    try:
        return checker()
    except Exception as exc:
        return _exception_result(name, exc, remediation)


def _check_os() -> CheckResult:
    details = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    return _pass_or_fail(
        name="system.os",
        passed=bool(details["system"]),
        details=details,
        remediation=_REMEDIATIONS["system.os"],
    )


def _check_python() -> CheckResult:
    version = sys.version_info
    supported = version.major == 3 and version.minor == 11
    return _pass_or_fail(
        name="system.python",
        passed=supported,
        details={
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "required": ">=3.11,<3.12",
        },
        remediation=_REMEDIATIONS["system.python"],
    )


def _check_cpu() -> CheckResult:
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    return _pass_or_fail(
        name="system.cpu",
        passed=logical is not None and logical > 0,
        details={
            "processor": platform.processor() or "UNAVAILABLE",
            "physical_cores": physical,
            "logical_cores": logical,
            "architecture": platform.machine(),
        },
        remediation=_REMEDIATIONS["system.cpu"],
    )


def _check_memory() -> CheckResult:
    memory = psutil.virtual_memory()
    return _pass_or_fail(
        name="system.memory",
        passed=memory.total > 0 and memory.available > 0,
        details={
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent_used": memory.percent,
        },
        remediation=_REMEDIATIONS["system.memory"],
    )


def _check_gpu() -> CheckResult:
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    if not available:
        return CheckResult(
            name="system.gpu",
            status="SKIP",
            details={
                "available": False,
                "device_count": count,
                "reason": "No CUDA GPU is exposed; Stage 0 is CPU-only.",
            },
            remediation=_REMEDIATIONS["system.gpu"],
        )
    devices = tuple(torch.cuda.get_device_name(index) for index in range(count))
    return CheckResult(
        name="system.gpu",
        status="PASS",
        details={"available": True, "device_count": count, "devices": devices},
    )


def _check_cuda() -> CheckResult:
    compiled_version = torch.version.cuda
    available = bool(torch.cuda.is_available())
    if compiled_version is None or not available:
        return CheckResult(
            name="system.cuda",
            status="SKIP",
            details={
                "torch_cuda_build": compiled_version,
                "runtime_available": available,
                "reason": "CPU-only PyTorch is expected for Stage 0.",
            },
            remediation=_REMEDIATIONS["system.cuda"],
        )
    return CheckResult(
        name="system.cuda",
        status="PASS",
        details={
            "torch_cuda_build": compiled_version,
            "runtime_available": True,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    )


def _check_disk(project_root: Path) -> CheckResult:
    usage = shutil.disk_usage(project_root)
    return _pass_or_fail(
        name="system.disk",
        passed=usage.total > 0 and usage.free > 0,
        details={
            "path": str(project_root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        remediation=_REMEDIATIONS["system.disk"],
    )


def _check_project_write(project_root: Path) -> CheckResult:
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".tarca-doctor-",
            suffix=".tmp",
            dir=project_root,
            delete=False,
        ) as probe:
            probe.write(b"tarca-stage0-write-probe\n")
            probe.flush()
            os.fsync(probe.fileno())
            probe_path = Path(probe.name)
        size = probe_path.stat().st_size
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
    return _pass_or_fail(
        name="system.project_write",
        passed=size > 0,
        details={
            "path": str(project_root),
            "write_probe_bytes": size,
            "probe_removed": not probe_path.exists(),
        },
        remediation=_REMEDIATIONS["system.project_write"],
    )


def _run_version_command(executable: str, argument: str = "--version") -> dict[str, Any]:
    completed = subprocess.run(
        [executable, argument],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=10,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"{Path(executable).name} exited {completed.returncode}: {output}")
    return {
        "executable": executable,
        "version_output": output,
        "exit_code": completed.returncode,
    }


def _check_git() -> CheckResult:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("git was not found on PATH")
    return CheckResult(
        name="system.git",
        status="PASS",
        details=_run_version_command(executable),
    )


def _find_uv() -> Path | None:
    candidates = (
        Path(sys.executable).parent / "uv.exe",
        Path(sys.executable).parent / "Scripts" / "uv.exe",
        Path(sys.executable).parent / "bin" / "uv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("uv")
    return Path(discovered) if discovered else None


def _check_uv() -> CheckResult:
    executable = _find_uv()
    if executable is None:
        raise FileNotFoundError("uv was not found in the active environment or on PATH")
    return CheckResult(
        name="system.uv",
        status="PASS",
        details=_run_version_command(str(executable)),
    )


def check_system(project_root: Path) -> tuple[CheckResult, ...]:
    """Check the local system without requiring a GPU or network access."""
    resolved_root = project_root.resolve(strict=True)
    checks: tuple[tuple[str, Callable[[], CheckResult]], ...] = (
        ("system.os", _check_os),
        ("system.python", _check_python),
        ("system.cpu", _check_cpu),
        ("system.memory", _check_memory),
        ("system.gpu", _check_gpu),
        ("system.cuda", _check_cuda),
        ("system.disk", lambda: _check_disk(resolved_root)),
        ("system.project_write", lambda: _check_project_write(resolved_root)),
        ("system.git", _check_git),
        ("system.uv", _check_uv),
    )
    return tuple(_safe_result(name, _REMEDIATIONS[name], checker) for name, checker in checks)


def _check_numpy_dtype(dtype: type[np.floating[Any]]) -> CheckResult:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    output = values @ values.T
    dtype_name = np.dtype(dtype).name
    return _pass_or_fail(
        name=f"numeric.numpy.{dtype_name}",
        passed=output.dtype == np.dtype(dtype) and bool(np.isfinite(output).all()),
        details={
            "dtype": str(output.dtype),
            "shape": output.shape,
            "finite": bool(np.isfinite(output).all()),
            "sum": float(output.sum()),
        },
        remediation=_REMEDIATIONS["numeric"],
    )


def _check_torch_dtype(dtype: torch.dtype) -> CheckResult:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device="cpu")
    output = values @ values.T
    dtype_name = str(dtype).removeprefix("torch.")
    return _pass_or_fail(
        name=f"numeric.torch.{dtype_name}",
        passed=output.dtype == dtype and bool(torch.isfinite(output).all().item()),
        details={
            "dtype": str(output.dtype),
            "device": str(output.device),
            "shape": tuple(output.shape),
            "finite": bool(torch.isfinite(output).all().item()),
            "sum": float(output.sum().item()),
        },
        remediation=_REMEDIATIONS["numeric"],
    )


def _check_cpu_matmul() -> CheckResult:
    left = torch.arange(12, dtype=torch.float64, device="cpu").reshape(3, 4)
    right = torch.arange(8, dtype=torch.float64, device="cpu").reshape(4, 2)
    output = left @ right
    expected = torch.tensor(
        [[28.0, 34.0], [76.0, 98.0], [124.0, 162.0]],
        dtype=torch.float64,
    )
    matches = bool(torch.equal(output, expected))
    return _pass_or_fail(
        name="numeric.cpu_matmul",
        passed=matches and bool(torch.isfinite(output).all().item()),
        details={
            "device": str(output.device),
            "shape": tuple(output.shape),
            "matches_expected": matches,
            "finite": bool(torch.isfinite(output).all().item()),
        },
        remediation=_REMEDIATIONS["numeric"],
    )


def _check_finite_detection() -> CheckResult:
    numpy_mask = np.isfinite(np.array([0.0, np.nan, np.inf]))
    torch_mask = torch.isfinite(torch.tensor([0.0, torch.nan, torch.inf]))
    expected = (True, False, False)
    numpy_detected = tuple(bool(value) for value in numpy_mask) == expected
    torch_detected = tuple(bool(value) for value in torch_mask) == expected
    return _pass_or_fail(
        name="numeric.finite_detection",
        passed=numpy_detected and torch_detected,
        details={
            "numpy_mask": tuple(bool(value) for value in numpy_mask),
            "torch_mask": tuple(bool(value) for value in torch_mask),
            "numpy_detected": numpy_detected,
            "torch_detected": torch_detected,
        },
        remediation=_REMEDIATIONS["numeric"],
    )


def check_numeric() -> tuple[CheckResult, ...]:
    """Exercise required NumPy and CPU PyTorch numeric paths."""
    checks: tuple[tuple[str, Callable[[], CheckResult]], ...] = (
        ("numeric.numpy.float32", lambda: _check_numpy_dtype(np.float32)),
        ("numeric.numpy.float64", lambda: _check_numpy_dtype(np.float64)),
        ("numeric.torch.float32", lambda: _check_torch_dtype(torch.float32)),
        ("numeric.torch.float64", lambda: _check_torch_dtype(torch.float64)),
        ("numeric.cpu_matmul", _check_cpu_matmul),
        ("numeric.finite_detection", _check_finite_detection),
    )
    return tuple(_safe_result(name, _REMEDIATIONS["numeric"], checker) for name, checker in checks)


def _reproducibility_samples(seed: int) -> dict[str, Any]:
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "python_random": tuple(python_rng.random() for _ in range(3)),
        "numpy_random": tuple(float(value) for value in numpy_rng.random(3)),
        "torch_cpu_random": tuple(
            float(value) for value in torch.rand(3, generator=torch_rng, device="cpu")
        ),
    }


def check_reproducibility(seed: int = 1729) -> CheckResult:
    """Confirm local RNG reproducibility without changing global RNG states."""
    first = _reproducibility_samples(seed)
    second = _reproducibility_samples(seed)
    matches = {
        name: first[name] == second[name]
        for name in ("python_random", "numpy_random", "torch_cpu_random")
    }
    all_match = all(matches.values())
    return _pass_or_fail(
        name="numeric.reproducibility",
        passed=all_match,
        details={
            "seed": seed,
            "samples": first,
            "repeat_samples": second,
            "matches": matches,
            "all_match": all_match,
            "scope": "Python, NumPy, and PyTorch CPU only",
        },
        remediation=_REMEDIATIONS["numeric.reproducibility"],
    )


def check_pot() -> CheckResult:
    """Solve and validate a deterministic 3x3 Sinkhorn transport problem."""
    source = np.full(3, 1.0 / 3.0, dtype=np.float64)
    target = np.full(3, 1.0 / 3.0, dtype=np.float64)
    cost = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    plan = np.asarray(ot.sinkhorn(source, target, cost, reg=0.05))
    shape_valid = plan.shape == (3, 3)
    finite = bool(np.isfinite(plan).all())
    nonnegative = bool((plan >= 0.0).all())
    serializable_plan = plan.astype(object)
    serializable_plan[~np.isfinite(plan)] = None
    if shape_valid and finite:
        row_error = float(np.max(np.abs(plan.sum(axis=1) - source)))
        column_error = float(np.max(np.abs(plan.sum(axis=0) - target)))
        max_error = max(row_error, column_error)
    else:
        row_error = None
        column_error = None
        max_error = None
    passed = shape_valid and finite and nonnegative and max_error is not None and max_error <= 1e-5
    return _pass_or_fail(
        name="numeric.pot",
        passed=passed,
        details={
            "shape": plan.shape,
            "transport_plan": serializable_plan.tolist(),
            "finite": finite,
            "nonnegative": nonnegative,
            "row_marginal_error": row_error,
            "column_marginal_error": column_error,
            "max_marginal_error": max_error,
            "tolerance": 1e-5,
        },
        remediation=_REMEDIATIONS["numeric.pot"],
    )


def check_torch_hook() -> CheckResult:
    """Exercise a temporary hook on a tiny CPU Transformer encoder layer."""
    captured: dict[str, Any] = {}
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(1729)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        ).to(device="cpu")
        layer.eval()
        inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0

        def capture_activation(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured["shape"] = tuple(output.shape)
            captured["finite"] = bool(torch.isfinite(output).all().item())
            captured["norm"] = float(output.norm().item())

        handle = layer.register_forward_hook(capture_activation)
        try:
            with torch.inference_mode():
                with_hook = layer(inputs)
        finally:
            handle.remove()
        with torch.inference_mode():
            without_hook = layer(inputs)
        remaining_hooks = len(layer._forward_hooks)

    captured_shape = captured.get("shape", ())
    activation_finite = bool(captured.get("finite", False))
    outputs_match = bool(torch.equal(with_hook, without_hook))
    passed = (
        captured_shape == tuple(inputs.shape)
        and activation_finite
        and outputs_match
        and remaining_hooks == 0
    )
    return _pass_or_fail(
        name="numeric.torch_hook",
        passed=passed,
        details={
            "module_type": type(layer).__name__,
            "device": str(inputs.device),
            "input_shape": tuple(inputs.shape),
            "captured_shape": captured_shape,
            "activation_finite": activation_finite,
            "activation_norm": captured.get("norm"),
            "outputs_match_after_removal": outputs_match,
            "remaining_forward_hooks": remaining_hooks,
        },
        remediation=_REMEDIATIONS["numeric.torch_hook"],
    )


def check_pyvene() -> CheckResult:
    """Import pyvene and execute a model-free tensor intervention smoke."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            import pyvene
    except Exception as exc:
        return CheckResult(
            name="numeric.pyvene",
            status="FAIL",
            details={
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "import_notices": {
                    "stdout": stdout_buffer.getvalue().strip()[:2000],
                    "stderr": stderr_buffer.getvalue().strip()[:2000],
                },
            },
            remediation=_REMEDIATIONS["numeric.pyvene"],
        )

    version = importlib.metadata.version("pyvene")
    import_notices = {
        "stdout": stdout_buffer.getvalue().strip()[:2000],
        "stderr": stderr_buffer.getvalue().strip()[:2000],
    }
    required_api = (
        "VanillaIntervention",
        "IntervenableConfig",
        "IntervenableModel",
    )
    missing_api = tuple(name for name in required_api if not hasattr(pyvene, name))
    if missing_api:
        return CheckResult(
            name="numeric.pyvene",
            status="WARN",
            details={
                "version": version,
                "import_succeeded": True,
                "import_notices": import_notices,
                "missing_api": missing_api,
                "tensor_intervention_exercised": False,
            },
            remediation=(
                "Review the installed pyvene API before Stage 1; no model was downloaded."
            ),
        )

    try:
        intervention = pyvene.VanillaIntervention(embed_dim=3)
        base = torch.tensor([[1.0, 2.0, 3.0]], device="cpu")
        source = torch.tensor([[4.0, 5.0, 6.0]], device="cpu")
        output = intervention(base, source)
        output_matches_source = bool(torch.equal(output, source))
    except Exception as exc:
        return CheckResult(
            name="numeric.pyvene",
            status="WARN",
            details={
                "version": version,
                "import_succeeded": True,
                "import_notices": import_notices,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "tensor_intervention_exercised": False,
            },
            remediation=(
                "Review the model-free VanillaIntervention API before Stage 1; "
                "do not download a model for this Stage 0 check."
            ),
        )

    return _pass_or_fail(
        name="numeric.pyvene",
        passed=output_matches_source,
        details={
            "version": version,
            "import_succeeded": True,
            "import_notices": import_notices,
            "required_api": required_api,
            "tensor_intervention_exercised": True,
            "output_shape": tuple(output.shape),
            "output_matches_source": output_matches_source,
            "model_downloaded": False,
        },
        remediation=_REMEDIATIONS["numeric.pyvene"],
    )


def run_diagnostics(project_root: Path) -> DoctorReport:
    """Run all components while isolating unexpected component exceptions."""
    try:
        system_results = check_system(project_root)
    except Exception as exc:
        system_results = (_exception_result("system", exc, _REMEDIATIONS["system"]),)
    try:
        numeric_results = check_numeric()
    except Exception as exc:
        numeric_results = (_exception_result("numeric", exc, _REMEDIATIONS["numeric"]),)

    single_checks: tuple[tuple[str, str, Callable[[], CheckResult]], ...] = (
        (
            "numeric.reproducibility",
            _REMEDIATIONS["numeric.reproducibility"],
            check_reproducibility,
        ),
        ("numeric.pot", _REMEDIATIONS["numeric.pot"], check_pot),
        ("numeric.torch_hook", _REMEDIATIONS["numeric.torch_hook"], check_torch_hook),
        ("numeric.pyvene", _REMEDIATIONS["numeric.pyvene"], check_pyvene),
    )
    single_results = tuple(
        _safe_result(name, remediation, checker) for name, remediation, checker in single_checks
    )
    return DoctorReport(results=tuple((*system_results, *numeric_results, *single_results)))


def render_markdown(report: DoctorReport) -> str:
    """Render a complete human-readable report with details and remediation."""
    lines = [
        "# TARCA Stage 0 Doctor Report",
        "",
        f"Overall status: **{report.overall_status}**",
        "",
        "| Component | Status | Remediation |",
        "| --- | --- | --- |",
    ]
    for result in report.results:
        remediation = (result.remediation or "None required.").replace("|", "\\|")
        lines.append(f"| `{result.name}` | `{result.status}` | {remediation} |")
    for result in report.results:
        details = json.dumps(
            result.to_dict()["details"],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        lines.extend(
            (
                "",
                f"## `{result.name}`",
                "",
                f"- Status: `{result.status}`",
                f"- Remediation: {result.remediation or 'None required.'}",
                "- Details:",
                "",
                "```json",
                details,
                "```",
            )
        )
    return "\n".join(lines) + "\n"


def report_to_json(report: DoctorReport) -> str:
    """Render the stable machine-readable report schema."""
    return (
        json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
