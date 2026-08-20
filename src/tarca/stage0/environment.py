from __future__ import annotations

import contextlib
import ctypes
import importlib
import importlib.metadata
import io
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from tarca.contracts import (
    AcceleratorCapabilities,
    DoctorCheckResults,
    DoctorReport,
    DoctorResources,
    DoctorVersions,
    EnvironmentPlatform,
    EnvironmentProfile,
    EnvironmentResources,
    PythonEnvironment,
)


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _total_memory_bytes() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
            return None
        return int(status.total_physical)
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * page_count


def _accelerator_capabilities() -> AcceleratorCapabilities:
    import torch

    for dtype in (torch.float32, torch.float64):
        value = torch.ones(2, dtype=dtype) + torch.ones(2, dtype=dtype)
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite dtype capability probe: {dtype}")
    mps_backend = getattr(torch.backends, "mps", None)
    return AcceleratorCapabilities(
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
        mps_available=bool(mps_backend is not None and mps_backend.is_available()),
        tested_torch_dtypes=("float32", "float64"),
    )


def capture_environment_profile(workspace: Path) -> EnvironmentProfile:
    disk = shutil.disk_usage(workspace.resolve())
    return EnvironmentProfile(
        schema_version="1.0.0",
        profile_id="stage0-default-local",
        profile_role="DEFAULT_EXECUTION_PROFILE",
        execution_backend_replaceable=True,
        compute_boundary_fixed=False,
        platform=EnvironmentPlatform(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
        ),
        python=PythonEnvironment(
            version=platform.python_version(),
            implementation=platform.python_implementation(),
            executable=str(Path(sys.executable).resolve()),
        ),
        resources=EnvironmentResources(
            logical_cpu_count=int(os.cpu_count() or 0),
            memory_total_bytes=_total_memory_bytes(),
            disk_total_bytes=int(disk.total),
            disk_free_bytes=int(disk.free),
        ),
        accelerators=_accelerator_capabilities(),
    )


def _doctor_checks(values: dict[str, str]) -> DoctorCheckResults:
    return DoctorCheckResults.model_validate(values)


def _doctor_versions(values: dict[str, str]) -> DoctorVersions:
    return DoctorVersions.model_validate(values)


def _doctor_resources(profile: EnvironmentProfile) -> DoctorResources:
    return DoctorResources(
        **profile.resources.model_dump(),
        python_version=profile.python.version,
        python_executable=profile.python.executable,
    )


def run_doctor(workspace: Path) -> DoctorReport:
    checks: dict[str, str] = {}
    versions: dict[str, str] = {}
    profile: EnvironmentProfile | None = None
    cuda_available = False
    cuda_device_count = 0
    try:
        profile = capture_environment_profile(workspace)
        resources = profile.resources
        if resources.disk_free_bytes <= 0:
            raise RuntimeError("workspace disk has no reported free space")
        checks["workspace_disk"] = "PASS"

        if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
            raise RuntimeError("Python 3.11 or 3.12 is required")
        checks["python_version"] = "PASS"

        import numpy as np
        import torch

        versions["numpy"] = np.__version__
        versions["torch"] = torch.__version__
        torch.manual_seed(1729)
        first = torch.randn(4, dtype=torch.float64)
        torch.manual_seed(1729)
        second = torch.randn(4, dtype=torch.float64)
        if not torch.equal(first, second) or not torch.isfinite(first).all():
            raise RuntimeError("deterministic finite torch smoke failed")
        checks["torch_basic"] = "PASS"

        layer = torch.nn.Linear(2, 1)
        captured: list[torch.Tensor] = []
        hook = layer.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
        try:
            output = layer(torch.ones((1, 2), dtype=torch.float32))
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != output.shape:
            raise RuntimeError("torch hook smoke failed")
        checks["torch_hook"] = "PASS"

        ot = importlib.import_module("ot")
        versions["pot"] = _version("POT")
        a = np.array([0.5, 0.5], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)
        cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        coupling = ot.sinkhorn(a, b, cost, reg=0.1)
        if not np.isfinite(coupling).all() or not np.isclose(coupling.sum(), 1.0):
            raise RuntimeError("POT Sinkhorn smoke failed")
        checks["pot_sinkhorn"] = "PASS"

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module("pyvene")
        versions["pyvene"] = _version("pyvene")
        checks["pyvene_import"] = "PASS"

        with tempfile.NamedTemporaryFile(dir=workspace, prefix=".tarca-write-", delete=True) as tmp:
            tmp.write(b"tarca")
            tmp.flush()
        checks["workspace_write"] = "PASS"
        cuda_available = profile.accelerators.cuda_available
        cuda_device_count = profile.accelerators.cuda_device_count
    except Exception as exc:  # pragma: no cover - exact dependency failures vary by host
        return DoctorReport(
            status="FAIL",
            gpu_required=False,
            checks=_doctor_checks(checks),
            versions=_doctor_versions(versions),
            resources=_doctor_resources(profile) if profile is not None else None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return DoctorReport(
        status="PASS",
        gpu_required=False,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        default_profile_id=profile.profile_id,
        execution_backend_replaceable=profile.execution_backend_replaceable,
        tested_torch_dtypes=profile.accelerators.tested_torch_dtypes,
        checks=_doctor_checks(checks),
        versions=_doctor_versions(versions),
        resources=_doctor_resources(profile),
    )
