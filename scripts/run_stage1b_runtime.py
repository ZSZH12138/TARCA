from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.contracts import canonical_json_bytes, canonical_json_hash  # noqa: E402
from tarca.execution.resources import (  # noqa: E402
    PrecisionProbeResult,
    select_precision,
)
from tarca.monitoring.repository import MonitoringRepository  # noqa: E402
from tarca.stage1b.compiler import repository_v2_inputs  # noqa: E402
from tarca.stage1b.runner import (  # noqa: E402
    run_hardware_probe,
    run_scheduled_qualification,
)
from tarca.stage1b.server_environment import (  # noqa: E402
    ServerEnvironmentExpectation,
    validate_server_environment,
)
from tarca.stage1b.sources import SubprocessGitRunner, materialize_source  # noqa: E402


@dataclass(frozen=True, slots=True)
class RuntimeArguments:
    command: str
    artifact_root: Path
    empty_ok: bool = False
    authorize_over_24_hours: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate the TARCA Stage1B v2 server runtime.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/stage1b",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Run bounded server checks and probes.")
    preflight.add_argument("--authorize-over-24-hours", action="store_true")
    subparsers.add_parser("launch", help="Start a new qualification-only task graph.")
    subparsers.add_parser("resume", help="Resume the existing qualification-only task graph.")
    status = subparsers.add_parser("status", help="Print a read-only runtime snapshot.")
    status.add_argument("--empty-ok", action="store_true")
    return parser


def _arguments(namespace: argparse.Namespace) -> RuntimeArguments:
    return RuntimeArguments(
        command=str(namespace.command),
        artifact_root=namespace.artifact_root.resolve(),
        empty_ok=bool(getattr(namespace, "empty_ok", False)),
        authorize_over_24_hours=bool(
            getattr(namespace, "authorize_over_24_hours", False)
        ),
    )


def _atomic_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return canonical_json_hash(value)


def _expectation() -> ServerEnvironmentExpectation:
    return ServerEnvironmentExpectation(
        python_minor=(3, 10),
        torch_version="2.2.2",
        cuda_version="12.1",
        gpu_count=2,
        gpu_name_substring="RTX 4090",
        minimum_vram_bytes=24 * 1024**3,
        minimum_cpu_count=28,
        minimum_ram_bytes=224 * 1024**3,
    )


def _precision_probes() -> tuple[PrecisionProbeResult, PrecisionProbeResult]:
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(104729)
    left = torch.randn((1024, 1024), generator=generator)
    right = torch.randn((1024, 1024), generator=generator)

    def measure(amp: bool) -> tuple[float, torch.Tensor]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = torch.empty(0)
        for _ in range(4):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = left.to(device) @ right.to(device)
        torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-9)
        return 4 * left.shape[0] / elapsed, output.detach().cpu().to(torch.float32)

    fp32_rate, fp32_output = measure(False)
    amp_rate, amp_output = measure(True)
    error = float(torch.max(torch.abs(fp32_output - amp_output)))
    finite = bool(torch.isfinite(amp_output).all())
    return (
        PrecisionProbeResult("FP32", fp32_rate, 0.0, bool(torch.isfinite(fp32_output).all())),
        PrecisionProbeResult("AMP_FP16", amp_rate, error, finite),
    )


def _materialize_sources() -> tuple[dict[str, Any], ...]:
    inputs = repository_v2_inputs(REPOSITORY_ROOT)
    cache = Path(
        os.environ.get(
            "TARCA_STAGE1B_SOURCE_CACHE_ROOT",
            str(REPOSITORY_ROOT / "third_party/stage1b"),
        )
    ).resolve()
    runner = SubprocessGitRunner.discover()
    return tuple(
        {
            "source_id": receipt.source_id,
            "commit": receipt.commit,
            "tree_sha256": receipt.tree_sha256,
            "asset_sha256": [list(item) for item in receipt.asset_sha256],
        }
        for source in inputs.world_suite.sources
        for receipt in (materialize_source(source, cache, runner),)
    )


def run_preflight(arguments: RuntimeArguments) -> dict[str, Any]:
    runtime_root = arguments.artifact_root / "runtime"
    environment = validate_server_environment(_expectation())
    sources = _materialize_sources()
    hardware = run_hardware_probe(
        REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml",
        REPOSITORY_ROOT / "configs/stage1b/qualification_v2.yaml",
        runtime_root,
        authorized_over_24_hours=arguments.authorize_over_24_hours,
    )
    fp32, amp = _precision_probes()
    precision = select_precision(fp32, amp, maximum_allowed_error=0.1)
    environment_hash = _atomic_json(
        runtime_root / "environment_receipt_v2.json",
        asdict(environment),
    )
    precision_hash = _atomic_json(
        runtime_root / "precision_receipt_v2.json",
        asdict(precision),
    )
    source_hash = _atomic_json(
        runtime_root / "official_sources_receipt_v2.json",
        {"sources": sources},
    )
    return {
        "status": "PASS",
        "environment_receipt_sha256": environment_hash,
        "precision_receipt_sha256": precision_hash,
        "official_sources_receipt_sha256": source_hash,
        "hardware": hardware["decision"],
    }


def _required_receipts(artifact_root: Path) -> None:
    runtime = artifact_root / "runtime"
    required = (
        runtime / "environment_receipt_v2.json",
        runtime / "precision_receipt_v2.json",
        runtime / "official_sources_receipt_v2.json",
        runtime / "hardware_probe_v2.json",
    )
    missing = tuple(path.name for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"passing preflight receipts are missing: {', '.join(missing)}")
    hardware = json.loads((runtime / "hardware_probe_v2.json").read_text(encoding="utf-8"))
    if hardware.get("decision", {}).get("feasible") is not True:
        raise RuntimeError("hardware probe receipt is not passing")


def _apply_precision_policy(artifact_root: Path) -> None:
    path = artifact_root / "runtime/precision_receipt_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if selected not in {"FP32", "AMP_FP16"}:
        raise RuntimeError("precision receipt contains no approved policy")
    os.environ["TARCA_STAGE1B_PRECISION"] = str(selected)


def launch_runtime(arguments: RuntimeArguments) -> dict[str, Any]:
    database = arguments.artifact_root / "runtime/execution.sqlite3"
    if database.exists():
        raise RuntimeError("execution database already exists; use resume")
    _required_receipts(arguments.artifact_root)
    _apply_precision_policy(arguments.artifact_root)
    return run_scheduled_qualification(REPOSITORY_ROOT, arguments.artifact_root)


def resume_runtime(arguments: RuntimeArguments) -> dict[str, Any]:
    database = arguments.artifact_root / "runtime/execution.sqlite3"
    if not database.is_file():
        raise RuntimeError("execution database is required before resume")
    _required_receipts(arguments.artifact_root)
    _apply_precision_policy(arguments.artifact_root)
    return run_scheduled_qualification(REPOSITORY_ROOT, arguments.artifact_root)


def emit_safe_status(arguments: RuntimeArguments) -> dict[str, Any]:
    database = arguments.artifact_root / "runtime/execution.sqlite3"
    if not database.is_file():
        if arguments.empty_ok:
            return {"status": "EMPTY"}
        raise RuntimeError("execution database is unavailable")
    return MonitoringRepository(database).snapshot().model_dump(mode="json")


RuntimeHandler = Callable[[RuntimeArguments], dict[str, Any]]
_HANDLERS: MappingProxyType[str, RuntimeHandler] = MappingProxyType(
    {
        "preflight": run_preflight,
        "launch": launch_runtime,
        "resume": resume_runtime,
        "status": emit_safe_status,
    }
)


def dispatch_runtime_command(arguments: RuntimeArguments) -> dict[str, Any]:
    try:
        handler = _HANDLERS[arguments.command]
    except KeyError as error:
        raise ValueError("runtime command is not allowlisted") from error
    return handler(arguments)


def main() -> int:
    arguments = _arguments(_parser().parse_args())
    try:
        result = dispatch_runtime_command(arguments)
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
