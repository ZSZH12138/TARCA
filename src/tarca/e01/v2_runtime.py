from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tarca.e01.resources import E01ServerInventory
from tarca.e01.v2_carry_forward import verify_e01_b_carry_forward
from tarca.e01.v2_config import E01V2Config, load_e01_v2_config
from tarca.e01.v2_resources import (
    E01V2ProbeObservation,
    choose_v2_capacity_plan,
    initial_v2_probe_candidates,
    v2_server_admission_check,
)
from tarca.e01.v2_tasks import compile_e01_v2_graph

E01_V2_FORMAL_RUN_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_E01_V2_FORMAL_RUN"


class E01V2RuntimeAuthorizationError(RuntimeError):
    """Raised when E01-v2 runtime identity or authorization is incomplete."""


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("receipt_sha256", None)
    return result


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload(value)
    return {**payload, "receipt_sha256": canonical_json_hash(payload)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".e01-v2-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_receipt(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise E01V2RuntimeAuthorizationError(f"{label} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E01V2RuntimeAuthorizationError(f"{label} is invalid")
    if value.get("receipt_sha256") != canonical_json_hash(_payload(value)):
        raise E01V2RuntimeAuthorizationError(f"{label} hash verification failed")
    return value


def _graph_summary(config: E01V2Config) -> dict[str, Any]:
    graph = compile_e01_v2_graph(config)
    return {
        "graph_id": graph.graph_id,
        "scientific_config_sha256": graph.scientific_config_hash,
        "total_tasks": len(graph.nodes),
        "gpu_generation_tasks": sum(node.phase == "E01_A_V2_GPU_GENERATE" for node in graph.nodes),
        "cpu_analysis_tasks": sum(node.phase == "E01_A_V2_CPU_ANALYZE" for node in graph.nodes),
        "aggregation_tasks": sum(node.phase == "E01_V2_AGGREGATE" for node in graph.nodes),
    }


def prepare_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    config_file = config_path.resolve()
    config = load_e01_v2_config(config_file)
    carry_forward = verify_e01_b_carry_forward(root, config)
    prepared = _seal(
        {
            "schema_version": "tarca-e01-v2-prepared-receipt",
            "status": "PREPARED",
            "config_file_sha256": sha256_file(config_file),
            "runtime_profile_sha256": config.runtime_hash(),
            "graph": _graph_summary(config),
            "e01_b_carry_forward": carry_forward.model_dump(mode="json"),
            "formal_tasks_executed": 0,
        }
    )
    path = artifact_root.resolve() / "runtime/prepared_receipt_v2.json"
    if path.is_file():
        existing = _read_receipt(path, "prepared receipt")
        if existing != prepared:
            raise E01V2RuntimeAuthorizationError(
                "prepared receipt already exists with identity drift"
            )
        return existing
    _atomic_json(path, prepared)
    return prepared


def _verify_prepared(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
) -> tuple[E01V2Config, dict[str, Any]]:
    config = load_e01_v2_config(config_path.resolve())
    receipt = _read_receipt(
        artifact_root.resolve() / "runtime/prepared_receipt_v2.json",
        "prepared receipt",
    )
    carry_forward = verify_e01_b_carry_forward(repository_root.resolve(), config)
    if receipt.get("config_file_sha256") != sha256_file(config_path.resolve()):
        raise E01V2RuntimeAuthorizationError("prepared receipt config hash drifted")
    if receipt.get("graph") != _graph_summary(config):
        raise E01V2RuntimeAuthorizationError("prepared receipt graph identity drifted")
    if receipt.get("e01_b_carry_forward") != carry_forward.model_dump(mode="json"):
        raise E01V2RuntimeAuthorizationError("prepared receipt E01-B evidence drifted")
    return config, receipt


def dry_run_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    config, prepared = _verify_prepared(repository_root, config_path, artifact_root)
    profile = config.runtime_profile
    declared = E01ServerInventory(
        physical_cpu_cores=profile.expected_physical_cpu_cores,
        logical_cpu_count=profile.expected_physical_cpu_cores,
        available_ram_gib=float(profile.expected_ram_gib),
        gpu_names=(profile.expected_gpu_name_substring,),
        gpu_vram_gib=(float(profile.expected_gpu_vram_gib),),
        free_storage_gib=float(profile.minimum_free_storage_gib + 1),
    )
    return {
        "status": "DRY_RUN_OK",
        "prepared_receipt_sha256": prepared["receipt_sha256"],
        "graph": prepared["graph"],
        "initial_capacity_candidates": len(initial_v2_probe_candidates(declared)),
        "formal_tasks_executed": 0,
        "scientific_results_visible": False,
    }


def _validate_runtime_identity(identity: Mapping[str, object]) -> None:
    expected = {"python": "3.10", "torch": "2.2.2", "cuda": "12.1", "cudnn_major": 8}
    if dict(identity) != expected:
        raise E01V2RuntimeAuthorizationError("server runtime identity does not match E01-v2")


def preflight_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    inventory: E01ServerInventory,
    observations: tuple[E01V2ProbeObservation, ...],
    estimated_runtime_hours: float,
    remaining_rental_hours: float,
    probe_elapsed_seconds: float,
    runtime_identity: Mapping[str, object],
) -> dict[str, Any]:
    config, prepared = _verify_prepared(repository_root, config_path, artifact_root)
    _validate_runtime_identity(runtime_identity)
    plan = choose_v2_capacity_plan(inventory, observations)
    profile = config.runtime_profile
    v2_server_admission_check(
        inventory,
        estimated_runtime_hours=estimated_runtime_hours,
        remaining_rental_hours=remaining_rental_hours,
        minimum_storage_gib=float(profile.minimum_free_storage_gib),
        reset_margin_hours=float(profile.reset_margin_hours),
        probe_elapsed_seconds=probe_elapsed_seconds,
    )
    receipt = _seal(
        {
            "schema_version": "tarca-e01-v2-preflight-receipt",
            "status": "PREFLIGHT_PASS",
            "prepared_receipt_sha256": prepared["receipt_sha256"],
            "graph_id": prepared["graph"]["graph_id"],
            "e01_b_carry_forward_receipt_sha256": prepared["e01_b_carry_forward"]["receipt_sha256"],
            "runtime_identity": dict(runtime_identity),
            "inventory": asdict(inventory),
            "capacity_observations": tuple(asdict(item) for item in observations),
            "capacity_plan": asdict(plan),
            "estimated_runtime_hours": estimated_runtime_hours,
            "remaining_rental_hours": remaining_rental_hours,
            "probe_elapsed_seconds": probe_elapsed_seconds,
            "formal_tasks_executed": 0,
            "scientific_results_visible": False,
        }
    )
    _atomic_json(artifact_root.resolve() / "runtime/preflight_receipt_v2.json", receipt)
    return receipt


def automatic_preflight_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    remaining_rental_hours: float,
) -> dict[str, Any]:
    config, _ = _verify_prepared(repository_root, config_path, artifact_root)
    from tarca.e01.v2_probe import run_bounded_e01_v2_probe

    inventory, observations, estimated, elapsed = run_bounded_e01_v2_probe(
        artifact_root.resolve(), config
    )
    return preflight_e01_v2(
        repository_root,
        config_path,
        artifact_root,
        inventory=inventory,
        observations=observations,
        estimated_runtime_hours=estimated,
        remaining_rental_hours=remaining_rental_hours,
        probe_elapsed_seconds=elapsed,
        runtime_identity=current_e01_v2_runtime_identity(),
    )


def _verified_launch_receipts(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
) -> tuple[E01V2Config, dict[str, Any], dict[str, Any]]:
    config, prepared = _verify_prepared(repository_root, config_path, artifact_root)
    preflight = _read_receipt(
        artifact_root.resolve() / "runtime/preflight_receipt_v2.json",
        "preflight receipt",
    )
    if preflight.get("prepared_receipt_sha256") != prepared.get("receipt_sha256"):
        raise E01V2RuntimeAuthorizationError("preflight does not bind prepared receipt")
    if preflight.get("status") != "PREFLIGHT_PASS":
        raise E01V2RuntimeAuthorizationError("preflight receipt did not pass")
    return config, prepared, preflight


def _authorize(acknowledgement: str) -> None:
    if acknowledgement != E01_V2_FORMAL_RUN_ACKNOWLEDGEMENT:
        raise E01V2RuntimeAuthorizationError("exact E01-v2 formal-run acknowledgement required")


def launch_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    _authorize(acknowledgement)
    database = artifact_root.resolve() / "runtime/execution.sqlite3"
    if database.exists():
        raise E01V2RuntimeAuthorizationError("execution database exists; use resume")
    config, prepared, preflight = _verified_launch_receipts(
        repository_root, config_path, artifact_root
    )
    from tarca.e01.v2_runner import run_formal_e01_v2

    return run_formal_e01_v2(
        repository_root.resolve(),
        config_path.resolve(),
        artifact_root.resolve(),
        config,
        prepared,
        preflight,
        resume=False,
    )


def resume_e01_v2(
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    *,
    acknowledgement: str,
) -> dict[str, Any]:
    _authorize(acknowledgement)
    database = artifact_root.resolve() / "runtime/execution.sqlite3"
    if not database.is_file():
        raise E01V2RuntimeAuthorizationError("execution database is required before resume")
    config, prepared, preflight = _verified_launch_receipts(
        repository_root, config_path, artifact_root
    )
    from tarca.e01.v2_runner import run_formal_e01_v2

    return run_formal_e01_v2(
        repository_root.resolve(),
        config_path.resolve(),
        artifact_root.resolve(),
        config,
        prepared,
        preflight,
        resume=True,
    )


def status_e01_v2(artifact_root: Path, *, empty_ok: bool = False) -> dict[str, Any]:
    database = artifact_root.resolve() / "runtime/execution.sqlite3"
    if not database.is_file():
        if empty_ok:
            return {"status": "NOT_STARTED", "scientific_results_visible": False}
        raise E01V2RuntimeAuthorizationError("execution database does not exist")
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT state, COUNT(*) FROM attempts GROUP BY state ORDER BY state"
        ).fetchall()
        planned = int(connection.execute("SELECT COUNT(*) FROM run_plan_nodes").fetchone()[0])
    counts = {str(state): int(count) for state, count in rows}
    completed = counts.get("COMPLETED", 0)
    terminal = planned == 101 and completed == planned and sum(counts.values()) == planned
    return {
        "status": "COMPLETED" if terminal else "RUNNING_OR_WAITING",
        "attempt_counts": counts,
        "planned_tasks": planned,
        "scientific_results_visible": terminal,
    }


def current_e01_v2_runtime_identity() -> dict[str, object]:
    cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__.split("+")[0],
        "cuda": torch.version.cuda or "NONE",
        "cudnn_major": int(cudnn // 1000) if cudnn is not None else 0,
    }


def dispatch_e01_v2_runtime_command(
    command: str,
    repository_root: Path,
    config_path: Path,
    artifact_root: Path,
    **arguments: Any,
) -> dict[str, Any]:
    commands: dict[str, Callable[..., dict[str, Any]]] = {
        "prepare": prepare_e01_v2,
        "dry-run": dry_run_e01_v2,
        "preflight": automatic_preflight_e01_v2,
        "launch": launch_e01_v2,
        "resume": resume_e01_v2,
    }
    if command == "status":
        return status_e01_v2(artifact_root, empty_ok=bool(arguments.get("empty_ok", False)))
    if command not in commands:
        raise ValueError("E01-v2 runtime command is not allowlisted")
    return commands[command](repository_root, config_path, artifact_root, **arguments)
