from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tarca.contracts import ArtifactRef, sha256_file
from tarca.stage2.config import load_stage2_config
from tarca.stage2.recovery_probe import (
    _development_data_ref,
    _probe_complete_checkpoint_worker,
    run_stage2_recovery_probe,
)

ROOT = Path(__file__).resolve().parents[2]


def test_development_data_ref_requires_exactly_one_completed_artifact(
    tmp_path: Path,
) -> None:
    database = tmp_path / "execution.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE job_nodes(task_id TEXT PRIMARY KEY, phase TEXT NOT NULL);
            CREATE TABLE attempts(
                attempt_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                state TEXT NOT NULL,
                artifact_json TEXT
            );
            """
        )

    with pytest.raises(RuntimeError, match="one completed development dataset"):
        _development_data_ref(database)


def test_recovery_probe_worker_rejects_ambiguous_task_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.load_stage2_recovery_spec",
        lambda path: SimpleNamespace(tasks=()),
    )

    with pytest.raises(RuntimeError, match="task identity is ambiguous"):
        _probe_complete_checkpoint_worker(
            str(tmp_path),
            str(ROOT / "configs/stage2/stage2_v1.yaml"),
            str(tmp_path / "runtime"),
            "PATCHTST",
            0,
        )


def test_recovery_probe_worker_loads_one_complete_checkpoint_on_its_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repository"
    runtime = root / "artifacts/stage2/runtime"
    checkpoint = runtime / "checkpoints/complete.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed-complete-checkpoint")
    data_ref = ArtifactRef(
        artifact_id="development-data",
        artifact_type="STAGE2_DEVELOPMENT_DATA",
        content_hash="a" * 64,
        schema_version="1.0.0",
        relative_path="artifacts/stage2/runtime/store/development.bin",
    )
    database = runtime / "execution.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE job_nodes(task_id TEXT PRIMARY KEY, phase TEXT NOT NULL);
            CREATE TABLE attempts(
                attempt_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                state TEXT NOT NULL,
                artifact_json TEXT
            );
            """
        )
        connection.execute("INSERT INTO job_nodes VALUES ('data-task', 'DEV_DATA')")
        connection.execute(
            "INSERT INTO attempts VALUES (?, ?, 1, 'COMPLETED', ?)",
            ("data-task-attempt-1", "data-task", data_ref.model_dump_json()),
        )
    assert _development_data_ref(database) == data_ref

    config = load_stage2_config(ROOT / "configs/stage2/stage2_v1.yaml")
    seed = config.training.initialization_seeds[0]
    task_sha256 = "b" * 64
    recovery_task = SimpleNamespace(
        model_id="PATCHTST",
        seed=seed,
        checkpoint_relative_path=checkpoint.relative_to(root).as_posix(),
        checkpoint_task_sha256=task_sha256,
        checkpoint_sha256=sha256_file(checkpoint),
    )
    observed: dict[str, object] = {}

    class FakeModel:
        def load_state_dict(self, state: object, *, strict: bool) -> None:
            observed["state"] = state
            observed["strict"] = strict

        def to(self, device: torch.device) -> None:
            observed["device"] = str(device)

        def freeze(self) -> None:
            observed["frozen"] = True

    validation_x = torch.ones(3, 64, 2)
    validation_y = torch.ones(3, 24, 2)
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.load_stage2_recovery_spec",
        lambda path: SimpleNamespace(tasks=(recovery_task,)),
    )
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.load_checkpoint",
        lambda path: {
            "status": "COMPLETE",
            "seed": seed,
            "task_sha256": task_sha256,
            "best_state": {"weight": torch.ones(1)},
        },
    )
    monkeypatch.setattr("tarca.stage2.recovery_probe._load_torch", lambda *args: {})
    monkeypatch.setattr("tarca.stage2.recovery_probe._bundle_from_payload", lambda value: value)
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.stack_partition",
        lambda *args: (validation_x, validation_y, ()),
    )
    monkeypatch.setattr("tarca.stage2.recovery_probe._new_neural", lambda *args: FakeModel())
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.forecast_fixed_batch_on_model_device",
        lambda model, inputs: SimpleNamespace(
            mean=torch.ones(2, 24, 2),
            scale=torch.ones(2, 24, 2),
        ),
    )
    monkeypatch.setattr("tarca.stage2.recovery_probe.torch.cuda.set_device", lambda device: None)
    monkeypatch.setattr("tarca.stage2.recovery_probe.torch.cuda.synchronize", lambda device: None)

    result = _probe_complete_checkpoint_worker(
        str(root),
        str(ROOT / "configs/stage2/stage2_v1.yaml"),
        str(runtime),
        "PATCHTST",
        0,
    )

    assert result["checkpoint_hash_unchanged"] is True
    assert result["optimizer_steps"] == 0
    assert result["forecast_finite"] is True
    assert observed["device"] == "cuda:0"
    assert observed["frozen"] is True


def test_recovery_probe_uses_both_gpus_concurrently_and_never_trains(
    tmp_path: Path, monkeypatch
) -> None:
    submitted: list[tuple[str, int]] = []

    class ImmediateFuture:
        def __init__(self, value: dict[str, Any]) -> None:
            self.value = value

        def result(self) -> dict[str, Any]:
            return self.value

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 2
            assert mp_context is marker

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, *args: object) -> ImmediateFuture:
            del function
            model_id = str(args[-2])
            gpu_id = int(args[-1])
            submitted.append((model_id, gpu_id))
            return ImmediateFuture(
                {
                    "model_id": model_id,
                    "gpu_id": gpu_id,
                    "elapsed_seconds": 10.0 + gpu_id,
                    "checkpoint_status": "COMPLETE",
                    "optimizer_steps": 0,
                    "forecast_finite": True,
                    "checkpoint_hash_unchanged": True,
                }
            )

    marker = object()
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.multiprocessing.get_context", lambda _: marker
    )
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.ProcessPoolExecutor", ImmediateExecutor
    )

    result = run_stage2_recovery_probe(
        ROOT,
        ROOT / "configs/stage2/stage2_v1.yaml",
        tmp_path / "artifacts/stage2/runtime",
        remaining_rental_hours=24,
    )

    assert submitted == [("PATCHTST", 0), ("ITRANSFORMER", 1)]
    assert result["zero_optimizer_steps"] is True
    assert result["checkpoint_hashes_unchanged"] is True
    assert result["complete_checkpoint_fast_path_passed"] is True
    assert result["eta_gate_passed"] is True


def test_recovery_probe_rejects_any_invalid_observation(
    tmp_path: Path, monkeypatch
) -> None:
    class ImmediateFuture:
        def result(self) -> dict[str, Any]:
            return {
                "elapsed_seconds": 1.0,
                "checkpoint_status": "COMPLETE",
                "optimizer_steps": 0,
                "forecast_finite": False,
                "checkpoint_hash_unchanged": True,
            }

    class InvalidObservationExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 2

        def __enter__(self) -> InvalidObservationExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, *args: object) -> ImmediateFuture:
            return ImmediateFuture()

    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.multiprocessing.get_context", lambda _: object()
    )
    monkeypatch.setattr(
        "tarca.stage2.recovery_probe.ProcessPoolExecutor",
        InvalidObservationExecutor,
    )

    with pytest.raises(RuntimeError, match="checkpoint probe failed"):
        run_stage2_recovery_probe(
            ROOT,
            ROOT / "configs/stage2/stage2_v1.yaml",
            tmp_path / "artifacts/stage2/runtime",
            remaining_rental_hours=24,
        )
