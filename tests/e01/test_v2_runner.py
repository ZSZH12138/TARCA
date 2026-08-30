from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.e01 import v2_runner
from tarca.e01.v2_config import load_e01_v2_config
from tarca.e01.v2_tasks import compile_e01_v2_graph
from tarca.execution.resources import ResourceCapacity


class RecordingState:
    def __init__(self, completed: dict[str, object] | None = None) -> None:
        self._completed = completed or {}
        self.enqueued: list[str] = []

    def completed_artifacts(self, _run_id: str):
        return dict(self._completed)

    def enqueue_task(
        self,
        _run_id: str,
        task,
        _executor_key: str,
        *,
        dependency_task_ids: tuple[str, ...],
    ) -> None:
        assert dependency_task_ids == ()
        self.enqueued.append(task.task_id)


def test_runner_builds_relative_paths_and_rejects_bind_mount_escape(tmp_path) -> None:
    root = tmp_path / "repo"
    inside = root / "configs/e01.yaml"
    outside = tmp_path / "outside.yaml"
    inside.parent.mkdir(parents=True)
    inside.touch()
    outside.touch()

    assert v2_runner._relative(root, inside, "config") == "configs/e01.yaml"
    with pytest.raises(ValueError, match="bind mount"):
        v2_runner._relative(root, outside, "config")


def test_runner_converts_frozen_graph_and_enqueues_only_ready_nodes() -> None:
    config = load_e01_v2_config(Path("configs/e01/e01_v2.yaml"))
    graph = compile_e01_v2_graph(config)

    plan = v2_runner._run_plan(graph)
    state = RecordingState()
    enqueued = v2_runner._enqueue_ready(graph, state, "run-test")

    assert len(plan) == len(graph.nodes) == 101
    assert len(enqueued) == 50
    assert enqueued == tuple(state.enqueued)
    assert all(task_id.startswith("e01v2-a-generate-") for task_id in enqueued)
    assert plan[0].identity == graph.nodes[0].task.identity


def test_runner_capacity_uses_preflight_inventory_and_current_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    gib = 1024**3
    preflight = {
        "inventory": {
            "logical_cpu_count": 14,
            "physical_cpu_cores": 14,
            "available_ram_gib": 112.0,
            "gpu_vram_gib": [24.0],
        }
    }
    monkeypatch.setattr(
        v2_runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=350 * gib),
    )

    assert v2_runner._capacity(preflight, tmp_path) == ResourceCapacity(
        logical_cpu_count=14,
        physical_cpu_count=14,
        available_memory_bytes=112 * gib,
        gpu_memory_bytes=(24 * gib,),
        local_storage_available=True,
        local_storage_free_bytes=350 * gib,
    )


def test_runner_reads_final_artifact_and_fails_closed_on_missing_or_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Store:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def load_bytes(self, _reference: object) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    state = RecordingState({"e01v2-aggregate": object()})
    monkeypatch.setattr(v2_runner, "e01_v2_artifact_store", lambda _root: Store({"status": "PASS"}))
    assert v2_runner._read_final(tmp_path, state, "run-test") == {"status": "PASS"}

    with pytest.raises(RuntimeError, match="missing"):
        v2_runner._read_final(tmp_path, RecordingState(), "run-test")

    monkeypatch.setattr(v2_runner, "e01_v2_artifact_store", lambda _root: Store(["invalid"]))
    with pytest.raises(RuntimeError, match="invalid"):
        v2_runner._read_final(tmp_path, state, "run-test")
