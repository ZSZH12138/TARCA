from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tarca.e01 import probe, v2_probe
from tarca.e01.resources import E01ServerInventory
from tarca.e01.v2_config import load_e01_v2_config
from tarca.e01.v2_resources import (
    E01V2CapacityCandidate,
    E01V2CapacityPlan,
    E01V2ProbeObservation,
)


def _inventory() -> E01ServerInventory:
    return E01ServerInventory(
        physical_cpu_cores=14,
        logical_cpu_count=14,
        available_ram_gib=112.0,
        gpu_names=("NVIDIA GeForce RTX 4090",),
        gpu_vram_gib=(24.0,),
        free_storage_gib=350.0,
    )


def test_inspect_e01_server_converts_host_inventory_to_gib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    gib = 1024**3
    hardware = SimpleNamespace(
        physical_cpu_count=14,
        logical_cpu_count=28,
        available_memory_bytes=112 * gib,
        gpu_names=("RTX 4090",),
        gpu_vram_bytes=(24 * gib,),
    )
    monkeypatch.setattr(probe, "inventory_hardware", lambda: hardware)
    monkeypatch.setattr(
        probe.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=350 * gib),
    )

    assert probe.inspect_e01_server(tmp_path) == E01ServerInventory(
        physical_cpu_cores=14,
        logical_cpu_count=28,
        available_ram_gib=112.0,
        gpu_names=("RTX 4090",),
        gpu_vram_gib=(24.0,),
        free_storage_gib=350.0,
    )


def test_capacity_observation_runs_one_mocked_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ImmediateFuture:
        def __init__(self, result: tuple[int, int]) -> None:
            self._result = result

        def result(self) -> tuple[int, int]:
            return self._result

    class ImmediatePool:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function, seed: int) -> ImmediateFuture:
            return ImmediateFuture(function(seed))

    candidate = E01V2CapacityCandidate(cpu_analysis_workers=1, gpu_batch_size=2048)
    monkeypatch.setattr(v2_probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(v2_probe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(v2_probe.torch.cuda, "reset_peak_memory_stats", lambda _index: None)
    monkeypatch.setattr(v2_probe.torch.cuda, "max_memory_allocated", lambda _index: 2 * 1024**3)
    monkeypatch.setattr(v2_probe.torch.cuda, "utilization", lambda _index: 75.0, raising=False)
    monkeypatch.setattr(
        v2_probe,
        "initial_v2_probe_candidates",
        lambda _inventory: (candidate,),
    )
    monkeypatch.setattr(v2_probe, "_gpu_generation_probe", lambda batch: batch * 60)
    monkeypatch.setattr(v2_probe.concurrent.futures, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(v2_probe.psutil, "swap_memory", lambda: SimpleNamespace(used=0))
    monkeypatch.setattr(
        v2_probe.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
    )

    observations, elapsed = v2_probe.run_v2_capacity_observations(
        _inventory(), deadline_seconds=10.0
    )

    assert elapsed <= 10.0
    assert len(observations) == 1
    assert observations[0].cpu_analysis_workers == 1
    assert observations[0].gpu_batch_size == 2048
    assert observations[0].gpu_utilization_percent == 75.0
    assert observations[0].peak_gpu_vram_gib == pytest.approx(2.0)
    assert observations[0].oom is False


def test_capacity_probe_rejects_missing_cuda_and_invalid_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v2_probe.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        v2_probe.run_v2_capacity_observations(_inventory())

    monkeypatch.setattr(v2_probe.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="inside"):
        v2_probe.run_v2_capacity_observations(_inventory(), deadline_seconds=0.0)


def test_runtime_estimate_and_bounded_probe_are_composed_from_measured_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = load_e01_v2_config(Path("configs/e01/e01_v2.yaml"))
    observations = (E01V2ProbeObservation(12, 8192, 250.0, 80.0, 95.0, 18.0),)
    capacity = E01V2CapacityPlan(12, 1, 8192, 2, 250.0, 80.0, 18.0)
    expected = v2_probe.estimate_e01_v2_runtime_hours(config, capacity)
    monkeypatch.setattr(v2_probe, "inspect_e01_server", lambda _root: _inventory())
    monkeypatch.setattr(
        v2_probe,
        "run_v2_capacity_observations",
        lambda _inventory: (observations, 40.0),
    )
    monkeypatch.setattr(
        v2_probe,
        "choose_v2_capacity_plan",
        lambda _inventory, _observations: capacity,
    )

    inventory, measured, estimated, elapsed = v2_probe.run_bounded_e01_v2_probe(tmp_path, config)

    assert inventory == _inventory()
    assert measured == observations
    assert estimated == expected
    assert estimated > 0.0
    assert elapsed == 40.0
