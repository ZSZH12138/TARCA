from __future__ import annotations

import pytest

from tarca.e01.resources import E01ServerInventory, ServerAdmissionError
from tarca.e01.v2_resources import (
    E01V2ProbeObservation,
    choose_v2_capacity_plan,
    initial_v2_probe_candidates,
    v2_server_admission_check,
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


def test_v2_probe_grid_uses_all_safe_cpu_lanes_and_bounded_batches() -> None:
    candidates = initial_v2_probe_candidates(_inventory())

    assert len(candidates) == 9
    assert {item.cpu_analysis_workers for item in candidates} == {8, 10, 12}
    assert {item.gpu_batch_size for item in candidates} == {2048, 4096, 8192}
    assert all(item.gpu_worker_count == 1 and item.service_cores == 2 for item in candidates)


def test_v2_capacity_selects_fastest_safe_measurement() -> None:
    observations = (
        E01V2ProbeObservation(8, 2048, 100.0, 24.0, 60.0, 2.0),
        E01V2ProbeObservation(10, 4096, 180.0, 44.0, 80.0, 8.0),
        E01V2ProbeObservation(12, 8192, 250.0, 80.0, 95.0, 18.0),
        E01V2ProbeObservation(12, 8192, 300.0, 96.0, 99.0, 22.0, oom=True),
    )

    plan = choose_v2_capacity_plan(_inventory(), observations)

    assert plan.cpu_analysis_workers == 12
    assert plan.gpu_batch_size == 8192
    assert plan.gpu_worker_count == 1
    assert plan.service_cores == 2


def test_v2_capacity_rejects_swap_oom_or_memory_pressure() -> None:
    observations = (
        E01V2ProbeObservation(12, 8192, 300.0, 96.0, 99.0, 22.0, oom=True),
        E01V2ProbeObservation(10, 4096, 200.0, 100.0, 80.0, 8.0, swap_used_gib=1.0),
    )

    with pytest.raises(ServerAdmissionError, match="no safe E01-v2 capacity"):
        choose_v2_capacity_plan(_inventory(), observations)


def test_v2_admission_enforces_five_minute_probe_and_reset_margin() -> None:
    v2_server_admission_check(
        _inventory(),
        estimated_runtime_hours=0.5,
        remaining_rental_hours=4.0,
        minimum_storage_gib=200.0,
        reset_margin_hours=1.0,
        probe_elapsed_seconds=299.0,
    )
    with pytest.raises(ServerAdmissionError, match="five-minute"):
        v2_server_admission_check(
            _inventory(),
            estimated_runtime_hours=0.5,
            remaining_rental_hours=4.0,
            minimum_storage_gib=200.0,
            reset_margin_hours=1.0,
            probe_elapsed_seconds=300.01,
        )
