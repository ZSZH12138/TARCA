from __future__ import annotations

from types import SimpleNamespace

import pytest

import tarca.stage1b.hardware as hardware
from tarca.stage1b.hardware import estimate_full_run


def test_hardware_inventory_honors_container_cpu_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hardware.os,
        "sched_getaffinity",
        lambda _pid: set(range(28)),
        raising=False,
    )
    monkeypatch.setattr(
        hardware.psutil,
        "cpu_count",
        lambda *, logical: 128 if logical else 64,
    )
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=1024 * 1024**3, available=900 * 1024**3),
    )
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)

    inventory = hardware.inventory_hardware()

    assert inventory.logical_cpu_count == 28
    assert inventory.physical_cpu_count == 28


def test_hardware_gate_blocks_estimate_over_120_hours() -> None:
    decision = estimate_full_run(
        probe_seconds=120.0,
        probe_work_units=1,
        full_work_units=4000,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )

    assert not decision.feasible
    assert decision.estimated_hours > 120.0
    assert "runtime" in decision.failed_checks


def test_hardware_gate_blocks_projected_memory_pressure() -> None:
    decision = estimate_full_run(
        probe_seconds=1.0,
        probe_work_units=10,
        full_work_units=100,
        projected_peak_memory_bytes=15 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )

    assert not decision.feasible
    assert "memory" in decision.failed_checks


def test_hardware_gate_accepts_safe_runtime_and_memory() -> None:
    decision = estimate_full_run(
        probe_seconds=2.0,
        probe_work_units=10,
        full_work_units=1000,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )

    assert decision.feasible
    assert decision.failed_checks == ()


def test_hardware_gate_accounts_for_parallel_slots_fixed_work_and_safety_margin() -> None:
    decision = estimate_full_run(
        probe_seconds=7200.0,
        probe_work_units=1,
        full_work_units=1,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        parallel_work_slots=2,
        fixed_seconds=3600.0,
        safety_factor=1.25,
    )

    assert decision.estimated_hours == 2.5
    assert decision.feasible


@pytest.mark.parametrize(
    ("parallel_work_slots", "fixed_seconds", "safety_factor"),
    ((0, 0.0, 1.0), (1, -1.0, 1.0), (1, 0.0, 0.0)),
)
def test_hardware_gate_rejects_invalid_projection_controls(
    parallel_work_slots: int,
    fixed_seconds: float,
    safety_factor: float,
) -> None:
    with pytest.raises(ValueError, match="projection controls"):
        estimate_full_run(
            probe_seconds=1.0,
            probe_work_units=1,
            full_work_units=1,
            projected_peak_memory_bytes=1,
            available_memory_bytes=2,
            parallel_work_slots=parallel_work_slots,
            fixed_seconds=fixed_seconds,
            safety_factor=safety_factor,
        )


def test_hardware_gate_requires_authorization_between_24_and_120_hours() -> None:
    blocked = estimate_full_run(
        probe_seconds=36.0,
        probe_work_units=1,
        full_work_units=3000,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
    )
    assert blocked.estimated_hours == 30.0
    assert not blocked.feasible
    assert blocked.requires_24_hour_authorization
    authorized = estimate_full_run(
        probe_seconds=36.0,
        probe_work_units=1,
        full_work_units=3000,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        authorized_over_24_hours=True,
    )
    assert authorized.feasible


def test_hardware_gate_remains_infeasible_over_120_hours_even_when_authorized() -> None:
    decision = estimate_full_run(
        probe_seconds=150.0,
        probe_work_units=1,
        full_work_units=3000,
        projected_peak_memory_bytes=4 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        authorized_over_24_hours=True,
    )
    assert not decision.feasible
    assert decision.infeasible_over_120_hours
