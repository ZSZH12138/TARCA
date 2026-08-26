from __future__ import annotations

from tarca.stage1b.hardware import estimate_full_run


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
