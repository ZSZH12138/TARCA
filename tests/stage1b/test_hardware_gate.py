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
