from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tarca.execution.telemetry import (
    GpuSample,
    HostTelemetry,
    TelemetryPolicy,
    collect_resource_sample,
    monitor_overhead_alerts,
)


class _Probe:
    def host_snapshot(self, process_id: int) -> HostTelemetry:
        assert process_id == 4242
        return HostTelemetry(
            host_cpu_percent=75.0,
            effective_busy_cores=14.5,
            process_rss_bytes=3 * 1024**3,
            process_pss_bytes=2 * 1024**3,
            process_affinity_cpu_ids=tuple(range(20)),
            host_memory_used_bytes=100 * 1024**3,
            disk_read_bytes_per_second=1000.0,
            disk_write_bytes_per_second=2000.0,
        )

    def gpu_samples(self) -> tuple[GpuSample, ...]:
        return (
            GpuSample(
                gpu_id=0,
                utilization_percent=91.0,
                memory_used_bytes=18 * 1024**3,
                memory_total_bytes=24 * 1024**3,
                power_watts=400.0,
                temperature_celsius=72.0,
                compute_pids=(4242,),
            ),
        )


def test_collect_resource_sample_preserves_actual_process_and_gpu_usage() -> None:
    sampled_at = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    sample = collect_resource_sample(4242, _Probe(), sampled_at_utc=sampled_at)
    assert sample.sampled_at_utc == sampled_at
    assert sample.effective_busy_cores == 14.5
    assert len(sample.process_affinity_cpu_ids) == 20
    assert sample.process_rss_bytes == 3 * 1024**3
    assert sample.gpu_samples[0].memory_used_bytes == 18 * 1024**3
    assert sample.gpu_samples[0].compute_pids == (4242,)


def test_telemetry_rejects_impossible_gpu_values() -> None:
    with pytest.raises(ValueError, match="utilization"):
        GpuSample(
            gpu_id=0,
            utilization_percent=101.0,
            memory_used_bytes=0,
            memory_total_bytes=24 * 1024**3,
            power_watts=0.0,
            temperature_celsius=20.0,
            compute_pids=(),
        )


def test_monitor_overhead_alerts_at_one_core_or_one_gib() -> None:
    sample = collect_resource_sample(4242, _Probe())
    alerts = monitor_overhead_alerts(sample)
    assert set(alerts) == {"MONITOR_CPU_OVERHEAD", "MONITOR_MEMORY_OVERHEAD"}
    policy = TelemetryPolicy()
    assert policy.sample_interval_seconds == 2.0
    assert 5.0 <= policy.persistence_interval_seconds <= 10.0
