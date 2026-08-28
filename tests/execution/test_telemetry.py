from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

import tarca.execution.telemetry as telemetry_module
from tarca.execution.telemetry import (
    GpuSample,
    HostTelemetry,
    PsutilNvmlTelemetryProbe,
    ResourceSample,
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


def _gpu(**changes: Any) -> GpuSample:
    values: dict[str, Any] = {
        "gpu_id": 0,
        "utilization_percent": 50.0,
        "memory_used_bytes": 8 * 1024**3,
        "memory_total_bytes": 24 * 1024**3,
        "power_watts": 250.0,
        "temperature_celsius": 60.0,
        "compute_pids": (123,),
    }
    return GpuSample(**{**values, **changes})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gpu_id": -1}, "GPU ID"),
        ({"utilization_percent": float("nan")}, "utilization"),
        ({"memory_total_bytes": 0}, "memory"),
        ({"memory_used_bytes": -1}, "memory"),
        ({"memory_used_bytes": 25 * 1024**3}, "memory"),
        ({"power_watts": -1.0}, "power"),
        ({"temperature_celsius": float("nan")}, "temperature"),
        ({"compute_pids": (0,)}, "PIDs"),
        ({"compute_pids": (123, 123)}, "unique"),
    ],
)
def test_gpu_sample_rejects_invalid_nvml_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _gpu(**changes)


def _host(**changes: Any) -> HostTelemetry:
    values: dict[str, Any] = {
        "host_cpu_percent": 50.0,
        "effective_busy_cores": 4.0,
        "process_rss_bytes": 1024,
        "process_pss_bytes": 512,
        "process_affinity_cpu_ids": (0, 1),
        "host_memory_used_bytes": 4096,
        "disk_read_bytes_per_second": 1.0,
        "disk_write_bytes_per_second": 2.0,
    }
    return HostTelemetry(**{**values, **changes})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"effective_busy_cores": -1.0}, "rates"),
        ({"host_cpu_percent": 101.0}, "100"),
        ({"process_rss_bytes": -1}, "byte counts"),
        ({"host_memory_used_bytes": -1}, "byte counts"),
        ({"process_pss_bytes": -1}, "PSS"),
        ({"process_affinity_cpu_ids": (-1,)}, "affinity"),
        ({"process_affinity_cpu_ids": (1, 1)}, "unique"),
    ],
)
def test_host_telemetry_rejects_invalid_process_values(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _host(**changes)


def test_resource_sample_policy_and_pid_boundaries() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        ResourceSample(
            sampled_at_utc=datetime(2026, 8, 26),
            **asdict(_host()),
            gpu_samples=(),
        )
    valid_host = _host()
    with pytest.raises(ValueError, match="GPU IDs"):
        ResourceSample(
            sampled_at_utc=datetime(2026, 8, 26, tzinfo=UTC),
            **asdict(valid_host),
            gpu_samples=(_gpu(), _gpu(compute_pids=(456,))),
        )
    with pytest.raises(ValueError, match="process ID"):
        collect_resource_sample(0, _Probe())
    with pytest.raises(ValueError, match="sample interval"):
        TelemetryPolicy(sample_interval_seconds=7.0, persistence_interval_seconds=6.0)
    with pytest.raises(ValueError, match="between five and ten"):
        TelemetryPolicy(sample_interval_seconds=1.0, persistence_interval_seconds=11.0)
    with pytest.raises(ValueError, match="overhead limits"):
        TelemetryPolicy(monitor_maximum_busy_cores=0.0)
    assert (
        monitor_overhead_alerts(
            collect_resource_sample(4242, _Probe()),
            TelemetryPolicy(monitor_maximum_busy_cores=20.0, monitor_maximum_rss_bytes=4 * 1024**3),
        )
        == ()
    )


def test_production_probe_gracefully_runs_without_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module.psutil, "disk_io_counters", lambda: None)

    def unavailable(_name: str) -> Any:
        raise ImportError("NVML is unavailable")

    monkeypatch.setattr(telemetry_module.importlib, "import_module", unavailable)
    probe = PsutilNvmlTelemetryProbe()
    assert probe.gpu_samples() == ()
    probe.close()


class _FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self) -> None:
        self.shutdown = False

    def nvmlInit(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, gpu_id: int) -> int:
        return gpu_id

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> Any:
        return SimpleNamespace(gpu=70 + handle)

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> Any:
        return SimpleNamespace(used=(8 + handle) * 1024**3, total=24 * 1024**3)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: int) -> tuple[Any, ...]:
        if handle == 1:
            raise RuntimeError("query not supported")
        return (SimpleNamespace(pid=321), SimpleNamespace(pid=321))

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return 300_000 + handle

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        assert sensor == self.NVML_TEMPERATURE_GPU
        return 65 + handle

    def nvmlShutdown(self) -> None:
        self.shutdown = True


def test_production_probe_collects_nvml_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeNvml()
    monkeypatch.setattr(
        telemetry_module.psutil,
        "disk_io_counters",
        lambda: SimpleNamespace(read_bytes=100, write_bytes=200),
    )
    monkeypatch.setattr(telemetry_module.importlib, "import_module", lambda _name: fake)
    probe = PsutilNvmlTelemetryProbe()

    samples = probe.gpu_samples()

    assert len(samples) == 2
    assert samples[0].compute_pids == (321,)
    assert samples[1].compute_pids == ()
    probe.close()
    assert fake.shutdown is True
    assert probe.gpu_samples() == ()


class _FakeProcess:
    def __init__(self, *, child: bool = False, affinity_error: bool = False) -> None:
        self.child = child
        self.affinity_error = affinity_error
        self.cpu_seconds = 1.0

    def children(self, recursive: bool) -> tuple[_FakeProcess, ...]:
        assert recursive is True
        return (_FakeProcess(child=True),)

    def cpu_times(self) -> Any:
        if self.child:
            raise psutil.NoSuchProcess(pid=999)
        return SimpleNamespace(user=self.cpu_seconds, system=0.0)

    def memory_info(self) -> Any:
        return SimpleNamespace(rss=2048)

    def memory_full_info(self) -> Any:
        return SimpleNamespace(pss=1024)

    def cpu_affinity(self) -> list[int]:
        if self.affinity_error:
            raise AttributeError("affinity unavailable")
        return [3, 1]


@pytest.mark.parametrize("affinity_error", [False, True])
def test_production_host_snapshot_collects_process_tree_and_disk_rates(
    monkeypatch: pytest.MonkeyPatch, affinity_error: bool
) -> None:
    disk_values = iter(
        (
            SimpleNamespace(read_bytes=100, write_bytes=200),
            SimpleNamespace(read_bytes=160, write_bytes=290),
            SimpleNamespace(read_bytes=220, write_bytes=380),
        )
    )
    monotonic_values = iter((10.0, 12.0, 14.0))
    monkeypatch.setattr(telemetry_module.psutil, "disk_io_counters", lambda: next(disk_values))
    monkeypatch.setattr(telemetry_module.importlib, "import_module", lambda _name: None)
    process = _FakeProcess(affinity_error=affinity_error)
    monkeypatch.setattr(telemetry_module.psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(telemetry_module.psutil, "cpu_count", lambda logical: 8)
    monkeypatch.setattr(telemetry_module.psutil, "cpu_percent", lambda interval: 25.0)
    monkeypatch.setattr(
        telemetry_module.psutil, "virtual_memory", lambda: SimpleNamespace(used=8192)
    )
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: next(monotonic_values))
    probe = PsutilNvmlTelemetryProbe()

    initial = probe.host_snapshot(123)
    process.cpu_seconds = 6.0
    sample = probe.host_snapshot(123)

    assert initial.effective_busy_cores == 0.0
    assert sample.effective_busy_cores == 2.5
    assert sample.process_rss_bytes == 2048
    assert sample.process_pss_bytes == 1024
    assert sample.process_affinity_cpu_ids == ((tuple(range(8))) if affinity_error else (1, 3))
    assert sample.disk_read_bytes_per_second == 30.0
    assert sample.disk_write_bytes_per_second == 45.0


def test_production_monitor_snapshot_excludes_worker_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(
        telemetry_module.psutil,
        "disk_io_counters",
        lambda: SimpleNamespace(read_bytes=0, write_bytes=0),
    )
    monkeypatch.setattr(telemetry_module.importlib, "import_module", lambda _name: None)
    monkeypatch.setattr(telemetry_module.psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(telemetry_module.psutil, "cpu_percent", lambda interval: 0.0)
    monkeypatch.setattr(
        telemetry_module.psutil, "virtual_memory", lambda: SimpleNamespace(used=8192)
    )
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: 10.0)
    probe = PsutilNvmlTelemetryProbe()

    snapshot = probe.monitor_snapshot(123)

    assert snapshot.process_rss_bytes == 2048
