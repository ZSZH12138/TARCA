from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import psutil


@dataclass(frozen=True, slots=True)
class GpuSample:
    gpu_id: int
    utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    power_watts: float
    temperature_celsius: float
    compute_pids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.gpu_id < 0:
            raise ValueError("GPU ID must be non-negative")
        if not math.isfinite(self.utilization_percent) or not 0 <= self.utilization_percent <= 100:
            raise ValueError("GPU utilization must be finite and within [0, 100]")
        if (
            self.memory_total_bytes <= 0
            or self.memory_used_bytes < 0
            or self.memory_used_bytes > self.memory_total_bytes
        ):
            raise ValueError("GPU memory values are invalid")
        if not math.isfinite(self.power_watts) or self.power_watts < 0:
            raise ValueError("GPU power must be finite and non-negative")
        if not math.isfinite(self.temperature_celsius):
            raise ValueError("GPU temperature must be finite")
        if any(type(pid) is not int or pid <= 0 for pid in self.compute_pids):
            raise ValueError("GPU compute PIDs must be positive integers")
        if len(self.compute_pids) != len(set(self.compute_pids)):
            raise ValueError("GPU compute PIDs must be unique")


@dataclass(frozen=True, slots=True)
class HostTelemetry:
    host_cpu_percent: float
    effective_busy_cores: float
    process_rss_bytes: int
    process_pss_bytes: int | None
    process_affinity_cpu_ids: tuple[int, ...]
    host_memory_used_bytes: int
    disk_read_bytes_per_second: float
    disk_write_bytes_per_second: float

    def __post_init__(self) -> None:
        finite_values = (
            self.host_cpu_percent,
            self.effective_busy_cores,
            self.disk_read_bytes_per_second,
            self.disk_write_bytes_per_second,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite_values):
            raise ValueError("host telemetry rates must be finite and non-negative")
        if self.host_cpu_percent > 100:
            raise ValueError("host CPU percent must not exceed 100")
        if self.process_rss_bytes < 0 or self.host_memory_used_bytes < 0:
            raise ValueError("host telemetry byte counts must be non-negative")
        if self.process_pss_bytes is not None and self.process_pss_bytes < 0:
            raise ValueError("process PSS must be non-negative")
        if any(type(cpu) is not int or cpu < 0 for cpu in self.process_affinity_cpu_ids):
            raise ValueError("CPU affinity IDs must be non-negative integers")
        if len(self.process_affinity_cpu_ids) != len(set(self.process_affinity_cpu_ids)):
            raise ValueError("CPU affinity IDs must be unique")


@dataclass(frozen=True, slots=True)
class ResourceSample:
    sampled_at_utc: datetime
    host_cpu_percent: float
    effective_busy_cores: float
    process_rss_bytes: int
    process_pss_bytes: int | None
    process_affinity_cpu_ids: tuple[int, ...]
    host_memory_used_bytes: int
    gpu_samples: tuple[GpuSample, ...]
    disk_read_bytes_per_second: float
    disk_write_bytes_per_second: float

    def __post_init__(self) -> None:
        if self.sampled_at_utc.tzinfo is None or self.sampled_at_utc.utcoffset() != UTC.utcoffset(
            self.sampled_at_utc
        ):
            raise ValueError("resource sample timestamp must be timezone-aware UTC")
        HostTelemetry(
            host_cpu_percent=self.host_cpu_percent,
            effective_busy_cores=self.effective_busy_cores,
            process_rss_bytes=self.process_rss_bytes,
            process_pss_bytes=self.process_pss_bytes,
            process_affinity_cpu_ids=self.process_affinity_cpu_ids,
            host_memory_used_bytes=self.host_memory_used_bytes,
            disk_read_bytes_per_second=self.disk_read_bytes_per_second,
            disk_write_bytes_per_second=self.disk_write_bytes_per_second,
        )
        gpu_ids = tuple(sample.gpu_id for sample in self.gpu_samples)
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("resource sample GPU IDs must be unique")


@dataclass(frozen=True, slots=True)
class TelemetryPolicy:
    sample_interval_seconds: float = 2.0
    persistence_interval_seconds: float = 6.0
    monitor_maximum_busy_cores: float = 1.0
    monitor_maximum_rss_bytes: int = 1024**3

    def __post_init__(self) -> None:
        if not 0 < self.sample_interval_seconds <= self.persistence_interval_seconds:
            raise ValueError(
                "telemetry sample interval must be positive and no larger than persistence"
            )
        if not 5.0 <= self.persistence_interval_seconds <= 10.0:
            raise ValueError("telemetry persistence interval must be between five and ten seconds")
        if self.monitor_maximum_busy_cores <= 0 or self.monitor_maximum_rss_bytes <= 0:
            raise ValueError("monitor overhead limits must be positive")


class TelemetryProbe(Protocol):
    def host_snapshot(self, process_id: int) -> HostTelemetry:
        raise NotImplementedError

    def gpu_samples(self) -> tuple[GpuSample, ...]:
        raise NotImplementedError


def collect_resource_sample(
    process_id: int,
    probe: TelemetryProbe,
    *,
    sampled_at_utc: datetime | None = None,
) -> ResourceSample:
    if process_id <= 0:
        raise ValueError("telemetry process ID must be positive")
    host = probe.host_snapshot(process_id)
    return ResourceSample(
        sampled_at_utc=sampled_at_utc or datetime.now(UTC),
        host_cpu_percent=host.host_cpu_percent,
        effective_busy_cores=host.effective_busy_cores,
        process_rss_bytes=host.process_rss_bytes,
        process_pss_bytes=host.process_pss_bytes,
        process_affinity_cpu_ids=host.process_affinity_cpu_ids,
        host_memory_used_bytes=host.host_memory_used_bytes,
        gpu_samples=probe.gpu_samples(),
        disk_read_bytes_per_second=host.disk_read_bytes_per_second,
        disk_write_bytes_per_second=host.disk_write_bytes_per_second,
    )


def monitor_overhead_alerts(
    sample: ResourceSample,
    policy: TelemetryPolicy | None = None,
) -> tuple[str, ...]:
    resolved = policy or TelemetryPolicy()
    alerts: list[str] = []
    if sample.effective_busy_cores > resolved.monitor_maximum_busy_cores:
        alerts.append("MONITOR_CPU_OVERHEAD")
    if sample.process_rss_bytes > resolved.monitor_maximum_rss_bytes:
        alerts.append("MONITOR_MEMORY_OVERHEAD")
    return tuple(alerts)


class PsutilNvmlTelemetryProbe:
    """Stateful production probe for process trees, disk rates, and NVIDIA NVML."""

    def __init__(self) -> None:
        disk = psutil.disk_io_counters()
        self._last_disk_read = int(disk.read_bytes) if disk is not None else 0
        self._last_disk_write = int(disk.write_bytes) if disk is not None else 0
        self._last_disk_time = time.monotonic()
        self._pynvml: Any | None = None
        try:
            module = importlib.import_module("pynvml")
            module.nvmlInit()
            self._pynvml = module
        except Exception:
            self._pynvml = None

    @staticmethod
    def _process_tree(process_id: int) -> tuple[Any, ...]:
        root = psutil.Process(process_id)
        children = tuple(root.children(recursive=True))
        return (root, *children)

    def host_snapshot(self, process_id: int) -> HostTelemetry:
        processes = self._process_tree(process_id)
        cpu_percent = 0.0
        rss = 0
        pss_total = 0
        has_pss = True
        for process in processes:
            try:
                cpu_percent += float(process.cpu_percent(interval=None))
                rss += int(process.memory_info().rss)
                full = process.memory_full_info()
                pss = getattr(full, "pss", None)
                if pss is None:
                    has_pss = False
                else:
                    pss_total += int(pss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        try:
            affinity = tuple(sorted(int(cpu) for cpu in processes[0].cpu_affinity()))
        except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess):
            affinity = tuple(range(psutil.cpu_count(logical=True) or 1))
        disk = psutil.disk_io_counters()
        now = time.monotonic()
        elapsed = max(now - self._last_disk_time, 1e-9)
        current_read = int(disk.read_bytes) if disk is not None else self._last_disk_read
        current_write = int(disk.write_bytes) if disk is not None else self._last_disk_write
        read_rate = max(0.0, (current_read - self._last_disk_read) / elapsed)
        write_rate = max(0.0, (current_write - self._last_disk_write) / elapsed)
        self._last_disk_read = current_read
        self._last_disk_write = current_write
        self._last_disk_time = now
        memory = psutil.virtual_memory()
        return HostTelemetry(
            host_cpu_percent=float(psutil.cpu_percent(interval=None)),
            effective_busy_cores=cpu_percent / 100.0,
            process_rss_bytes=rss,
            process_pss_bytes=pss_total if has_pss else None,
            process_affinity_cpu_ids=affinity,
            host_memory_used_bytes=int(memory.used),
            disk_read_bytes_per_second=read_rate,
            disk_write_bytes_per_second=write_rate,
        )

    def gpu_samples(self) -> tuple[GpuSample, ...]:
        module = self._pynvml
        if module is None:
            return ()
        samples: list[GpuSample] = []
        for gpu_id in range(int(module.nvmlDeviceGetCount())):
            handle = module.nvmlDeviceGetHandleByIndex(gpu_id)
            utilization = module.nvmlDeviceGetUtilizationRates(handle)
            memory = module.nvmlDeviceGetMemoryInfo(handle)
            try:
                processes = module.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                processes = ()
            samples.append(
                GpuSample(
                    gpu_id=gpu_id,
                    utilization_percent=float(utilization.gpu),
                    memory_used_bytes=int(memory.used),
                    memory_total_bytes=int(memory.total),
                    power_watts=float(module.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                    temperature_celsius=float(
                        module.nvmlDeviceGetTemperature(handle, module.NVML_TEMPERATURE_GPU)
                    ),
                    compute_pids=tuple(sorted({int(process.pid) for process in processes})),
                )
            )
        return tuple(samples)

    def close(self) -> None:
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()
            self._pynvml = None
