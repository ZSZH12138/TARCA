from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RuntimeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JobStatusView(RuntimeView):
    task_id: str
    phase: str
    world_id: str | None
    model_id: str | None
    seed: int | None
    state: str
    pid: int | None
    alive: bool
    gpu_ids: tuple[int, ...]
    expected_cpu_cores: int
    actual_effective_busy_cores: float
    expected_ram_bytes: int
    actual_rss_bytes: int
    expected_vram_bytes: int
    actual_vram_bytes: int
    epoch: int | None
    batch: int | None
    heartbeat_at_utc: datetime | None
    retry_count: int
    eta_seconds: float | None
    error_category: str | None


class RunSummaryView(RuntimeView):
    run_id: str
    graph_id: str
    status: str
    phase: str | None
    total_tasks: int
    completed_tasks: int
    running_tasks: int
    failed_tasks: int
    pending_tasks: int
    progress_fraction: float
    eta_seconds: float | None
    eta_status: Literal["CALIBRATING", "AVAILABLE", "COMPLETE", "FAILED"]
    created_at_utc: datetime


class ResourceView(RuntimeView):
    resource_id: str
    label: str
    kind: Literal["HOST", "GPU"]
    expected_cpu_cores: int
    actual_effective_busy_cores: float
    expected_memory_bytes: int
    actual_memory_bytes: int
    utilization_percent: float
    temperature_celsius: float | None
    power_watts: float | None
    active_processes: int
    sampled_at_utc: datetime | None


class AlertView(RuntimeView):
    alert_id: int
    task_id: str | None
    category: str
    message: str
    created_at_utc: datetime


class RuntimeSnapshotView(RuntimeView):
    run: RunSummaryView
    jobs: tuple[JobStatusView, ...]
    resources: tuple[ResourceView, ...]
    alerts: tuple[AlertView, ...]
