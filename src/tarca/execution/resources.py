from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from tarca.contracts import canonical_json_hash
from tarca.execution.contracts import ResourceAllocation, TaskSpec
from tarca.execution.telemetry import GpuSample

_GIB = 1024**3


class ResourcePlanningError(RuntimeError):
    """Raised when the declared workload cannot be admitted without changing science."""


@dataclass(frozen=True, slots=True)
class HostAdmissionPolicy:
    scheduler_monitor_cores: int = 1
    system_io_reserved_cores: int = 3
    maximum_data_cores: int = 24
    maximum_host_memory_bytes: int = 200 * _GIB
    minimum_local_storage_free_bytes: int = 100 * _GIB
    initial_loader_workers_per_gpu_job: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.system_io_reserved_cores <= 4:
            raise ValueError("system and I/O reserve must use one to four CPU cores")
        if self.scheduler_monitor_cores != 1:
            raise ValueError("scheduler and monitoring must reserve exactly one CPU core")
        if not 1 <= self.maximum_data_cores <= 256:
            raise ValueError("maximum data CPU allocation must stay within one to 256 cores")
        if self.maximum_host_memory_bytes <= 0 or self.minimum_local_storage_free_bytes <= 0:
            raise ValueError("host memory and local storage limits must be positive")
        if not 1 <= self.initial_loader_workers_per_gpu_job <= 4:
            raise ValueError("initial GPU DataLoader workers must stay within one to four")


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    logical_cpu_count: int
    physical_cpu_count: int
    available_memory_bytes: int
    gpu_memory_bytes: tuple[int, ...]
    local_storage_available: bool
    local_storage_free_bytes: int

    def __post_init__(self) -> None:
        if min(self.logical_cpu_count, self.physical_cpu_count, self.available_memory_bytes) <= 0:
            raise ValueError("host resource capacity must be positive")
        if self.physical_cpu_count > self.logical_cpu_count:
            raise ValueError("physical CPU count cannot exceed logical CPU count")
        if any(memory <= 0 for memory in self.gpu_memory_bytes):
            raise ValueError("GPU memory capacities must be positive")
        if self.local_storage_free_bytes < 0:
            raise ValueError("local storage free bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class GpuPackingDecision:
    current_jobs: int
    target_jobs: int
    reason: str

    def __post_init__(self) -> None:
        if not 1 <= self.current_jobs <= 3 or not 1 <= self.target_jobs <= 3:
            raise ValueError("GPU packing supports one to three jobs per card")
        if not self.reason.strip():
            raise ValueError("GPU packing decision requires a reason")


def decide_gpu_packing(
    sample: GpuSample,
    stable_seconds: float,
    current_jobs: int,
    *,
    oom: bool = False,
    throttled: bool = False,
    throughput_loss: bool = False,
    data_wait: bool = False,
) -> GpuPackingDecision:
    if not math.isfinite(stable_seconds) or stable_seconds < 0:
        raise ValueError("GPU stable observation time must be finite and non-negative")
    if not 1 <= current_jobs <= 3:
        raise ValueError("current GPU packing must be one to three jobs")
    used_gib = sample.memory_used_bytes / _GIB
    if oom or throttled or throughput_loss or data_wait or used_gib > 20.0:
        return GpuPackingDecision(
            current_jobs=current_jobs,
            target_jobs=max(1, current_jobs - 1),
            reason="PRESSURE_OR_DATA_WAIT",
        )
    if stable_seconds < 180.0:
        return GpuPackingDecision(current_jobs, current_jobs, "OBSERVING_STABLE_WINDOW")
    if current_jobs == 1 and sample.utilization_percent < 70.0 and used_gib < 8.0:
        return GpuPackingDecision(current_jobs, 2, "LOW_UTILIZATION_AND_MEMORY")
    if current_jobs == 2 and sample.utilization_percent < 80.0 and used_gib < 18.0:
        return GpuPackingDecision(current_jobs, 3, "SECOND_PACKING_PROBE_PASSED")
    return GpuPackingDecision(current_jobs, current_jobs, "KEEP_CURRENT_PACKING")


def plan_resources(
    tasks: tuple[TaskSpec, ...],
    capacity: ResourceCapacity,
    policy: HostAdmissionPolicy | None = None,
    *,
    active: tuple[tuple[TaskSpec, ResourceAllocation], ...] = (),
) -> tuple[ResourceAllocation, ...]:
    resolved = policy or HostAdmissionPolicy()
    if (
        not capacity.local_storage_available
        or capacity.local_storage_free_bytes < resolved.minimum_local_storage_free_bytes
    ):
        raise ResourcePlanningError("suitable local storage is unavailable")
    reserved_cores = resolved.scheduler_monitor_cores + resolved.system_io_reserved_cores
    usable_cores = min(
        resolved.maximum_data_cores,
        capacity.physical_cpu_count - reserved_cores,
    )
    if usable_cores <= 0:
        raise ResourcePlanningError("no CPU cores remain after scheduler and system reserves")
    memory_budget = min(
        capacity.available_memory_bytes,
        resolved.maximum_host_memory_bytes,
    )
    active_cpu = 0
    active_memory = 0
    active_gpu_ids: set[int] = set()
    for task, allocation in active:
        request = task.resource_request
        if allocation.cpu_threads != request.cpu_threads:
            raise ResourcePlanningError("active CPU allocation drifted from its task request")
        if not math.isclose(
            allocation.host_memory_gib_limit,
            request.host_memory_gib,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ResourcePlanningError("active memory allocation drifted from its task request")
        if len(allocation.gpu_ids) != request.gpu_count:
            raise ResourcePlanningError("active GPU allocation drifted from its task request")
        if any(gpu_id >= len(capacity.gpu_memory_bytes) for gpu_id in allocation.gpu_ids):
            raise ResourcePlanningError("active GPU allocation is outside host capacity")
        if active_gpu_ids.intersection(allocation.gpu_ids):
            raise ResourcePlanningError("multiple active tasks claim the same GPU")
        active_cpu += allocation.cpu_threads
        active_memory += math.ceil(allocation.host_memory_gib_limit * _GIB)
        active_gpu_ids.update(allocation.gpu_ids)
    if active_cpu > usable_cores or active_memory > memory_budget:
        raise ResourcePlanningError("active allocations exceed the host admission budget")
    remaining_cores = usable_cores - active_cpu
    remaining_memory = memory_budget - active_memory
    free_gpu_ids = [
        gpu_id for gpu_id in range(len(capacity.gpu_memory_bytes)) if gpu_id not in active_gpu_ids
    ]
    allocations: list[ResourceAllocation] = []
    for index, task in enumerate(tasks):
        request = task.resource_request
        requested_memory = math.ceil(request.host_memory_gib * _GIB)
        if request.cpu_threads > usable_cores:
            raise ResourcePlanningError(f"task {task.task_id} exceeds the CPU admission ceiling")
        if requested_memory > memory_budget:
            raise ResourcePlanningError(f"task {task.task_id} exceeds the RAM admission ceiling")
        gpu_ids: tuple[int, ...] = ()
        if request.gpu_count:
            if request.gpu_count != 1:
                raise ResourcePlanningError(
                    "default Stage1B planning uses independent single-GPU tasks"
                )
            fitting = next(
                (
                    gpu_id
                    for gpu_id in free_gpu_ids
                    if capacity.gpu_memory_bytes[gpu_id] >= request.gpu_memory_gib * _GIB
                ),
                None,
            )
            if fitting is None:
                continue
            gpu_ids = (fitting,)
        if request.cpu_threads > remaining_cores or requested_memory > remaining_memory:
            continue
        if gpu_ids:
            free_gpu_ids.remove(gpu_ids[0])
        remaining_cores -= request.cpu_threads
        remaining_memory -= requested_memory
        allocations.append(
            ResourceAllocation(
                cpu_threads=request.cpu_threads,
                gpu_ids=gpu_ids,
                host_memory_gib_limit=request.host_memory_gib,
                worker_id=f"planned-worker-{index}",
            )
        )
    return tuple(allocations)


class DdpMode(StrEnum):
    TASK_PARALLEL = "TASK_PARALLEL"
    DUAL_GPU_DDP = "DUAL_GPU_DDP"


def select_ddp_mode(*, single_gpu_seconds: float, dual_gpu_seconds: float) -> DdpMode:
    if (
        not math.isfinite(single_gpu_seconds)
        or not math.isfinite(dual_gpu_seconds)
        or min(single_gpu_seconds, dual_gpu_seconds) <= 0
    ):
        raise ValueError("DDP probe times must be finite and positive")
    reduction = 1.0 - dual_gpu_seconds / single_gpu_seconds
    return DdpMode.DUAL_GPU_DDP if reduction >= 0.30 else DdpMode.TASK_PARALLEL


@dataclass(frozen=True, slots=True)
class PrecisionProbeResult:
    precision: Literal["FP32", "AMP_FP16"]
    samples_per_second: float
    maximum_absolute_error: float
    finite: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.samples_per_second) or self.samples_per_second <= 0:
            raise ValueError("precision probe throughput must be finite and positive")
        if not math.isfinite(self.maximum_absolute_error) or self.maximum_absolute_error < 0:
            raise ValueError("precision probe error must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PrecisionDecision:
    selected: Literal["FP32", "AMP_FP16"]
    fp32_probe: PrecisionProbeResult
    amp_probe: PrecisionProbeResult
    maximum_allowed_error: float
    decision_sha256: str

    def __post_init__(self) -> None:
        if len(self.decision_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.decision_sha256
        ):
            raise ValueError("precision decision hash must be a lowercase SHA-256")


def select_precision(
    fp32: PrecisionProbeResult,
    amp: PrecisionProbeResult,
    *,
    maximum_allowed_error: float,
) -> PrecisionDecision:
    if fp32.precision != "FP32" or amp.precision != "AMP_FP16":
        raise ValueError("precision probes must contain FP32 and AMP_FP16 in order")
    if not math.isfinite(maximum_allowed_error) or maximum_allowed_error < 0:
        raise ValueError("maximum allowed precision error must be finite and non-negative")
    if not fp32.finite:
        raise ResourcePlanningError("FP32 precision probe produced non-finite values")
    selected: Literal["FP32", "AMP_FP16"] = "FP32"
    if (
        fp32.finite
        and amp.finite
        and amp.maximum_absolute_error <= maximum_allowed_error
        and amp.samples_per_second > fp32.samples_per_second
    ):
        selected = "AMP_FP16"
    decision_hash = canonical_json_hash(
        {
            "selected": selected,
            "fp32": {
                "samples_per_second": fp32.samples_per_second,
                "maximum_absolute_error": fp32.maximum_absolute_error,
                "finite": fp32.finite,
            },
            "amp": {
                "samples_per_second": amp.samples_per_second,
                "maximum_absolute_error": amp.maximum_absolute_error,
                "finite": amp.finite,
            },
            "maximum_allowed_error": maximum_allowed_error,
        }
    )
    return PrecisionDecision(
        selected=selected,
        fp32_probe=fp32,
        amp_probe=amp,
        maximum_allowed_error=maximum_allowed_error,
        decision_sha256=decision_hash,
    )
