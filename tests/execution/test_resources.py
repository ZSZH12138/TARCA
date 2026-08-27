from __future__ import annotations

import pytest

from tarca.execution import ResourceAllocation, ResourceRequest, ScientificIdentity, TaskSpec
from tarca.execution.resources import (
    DdpMode,
    HostAdmissionPolicy,
    PrecisionProbeResult,
    ResourceCapacity,
    ResourcePlanningError,
    decide_gpu_packing,
    plan_resources,
    select_ddp_mode,
    select_precision,
)
from tarca.execution.telemetry import GpuSample


def _gib(value: float) -> int:
    return int(value * 1024**3)


def _gpu_sample(*, utilization: float, used_gib: float) -> GpuSample:
    return GpuSample(
        gpu_id=0,
        utilization_percent=utilization,
        memory_used_bytes=_gib(used_gib),
        memory_total_bytes=_gib(24.0),
        power_watts=300.0,
        temperature_celsius=65.0,
        compute_pids=(1001,),
    )


def _gpu_task(task_id: str) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="TARCA-E2E-STAGE-PROTOCOL-2.0",
            experiment_id="stage1b-qualification-v2",
            task_id=task_id,
            model_id="itransformer_reference",
            data_id="lorenz96_f10_v2",
            seed=104729,
        ),
        phase="NEURAL_TRAIN",
        inputs=(),
        output_artifact_type="trained_neural_checkpoint",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=1,
            gpu_memory_gib=20.0,
            host_memory_gib=32.0,
        ),
    )


def test_low_utilization_admits_second_and_then_third_job() -> None:
    second = decide_gpu_packing(_gpu_sample(utilization=62.0, used_gib=7.0), 181.0, 1)
    assert second.target_jobs == 2
    third = decide_gpu_packing(_gpu_sample(utilization=75.0, used_gib=17.0), 181.0, 2)
    assert third.target_jobs == 3


def test_gpu_pressure_or_data_wait_reduces_packing() -> None:
    memory_pressure = decide_gpu_packing(
        _gpu_sample(utilization=95.0, used_gib=21.0),
        200.0,
        3,
    )
    assert memory_pressure.target_jobs == 2
    data_wait = decide_gpu_packing(
        _gpu_sample(utilization=30.0, used_gib=6.0),
        200.0,
        2,
        data_wait=True,
    )
    assert data_wait.target_jobs == 1


def test_resource_plan_starts_one_independent_task_per_gpu() -> None:
    capacity = ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=_gib(224.0),
        gpu_memory_bytes=(_gib(24.0), _gib(24.0)),
        local_storage_available=True,
        local_storage_free_bytes=_gib(1000.0),
    )
    tasks = (_gpu_task("task-a"), _gpu_task("task-b"), _gpu_task("task-c"))
    allocations = plan_resources(tasks, capacity, HostAdmissionPolicy())
    assert tuple(allocation.gpu_ids for allocation in allocations) == ((0,), (1,))
    assert all(allocation.cpu_threads == 4 for allocation in allocations)


def test_resource_plan_subtracts_active_cpu_memory_and_gpu_allocations() -> None:
    capacity = ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=_gib(224.0),
        gpu_memory_bytes=(_gib(24.0), _gib(24.0)),
        local_storage_available=True,
        local_storage_free_bytes=_gib(1000.0),
    )
    active_task = _gpu_task("active-task")
    active_allocation = ResourceAllocation(
        cpu_threads=4,
        gpu_ids=(0,),
        host_memory_gib_limit=32.0,
        worker_id="active-worker",
    )

    allocations = plan_resources(
        (_gpu_task("queued-a"), _gpu_task("queued-b")),
        capacity,
        active=((active_task, active_allocation),),
    )

    assert tuple(allocation.gpu_ids for allocation in allocations) == ((1,),)


def test_resource_plan_keeps_second_24_core_task_queued() -> None:
    capacity = ResourceCapacity(
        logical_cpu_count=56,
        physical_cpu_count=28,
        available_memory_bytes=_gib(224.0),
        gpu_memory_bytes=(_gib(24.0), _gib(24.0)),
        local_storage_available=True,
        local_storage_free_bytes=_gib(1000.0),
    )
    cpu_task = _gpu_task("cpu-task").model_copy(
        update={
            "resource_request": ResourceRequest(
                cpu_threads=24,
                gpu_count=0,
                gpu_memory_gib=0.0,
                host_memory_gib=96.0,
            )
        }
    )
    active_allocation = ResourceAllocation(
        cpu_threads=24,
        gpu_ids=(),
        host_memory_gib_limit=96.0,
        worker_id="active-worker",
    )

    assert (
        plan_resources(
            (
                cpu_task.model_copy(
                    update={
                        "identity": cpu_task.identity.model_copy(update={"task_id": "queued-task"})
                    }
                ),
            ),
            capacity,
            active=((cpu_task, active_allocation),),
        )
        == ()
    )


def test_resource_plan_fails_without_local_storage() -> None:
    capacity = ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=_gib(224.0),
        gpu_memory_bytes=(_gib(24.0), _gib(24.0)),
        local_storage_available=False,
        local_storage_free_bytes=0,
    )
    try:
        plan_resources((_gpu_task("task-a"),), capacity)
    except ResourcePlanningError as error:
        assert "local storage" in str(error)
    else:
        raise AssertionError("resource planning must fail closed without local storage")


def test_ddp_requires_at_least_thirty_percent_wall_time_reduction() -> None:
    assert select_ddp_mode(single_gpu_seconds=100.0, dual_gpu_seconds=71.0) is DdpMode.TASK_PARALLEL
    assert select_ddp_mode(single_gpu_seconds=100.0, dual_gpu_seconds=69.0) is DdpMode.DUAL_GPU_DDP


def test_precision_is_selected_only_from_preflight_evidence() -> None:
    fp32 = PrecisionProbeResult("FP32", samples_per_second=100.0, maximum_absolute_error=0.0)
    safe_amp = PrecisionProbeResult(
        "AMP_FP16",
        samples_per_second=160.0,
        maximum_absolute_error=5e-5,
    )
    unsafe_amp = PrecisionProbeResult(
        "AMP_FP16",
        samples_per_second=170.0,
        maximum_absolute_error=5e-2,
    )
    assert select_precision(fp32, safe_amp, maximum_allowed_error=1e-3).selected == "AMP_FP16"
    assert select_precision(fp32, unsafe_amp, maximum_allowed_error=1e-3).selected == "FP32"
    with pytest.raises(ResourcePlanningError, match="FP32"):
        select_precision(
            PrecisionProbeResult(
                "FP32",
                samples_per_second=100.0,
                maximum_absolute_error=0.0,
                finite=False,
            ),
            safe_amp,
            maximum_allowed_error=1e-3,
        )
