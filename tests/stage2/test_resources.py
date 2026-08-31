from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tarca.execution import (
    DdpMode,
    ResourceRequest,
    ScientificIdentity,
    TaskSpec,
    plan_resources,
    select_ddp_mode,
)
from tarca.execution.telemetry import GpuSample
from tarca.stage2.resources import (
    InferenceBundleController,
    Stage2ProbeObservation,
    Stage2ServerInventory,
    choose_stage2_capacity_plan,
    stage2_reset_time_gate,
    stage2_resource_capacity,
)
from tarca.stage2.server_probe import (
    _probe_neural_worker,
    estimate_stage2_critical_path_seconds,
    run_stage2_server_probe,
)

GIB = 1024**3
TARGET = Stage2ServerInventory(
    logical_cpu_count=28,
    physical_cpu_count=28,
    ram_bytes=224 * GIB,
    gpu_names=("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"),
    gpu_vram_bytes=(24 * GIB, 24 * GIB),
    free_storage_bytes=300 * GIB,
)
SAFE = Stage2ProbeObservation(
    cuda_available=True,
    cuda_device_count=2,
    source_hashes_verified=True,
    checkpoint_roundtrip_passed=True,
    fp32_finite=True,
    amp_finite=True,
)


def _task(name: str, *, gpu: bool, cpu: int) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v2",
            experiment_id="stage2-v1",
            task_id=name,
            model_id="ITRANSFORMER" if gpu else "CPU",
            data_id="dev",
            seed=1,
        ),
        phase="NEURAL_TRAIN" if gpu else "VALIDATION_PREDICT",
        inputs=(),
        output_artifact_type="TEST_ARTIFACT",
        resource_request=ResourceRequest(
            cpu_threads=cpu,
            gpu_count=1 if gpu else 0,
            gpu_memory_gib=20.0 if gpu else 0.0,
            host_memory_gib=32.0,
        ),
    )


def test_stage2_admits_target_server_and_reserves_four_cores() -> None:
    plan = choose_stage2_capacity_plan(TARGET, SAFE)
    assert plan.work_cpu_cores == 24
    assert plan.scheduler_monitor_cores == 1
    assert plan.system_io_cores == 3
    assert plan.gpu_worker_count == 2
    assert plan.host_memory_ceiling_gib == 200


def test_two_active_gpu_tasks_leave_sixteen_cpu_cores_for_backfill() -> None:
    capacity = stage2_resource_capacity(TARGET)
    tasks = (
        _task("gpu-a", gpu=True, cpu=4),
        _task("gpu-b", gpu=True, cpu=4),
        _task("cpu-a", gpu=False, cpu=8),
        _task("cpu-b", gpu=False, cpu=8),
    )
    allocations = plan_resources(tasks, capacity, choose_stage2_capacity_plan(TARGET, SAFE).policy)
    assert sum(item.cpu_threads for item in allocations) == 24
    assert {item.gpu_ids for item in allocations if item.gpu_ids} == {(0,), (1,)}


def test_stage2_refuses_missing_second_gpu_and_storage_below_200_gib() -> None:
    with pytest.raises(RuntimeError, match="two RTX 4090"):
        choose_stage2_capacity_plan(
            replace(TARGET, gpu_names=(TARGET.gpu_names[0],), gpu_vram_bytes=(24 * GIB,)), SAFE
        )
    with pytest.raises(RuntimeError, match="storage"):
        choose_stage2_capacity_plan(replace(TARGET, free_storage_bytes=199 * GIB), SAFE)


def test_inference_bundle_controller_backs_off_on_oom() -> None:
    controller = InferenceBundleController()
    sample = GpuSample(
        gpu_id=0,
        utilization_percent=20.0,
        memory_used_bytes=6 * GIB,
        memory_total_bytes=24 * GIB,
        temperature_celsius=50.0,
        power_watts=120.0,
        compute_pids=(),
    )
    decision = controller.observe(sample, stable_seconds=181.0, current_jobs=2, oom=True)
    assert decision.target_jobs == 1


def test_ddp_boundary_is_exactly_thirty_percent() -> None:
    assert (
        select_ddp_mode(single_gpu_seconds=1000.0, dual_gpu_seconds=701.0)
        is DdpMode.TASK_PARALLEL
    )
    assert (
        select_ddp_mode(single_gpu_seconds=1000.0, dual_gpu_seconds=700.0)
        is DdpMode.DUAL_GPU_DDP
    )


def test_eta_plus_one_hour_equal_to_rental_boundary_refuses_launch() -> None:
    with pytest.raises(RuntimeError, match="reset boundary"):
        stage2_reset_time_gate(estimated_remaining_seconds=23 * 3600, remaining_rental_hours=24)


def test_server_probe_projects_all_six_neural_runs_with_conservative_overhead() -> None:
    estimate = estimate_stage2_critical_path_seconds(
        train_window_count=30_600,
        initialization_count=3,
        maximum_epochs={"PATCHTST": 100, "ITRANSFORMER": 100},
        batches_per_second={"PATCHTST": 10.0, "ITRANSFORMER": 8.0},
        batch_sizes={"PATCHTST": 64, "ITRANSFORMER": 32},
        checkpoint_seconds={"PATCHTST": 2.0, "ITRANSFORMER": 3.0},
        fixed_overhead_seconds=4 * 3600,
        safety_multiplier=1.35,
    )
    itransformer = 30_600 * 3 * 100 / (8.0 * 32) + 3 * 3.0
    assert estimate == pytest.approx(itransformer * 1.35 + 4 * 3600)


def test_server_probe_orchestrates_two_models_and_enforces_eta_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[tuple[str, int]] = []

    class ImmediateFuture:
        def __init__(self, value: dict[str, Any]) -> None:
            self.value = value

        def result(self) -> dict[str, Any]:
            return self.value

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            assert max_workers == 2
            assert mp_context is marker

        def __enter__(self) -> "ImmediateExecutor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, *args: object) -> ImmediateFuture:
            del function
            model_id = str(args[-2])
            assert isinstance(args[-1], int)
            gpu_id = args[-1]
            submitted.append((model_id, gpu_id))
            batch_size = 64 if model_id == "PATCHTST" else 32
            return ImmediateFuture(
                {
                    "model_id": model_id,
                    "gpu_id": gpu_id,
                    "batch_size": batch_size,
                    "batches_per_second": 100.0,
                    "checkpoint_seconds": 1.0,
                    "checkpoint_reload_finite": True,
                }
            )

    marker = object()
    monkeypatch.setattr("tarca.stage2.server_probe.multiprocessing.get_context", lambda _: marker)
    monkeypatch.setattr("tarca.stage2.server_probe.ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tarca.stage2.server_probe._linear_probe_seconds", lambda: 0.25)
    observation = run_stage2_server_probe(
        Path.cwd(),
        Path("configs/stage2/stage2_v1.yaml"),
        tmp_path / "runtime",
        remaining_rental_hours=24,
    )
    assert submitted == [("PATCHTST", 0), ("ITRANSFORMER", 1)]
    assert observation["probe_contract"].startswith("stage2-v1")
    assert observation["train_window_count"] == 30_600
    assert observation["linear_probe_seconds"] == 0.25
    assert observation["eta_gate_passed"] is True


def test_exact_neural_probe_worker_runs_forward_backward_and_checkpoint_on_device_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProbabilisticPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.location = torch.nn.Parameter(torch.tensor(0.0))

        def forward_distribution(self, histories: torch.Tensor) -> SimpleNamespace:
            shape = (histories.shape[0], 24, 8)
            mean = self.location * torch.ones(shape, device=histories.device)
            return SimpleNamespace(mean=mean, scale=torch.ones_like(mean))

        def to(self, *args: object, **kwargs: object) -> "FakeProbabilisticPredictor":
            del args, kwargs
            return self

    original_randn = torch.randn
    original_load = torch.load
    monkeypatch.setattr("tarca.stage2.server_probe.torch.cuda.set_device", lambda _: None)
    monkeypatch.setattr("tarca.stage2.server_probe.torch.cuda.synchronize", lambda _: None)
    monkeypatch.setattr(
        "tarca.stage2.server_probe.torch.randn",
        lambda *shape, **kwargs: original_randn(*shape),
    )
    monkeypatch.setattr(
        "tarca.stage2.server_probe.torch.load",
        lambda path, **kwargs: original_load(path, map_location="cpu"),
    )
    monkeypatch.setattr(
        "tarca.stage2.server_probe._new_neural",
        lambda *args, **kwargs: FakeProbabilisticPredictor(),
    )
    result = _probe_neural_worker(
        str(Path.cwd()),
        str(Path("configs/stage2/stage2_v1.yaml").resolve()),
        str(tmp_path),
        "ITRANSFORMER",
        1,
    )
    assert result["model_id"] == "ITRANSFORMER"
    assert result["gpu_id"] == 1
    assert isinstance(result["batches_per_second"], float)
    assert result["batches_per_second"] > 0
    assert result["checkpoint_reload_finite"] is True
    assert not tuple(tmp_path.glob("probe-*.pt"))
