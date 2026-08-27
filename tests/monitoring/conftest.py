from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tarca.execution.contracts import (
    ResourceAllocation,
    ResourceRequest,
    RunPlanNode,
    ScientificIdentity,
    TaskSpec,
)
from tarca.execution.state import ExecutionStateStore
from tarca.execution.telemetry import GpuSample, ResourceSample


@pytest.fixture
def monitoring_database(tmp_path: Path) -> Path:
    path = tmp_path / "execution.sqlite3"
    store = ExecutionStateStore(path, artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    task = TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id="task-1",
            model_id="itransformer_reference",
            data_id="lorenz96_f10_v2",
            seed=104729,
        ),
        phase="NEURAL_TRAIN",
        inputs=(),
        output_artifact_type="TRAINED_NEURAL_CHECKPOINT",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=1,
            gpu_memory_gib=20.0,
            host_memory_gib=32.0,
        ),
    )
    attempt = store.enqueue_task("run-a", task, "stage1b.train_neural")
    pending_identity = task.identity.model_copy(update={"task_id": "task-2", "seed": 104759})
    store.register_run_plan(
        "run-a",
        (
            RunPlanNode(
                identity=task.identity,
                phase=task.phase,
                resource_request=task.resource_request,
                dependency_task_ids=(),
            ),
            RunPlanNode(
                identity=pending_identity,
                phase="NEURAL_SCORE",
                resource_request=task.resource_request,
                dependency_task_ids=("task-1",),
            ),
        ),
    )
    allocation = ResourceAllocation(
        cpu_threads=4,
        gpu_ids=(0,),
        host_memory_gib_limit=32.0,
        worker_id="worker-task-1",
    )
    assert store.claim_attempt(attempt, allocation.worker_id, allocation) is not None
    store.bind_running_process(
        attempt,
        allocation.worker_id,
        321,
        datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
        now=datetime(2026, 8, 26, 11, 59, 40, tzinfo=UTC),
    )
    store.record_progress(
        attempt,
        {
            "epoch": 3,
            "batch": 7,
            "completed_steps": 20,
            "total_steps": 100,
            "samples_per_second": 125.0,
            "crps": 0.01,
            "truth": "must-not-leak",
        },
        now=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
    )
    sample = ResourceSample(
            sampled_at_utc=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
            host_cpu_percent=75.0,
            effective_busy_cores=18.5,
            process_rss_bytes=12 * 1024**3,
            process_pss_bytes=10 * 1024**3,
            process_affinity_cpu_ids=tuple(range(4, 8)),
            host_memory_used_bytes=80 * 1024**3,
            gpu_samples=(
                GpuSample(
                    gpu_id=0,
                    utilization_percent=92.0,
                    memory_used_bytes=18 * 1024**3,
                    memory_total_bytes=24 * 1024**3,
                    power_watts=410.0,
                    temperature_celsius=71.0,
                    compute_pids=(321,),
                ),
                GpuSample(
                    gpu_id=1,
                    utilization_percent=5.0,
                    memory_used_bytes=1 * 1024**3,
                    memory_total_bytes=24 * 1024**3,
                    power_watts=75.0,
                    temperature_celsius=42.0,
                    compute_pids=(),
                ),
            ),
            disk_read_bytes_per_second=1000.0,
            disk_write_bytes_per_second=2000.0,
        )
    store.record_resource_sample("run-a", sample)
    store.record_resource_sample(
        "run-a",
        sample,
        attempt_id=attempt,
    )
    store.add_alert("run-a", "GPU_PRESSURE", "GPU memory pressure", attempt_id=attempt)
    return path
