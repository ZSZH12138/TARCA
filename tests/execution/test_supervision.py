from __future__ import annotations

from pathlib import Path

from tarca.execution import (
    HostTelemetry,
    ResourceAllocation,
    ResourceRequest,
    ScientificIdentity,
    TaskSpec,
    TelemetryPolicy,
)
from tarca.execution.state import ExecutionStateStore
from tarca.execution.supervision import RuntimeSupervisor


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


class _Probe:
    def host_snapshot(self, process_id: int) -> HostTelemetry:
        return HostTelemetry(
            host_cpu_percent=64.0,
            effective_busy_cores=12.5 if process_id == 5000 else 3.5,
            process_rss_bytes=4 * 1024**3,
            process_pss_bytes=3 * 1024**3,
            process_affinity_cpu_ids=tuple(range(24)),
            host_memory_used_bytes=96 * 1024**3,
            disk_read_bytes_per_second=1024.0,
            disk_write_bytes_per_second=2048.0,
        )

    def gpu_samples(self) -> tuple[object, ...]:
        return ()

    def monitor_snapshot(self, process_id: int) -> HostTelemetry:
        assert process_id == 5000
        return HostTelemetry(
            host_cpu_percent=64.0,
            effective_busy_cores=0.25,
            process_rss_bytes=256 * 1024**2,
            process_pss_bytes=192 * 1024**2,
            process_affinity_cpu_ids=tuple(range(24)),
            host_memory_used_bytes=96 * 1024**3,
            disk_read_bytes_per_second=0.0,
            disk_write_bytes_per_second=0.0,
        )


class _FailingProbe:
    def host_snapshot(self, process_id: int) -> HostTelemetry:
        del process_id
        raise RuntimeError("NVML unavailable")

    def gpu_samples(self) -> tuple[object, ...]:
        raise RuntimeError("NVML unavailable")


def _running_store(tmp_path: Path) -> tuple[ExecutionStateStore, str]:
    store = ExecutionStateStore(tmp_path / "execution.sqlite3", artifact_verifier=lambda _: True)
    store.create_run("run-a", "graph-a")
    task = TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id="task-a",
            model_id="itransformer",
            data_id="world-a",
            seed=104729,
        ),
        phase="NEURAL_TRAIN",
        inputs=(),
        output_artifact_type="TEST_ARTIFACT",
        resource_request=ResourceRequest(
            cpu_threads=4,
            gpu_count=1,
            gpu_memory_gib=20.0,
            host_memory_gib=16.0,
        ),
    )
    attempt_id = store.enqueue_task("run-a", task, "test.execute")
    allocation = ResourceAllocation(
        cpu_threads=4,
        gpu_ids=(0,),
        host_memory_gib_limit=16.0,
        worker_id="worker-a",
    )
    assert store.claim_attempt(attempt_id, "worker-a", allocation) is not None
    store.bind_running_process(
        attempt_id,
        "worker-a",
        6000,
        process_started_at_utc=store.running_attempts("run-a")[0].heartbeat_at_utc,
    )
    return store, attempt_id


def test_supervisor_records_due_run_and_process_samples(tmp_path: Path) -> None:
    store, attempt_id = _running_store(tmp_path)
    clock = _Clock()
    supervisor = RuntimeSupervisor(
        store,
        _Probe(),
        TelemetryPolicy(sample_interval_seconds=2.0),
        clock=clock,
    )

    assert supervisor.sample_if_due("run-a", supervisor_pid=5000) is True
    assert len(store.resource_samples("run-a", attempt_id=None)) == 1
    assert len(store.resource_samples("run-a", attempt_id=attempt_id)) == 1

    clock.value += 1.9
    assert supervisor.sample_if_due("run-a", supervisor_pid=5000) is False
    assert len(store.resource_samples("run-a", attempt_id=None)) == 1

    clock.value += 0.1
    assert supervisor.sample_if_due("run-a", supervisor_pid=5000) is True
    assert len(store.resource_samples("run-a", attempt_id=None)) == 2
    assert len(store.resource_samples("run-a", attempt_id=attempt_id)) == 2
    assert store.alerts("run-a") == ()


def test_supervisor_deduplicates_probe_failure_alerts(tmp_path: Path) -> None:
    store, _ = _running_store(tmp_path)
    clock = _Clock()
    supervisor = RuntimeSupervisor(store, _FailingProbe(), clock=clock)

    assert supervisor.sample_if_due("run-a", supervisor_pid=5000) is False
    clock.value += 2.0
    assert supervisor.sample_if_due("run-a", supervisor_pid=5000) is False

    alerts = store.alerts("run-a")
    assert len(alerts) == 1
    assert alerts[0]["category"] == "TELEMETRY_UNAVAILABLE"
