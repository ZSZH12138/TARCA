from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tarca.contracts import ArtifactRef, canonical_json_hash
from tarca.execution.contracts import (
    ExecutionContext,
    ResourceAllocation,
    ResourceRequest,
    ScientificIdentity,
    TaskSpec,
)
from tarca.execution.registry import ExecutorRegistry
from tarca.execution.resources import ResourceCapacity
from tarca.execution.scheduler import (
    LocalMultiProcessBackend,
    PsutilProcessProbe,
    RunTerminalStatus,
    Scheduler,
    SynchronousTestBackend,
)
from tarca.execution.state import AttemptState, ExecutionStateStore
from tarca.stage1b.compiler import compile_stage1b_graph, repository_v2_inputs
from tarca.stage1b.jobs import stage1b_executor_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _task(
    index: int,
    *,
    gpu: bool = True,
    gpu_memory_gib: float = 4.0,
) -> TaskSpec:
    return TaskSpec(
        identity=ScientificIdentity(
            protocol_id="tarca-v1",
            experiment_id="stage1b-v2",
            task_id=f"task-{index}",
            model_id="itransformer" if gpu else "var",
            data_id="world-a",
            seed=index,
        ),
        phase="NEURAL_TRAIN" if gpu else "VAR_SCORE",
        inputs=(),
        output_artifact_type="TEST_ARTIFACT",
        resource_request=ResourceRequest(
            cpu_threads=2,
            gpu_count=1 if gpu else 0,
            gpu_memory_gib=gpu_memory_gib if gpu else 0.0,
            host_memory_gib=2.0,
        ),
    )


def _capacity() -> ResourceCapacity:
    return ResourceCapacity(
        logical_cpu_count=28,
        physical_cpu_count=28,
        available_memory_bytes=224 * 1024**3,
        gpu_memory_bytes=(24 * 1024**3, 24 * 1024**3),
        local_storage_available=True,
        local_storage_free_bytes=500 * 1024**3,
    )


class _FakeProcess:
    pid = 321

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return -15 if self.terminated or self.killed else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _RecordingBackend:
    backend_id = "recording"

    def __init__(self) -> None:
        self.launched: list[object] = []

    def launch(self, task: object, database_path: Path) -> object:
        del database_path
        self.launched.append(task)
        return _FakeProcess()

    def poll(self) -> tuple[object, ...]:
        return ()


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def sample_if_due(self, run_id: str, supervisor_pid: int) -> bool:
        self.calls.append((run_id, supervisor_pid))
        return True


def test_scheduler_never_exposes_scientific_result_columns() -> None:
    lowered = {column.lower() for column in Scheduler.visible_columns}
    assert lowered.isdisjoint({"crps", "nll", "mae", "ranking", "truth", "best_seed"})


def test_every_compiled_stage1b_executor_is_exactly_allowlisted() -> None:
    graph = compile_stage1b_graph(repository_v2_inputs(REPOSITORY_ROOT))
    registry = stage1b_executor_registry(REPOSITORY_ROOT)

    assert set(registry.keys) == {node.executor_key for node in graph.nodes}


def test_two_gpus_launch_two_of_three_ready_gpu_jobs(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    for index in range(3):
        store.enqueue_task("run-a", _task(index), "test.execute")
    scheduler = Scheduler(store, _RecordingBackend(), _capacity())

    launches = scheduler.tick("run-a")

    assert len(launches) == 2
    assert {launch.task.allocation.gpu_ids for launch in launches} == {(0,), (1,)}
    assert store.attempt_state("task-2-attempt-1") is AttemptState.READY


def test_scheduler_fills_idle_gpus_before_an_older_cpu_only_job(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    data_task = _task(90, gpu=False).model_copy(
        update={
            "phase": "DATA_GENERATE",
            "resource_request": ResourceRequest(
                cpu_threads=16,
                gpu_count=0,
                gpu_memory_gib=0.0,
                host_memory_gib=96.0,
            ),
        }
    )
    data_attempt = store.enqueue_task("run-a", data_task, "test.execute")
    store.claim_attempt(
        data_attempt,
        "data-worker",
        ResourceAllocation(
            cpu_threads=16,
            gpu_ids=(),
            host_memory_gib_limit=96.0,
            worker_id="data-worker",
        ),
    )
    older_cpu = _task(91, gpu=False).model_copy(
        update={
            "resource_request": ResourceRequest(
                cpu_threads=8,
                gpu_count=0,
                gpu_memory_gib=0.0,
                host_memory_gib=32.0,
            )
        }
    )
    store.enqueue_task("run-a", older_cpu, "test.execute")
    store.enqueue_task("run-a", _task(92, gpu_memory_gib=20.0), "test.execute")
    store.enqueue_task("run-a", _task(93, gpu_memory_gib=20.0), "test.execute")

    launches = Scheduler(store, _RecordingBackend(), _capacity()).tick("run-a")

    assert tuple(launch.task.task_id for launch in launches) == ("task-92", "task-93")
    assert {launch.task.allocation.gpu_ids for launch in launches} == {(0,), (1,)}
    assert store.attempt_state("task-91-attempt-1") is AttemptState.READY


def test_repeated_ticks_never_reuse_resources_held_by_running_attempts(
    tmp_path: Path,
) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    for index in range(8):
        store.enqueue_task(
            "run-a",
            _task(index, gpu_memory_gib=20.0),
            "test.execute",
        )
    scheduler = Scheduler(store, _RecordingBackend(), _capacity())

    launches_per_tick = tuple(len(scheduler.tick("run-a")) for _ in range(5))

    assert launches_per_tick == (2, 0, 0, 0, 0)
    assert store.run_attempt_counts("run-a") == {"READY": 6, "RUNNING": 2}


def test_scheduler_samples_after_launch_and_when_no_work_is_ready(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(0), "test.execute")
    supervisor = _RecordingSupervisor()
    scheduler = Scheduler(store, _RecordingBackend(), _capacity(), supervisor=supervisor)

    scheduler.tick("run-a")
    scheduler.tick("run-a")

    assert len(supervisor.calls) == 2
    assert {run_id for run_id, _ in supervisor.calls} == {"run-a"}
    assert all(pid > 0 for _, pid in supervisor.calls)


def test_completed_gpu_attempt_releases_exactly_one_card(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    for index in range(3):
        store.enqueue_task(
            "run-a",
            _task(index, gpu_memory_gib=20.0),
            "test.execute",
        )
    scheduler = Scheduler(store, _RecordingBackend(), _capacity())
    first = scheduler.tick("run-a")
    released = first[0]
    store.complete_attempt(
        released.task.attempt_id,
        ArtifactRef(
            artifact_id="released-artifact",
            artifact_type="TEST_ARTIFACT",
            content_hash="a" * 64,
            schema_version="1.0.0",
            relative_path="artifacts/released.bin",
        ),
    )

    second = scheduler.tick("run-a")

    assert len(second) == 1
    assert second[0].task.allocation.gpu_ids == released.task.allocation.gpu_ids
    assert store.run_attempt_counts("run-a") == {"COMPLETED": 1, "RUNNING": 2}


def test_backend_choice_does_not_change_scientific_identity(tmp_path: Path) -> None:
    task = _task(1)
    store_a = ExecutionStateStore(tmp_path / "a.sqlite", artifact_verifier=lambda ref: True)
    store_b = ExecutionStateStore(tmp_path / "b.sqlite", artifact_verifier=lambda ref: True)
    for store, run_id in ((store_a, "run-a"), (store_b, "run-b")):
        store.create_run(run_id, "graph-a")
        store.enqueue_task(run_id, task, "test.execute")
    launch_a = Scheduler(store_a, _RecordingBackend(), _capacity()).tick("run-a")[0]
    launch_b = Scheduler(store_b, _RecordingBackend(), _capacity()).tick("run-b")[0]

    assert launch_a.scientific_identity_sha256 == launch_b.scientific_identity_sha256
    assert launch_a.scientific_identity_sha256 == canonical_json_hash(task)


class _PopenRecorder:
    def __init__(self) -> None:
        self.arguments: tuple[str, ...] | None = None
        self.options: dict[str, Any] = {}

    def __call__(self, arguments: tuple[str, ...], **options: Any) -> _FakeProcess:
        self.arguments = arguments
        self.options = options
        return _FakeProcess()


def test_local_backend_launches_tuple_without_shell_and_sets_resource_environment(
    tmp_path: Path,
) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(1), "test.execute")
    recorder = _PopenRecorder()
    backend = LocalMultiProcessBackend(
        repository_root=tmp_path,
        popen_factory=recorder,
        python_executable="python",
        cpu_ids=tuple(range(24)),
        environment_overrides={"TARCA_EXECUTION_KIND": "stage2-v1"},
    )

    Scheduler(store, backend, _capacity()).tick("run-a")

    assert isinstance(recorder.arguments, tuple)
    assert recorder.options["shell"] is False
    environment = recorder.options["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] in {"0", "1"}
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert environment["TARCA_CPU_AFFINITY"]
    assert environment["TARCA_EXECUTION_KIND"] == "stage2-v1"


def test_local_backend_terminates_every_active_worker(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite", artifact_verifier=lambda ref: True)
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(1), "test.execute")
    recorder = _PopenRecorder()
    backend = LocalMultiProcessBackend(
        repository_root=tmp_path,
        popen_factory=recorder,
        python_executable="python",
        cpu_ids=tuple(range(24)),
    )
    Scheduler(store, backend, _capacity()).tick("run-a")

    terminated = backend.terminate_all(timeout_seconds=0.01)

    assert terminated == ("task-1-attempt-1",)
    assert recorder.options


def test_psutil_process_probe_reads_worker_identity_from_argv() -> None:
    started = datetime(2026, 8, 28, 1, 2, 3, tzinfo=UTC)

    class Process:
        def cmdline(self) -> list[str]:
            return [
                "python",
                "-m",
                "tarca.execution.worker_entry",
                "--run-id",
                "run-a",
                "--task-id",
                "task-a",
            ]

        def create_time(self) -> float:
            return started.timestamp()

    identity = PsutilProcessProbe(process_factory=lambda _pid: Process()).inspect(321)

    assert identity is not None
    assert identity.pid == 321
    assert identity.run_id == "run-a"
    assert identity.task_id == "task-a"
    assert identity.process_started_at_utc == started


def test_synchronous_backend_reaches_terminal_completion(tmp_path: Path) -> None:
    artifacts: dict[str, bytes] = {}

    def executor(task: TaskSpec, context: ExecutionContext, progress: object) -> ArtifactRef:
        del context, progress
        content_hash = canonical_json_hash(task)
        ref = ArtifactRef(
            artifact_id=f"test-{content_hash}",
            artifact_type=task.output_artifact_type,
            content_hash=content_hash,
            schema_version="1.0.0",
            relative_path=f"artifacts/{content_hash}.bin",
        )
        artifacts[ref.artifact_id] = b"verified"
        return ref

    store = ExecutionStateStore(
        tmp_path / "state.sqlite",
        artifact_verifier=lambda ref: ref.artifact_id in artifacts,
    )
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(1, gpu=False), "test.execute")
    backend = SynchronousTestBackend(store, ExecutorRegistry({"test.execute": executor}))
    scheduler = Scheduler(store, backend, _capacity(), poll_interval_seconds=0.001)

    assert scheduler.run_until_terminal("run-a") is RunTerminalStatus.COMPLETED


def test_scheduler_retries_one_transient_worker_failure(tmp_path: Path) -> None:
    calls = 0
    artifacts: set[str] = set()

    def executor(task: TaskSpec, context: ExecutionContext, progress: object) -> ArtifactRef:
        nonlocal calls
        del context, progress
        calls += 1
        if calls == 1:
            raise OSError("temporary local I/O failure")
        content_hash = canonical_json_hash(task)
        artifact = ArtifactRef(
            artifact_id=f"test-{content_hash}",
            artifact_type=task.output_artifact_type,
            content_hash=content_hash,
            schema_version="1.0.0",
            relative_path=f"artifacts/{content_hash}.bin",
        )
        artifacts.add(artifact.artifact_id)
        return artifact

    store = ExecutionStateStore(
        tmp_path / "state.sqlite",
        artifact_verifier=lambda ref: ref.artifact_id in artifacts,
    )
    store.create_run("run-a", "graph-a")
    store.enqueue_task("run-a", _task(1, gpu=False), "test.execute")
    scheduler = Scheduler(
        store,
        SynchronousTestBackend(store, ExecutorRegistry({"test.execute": executor})),
        _capacity(),
        poll_interval_seconds=0.001,
    )

    assert scheduler.run_until_terminal("run-a") is RunTerminalStatus.COMPLETED
    assert calls == 2
