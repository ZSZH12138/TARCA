from __future__ import annotations

import argparse
import os
from pathlib import Path

import psutil

from tarca.execution.contracts import ExecutionContext, TaskState
from tarca.execution.state import ExecutionStateStore
from tarca.execution.worker import run_worker
from tarca.stage1b.jobs import stage1b_artifact_store, stage1b_executor_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one allowlisted TARCA execution task")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--worker-id", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    affinity = os.environ.get("TARCA_CPU_AFFINITY", "")
    if affinity:
        psutil.Process().cpu_affinity([int(item) for item in affinity.split(",")])
    repository_root = arguments.repository_root.resolve()
    execution_kind = os.environ.get("TARCA_EXECUTION_KIND", "stage1b")
    if execution_kind == "e01-v2":
        from tarca.e01.v2_jobs import e01_v2_artifact_store, e01_v2_executor_registry

        verifier = e01_v2_artifact_store(repository_root).verify_artifact
        registry = e01_v2_executor_registry(repository_root)
        runtime_identity = "e01-v2-py310-cuda121"
    elif execution_kind == "stage2-v1":
        from tarca.stage2.jobs import stage2_artifact_store, stage2_executor_registry

        verifier = stage2_artifact_store(repository_root).verify_artifact
        registry = stage2_executor_registry(repository_root)
        runtime_identity = "stage2-v1-py310-cuda121"
    elif execution_kind == "e02-v1":
        from tarca.e02.jobs import e02_artifact_store, e02_executor_registry

        verifier = e02_artifact_store(repository_root).verify_artifact
        registry = e02_executor_registry(repository_root)
        runtime_identity = "e02-v1-py310-cuda121"
    elif execution_kind == "stage1b":
        verifier = stage1b_artifact_store(repository_root).verify_artifact
        registry = stage1b_executor_registry(repository_root)
        runtime_identity = "stage1b-py310-cuda121"
    else:
        raise ValueError("TARCA execution kind is not allowlisted")
    store = ExecutionStateStore(arguments.database, artifact_verifier=verifier)
    context = ExecutionContext(
        run_id=arguments.run_id,
        task_id=arguments.task_id,
        attempt_id=arguments.attempt_id,
        runtime_identity=runtime_identity,
        worker_identity=arguments.worker_id,
    )
    result = run_worker(context, store, registry)
    return 0 if result.state is TaskState.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
