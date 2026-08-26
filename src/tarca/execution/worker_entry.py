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
    parser = argparse.ArgumentParser(description="Run one allowlisted TARCA Stage1B task")
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
    verifier = stage1b_artifact_store(repository_root).verify_artifact
    store = ExecutionStateStore(arguments.database, artifact_verifier=verifier)
    context = ExecutionContext(
        run_id=arguments.run_id,
        task_id=arguments.task_id,
        attempt_id=arguments.attempt_id,
        runtime_identity="stage1b-py310-cuda121",
        worker_identity=arguments.worker_id,
    )
    result = run_worker(context, store, stage1b_executor_registry(repository_root))
    return 0 if result.state is TaskState.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
