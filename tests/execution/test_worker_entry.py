from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tarca.execution import worker_entry


def test_worker_entry_rejects_removed_e01_v1_execution_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARCA_EXECUTION_KIND", "e01")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tarca.execution.worker_entry",
            "--database",
            str(tmp_path / "execution.sqlite3"),
            "--repository-root",
            str(tmp_path),
            "--run-id",
            "run-a",
            "--task-id",
            "task-a",
            "--attempt-id",
            "attempt-a",
            "--worker-id",
            "worker-a",
        ],
    )

    with pytest.raises(ValueError, match="execution kind is not allowlisted"):
        worker_entry.main()
