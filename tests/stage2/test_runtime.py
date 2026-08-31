from pathlib import Path

import pytest

from tarca.stage2.runtime import (
    Stage2RuntimeAuthorizationError,
    dispatch_stage2_runtime_command,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage2/stage2_v1.yaml"


def test_stage2_launch_requires_exact_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(Stage2RuntimeAuthorizationError):
        dispatch_stage2_runtime_command("launch", ROOT, CONFIG, tmp_path, acknowledgement="close")
    assert not (tmp_path / "runtime").exists()


def test_stage2_prepare_is_idempotent_and_executes_no_formal_tasks(tmp_path: Path) -> None:
    first = dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    second = dispatch_stage2_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    assert first == second
    assert first["formal_tasks_executed"] == 0
    assert first["status"] == "PREPARED"
