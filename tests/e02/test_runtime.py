from pathlib import Path

import pytest

from tarca.e02.runtime import E02RuntimeAuthorizationError, dispatch_e02_runtime_command

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e02/e02_v1.yaml"


def test_e02_prepare_does_not_open_formal_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tarca.stage2.data._read_formal_storage",
        lambda: (_ for _ in ()).throw(AssertionError("formal storage opened")),
    )
    receipt = dispatch_e02_runtime_command("prepare", ROOT, CONFIG, tmp_path)
    assert receipt["formal_tasks_executed"] == 0
    assert not (tmp_path / "runtime/sealed_access_grant.json").exists()


def test_e02_launch_wrong_token_creates_no_grant(tmp_path: Path) -> None:
    with pytest.raises(E02RuntimeAuthorizationError):
        dispatch_e02_runtime_command("launch", ROOT, CONFIG, tmp_path, acknowledgement="close")
    assert not (tmp_path / "runtime/sealed_access_grant.json").exists()
