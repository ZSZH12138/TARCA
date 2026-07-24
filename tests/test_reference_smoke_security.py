from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0 import reference_smoke as smoke  # noqa: E402
from tarca.stage0.reference_smoke import CommandOutcome, SmokeStatus  # noqa: E402

PLOT_COMMIT = "96dbec5f04bc03aea6e55c430eeafd5c9be27fb2"
PLOT_REPOSITORY = "https://github.com/jchang153/causal-abstractions-ot"
DIROCA_COMMIT = "7002947b4954abea1f3d11fcb6f36e7f3c43e8bd"


@pytest.mark.parametrize(
    ("branch", "status", "expected_message"),
    [
        ("main\n", "", "detached"),
        ("HEAD\n", " M experiments/transport.py\n?? malicious.py\n", "clean"),
    ],
)
def test_existing_repository_requires_detached_clean_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    status: str,
    expected_message: str,
) -> None:
    repo = tmp_path / "plot"
    (repo / ".git").mkdir(parents=True)
    commands: list[tuple[str, ...]] = []

    def repository_state(
        command: tuple[str, ...],
        *args: object,
        **kwargs: object,
    ) -> CommandOutcome:
        commands.append(command)
        stdout = ""
        if command[-3:] == ("remote", "get-url", "origin"):
            stdout = PLOT_REPOSITORY
        elif command[-2:] == ("rev-parse", "HEAD"):
            stdout = PLOT_COMMIT
        elif command[-3:] == ("rev-parse", "--abbrev-ref", "HEAD"):
            stdout = branch
        elif "status" in command:
            stdout = status
        return CommandOutcome(command, 0, stdout, "", 0.01)

    monkeypatch.setattr(smoke, "_run_process", repository_state)
    policy = smoke.BUILTIN_POLICIES["plot"]

    with pytest.raises(smoke.RepositoryPolicyError, match=expected_message) as caught:
        smoke._verify_existing_repository(repo, policy, 15)

    if status:
        assert "move or remove" in str(caught.value).lower()
        assert "rerun" in str(caught.value).lower()
        status_command = next(command for command in commands if "status" in command)
        assert "--untracked-files=all" in status_command
        assert "--ignored=matching" in status_command


def test_diroca_static_compile_keeps_reference_worktree_clean(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = smoke._run_diroca(
        tmp_path,
        15,
        DIROCA_COMMIT,
        (),
    )

    assert result.status is SmokeStatus.IMPORT_ONLY
    assert list(tmp_path.rglob("*.pyc")) == []
    assert list(tmp_path.rglob("__pycache__")) == []
