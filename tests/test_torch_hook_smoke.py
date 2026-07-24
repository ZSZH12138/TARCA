from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.diagnostics import check_torch_hook  # noqa: E402


def test_torch_hook_smoke_captures_shape_and_leaves_no_hooks() -> None:
    result = check_torch_hook()

    assert result.status == "PASS"
    assert result.details["module_type"] == "TransformerEncoderLayer"
    assert tuple(result.details["input_shape"]) == (2, 3, 4)
    assert tuple(result.details["captured_shape"]) == (2, 3, 4)
    assert result.details["activation_finite"] is True
    assert result.details["outputs_match_after_removal"] is True
    assert result.details["remaining_forward_hooks"] == 0


def test_torch_hook_smoke_does_not_pollute_global_rng_state() -> None:
    torch.manual_seed(991)
    state_before = torch.random.get_rng_state().clone()

    result = check_torch_hook()

    assert result.status == "PASS"
    assert torch.equal(state_before, torch.random.get_rng_state())
