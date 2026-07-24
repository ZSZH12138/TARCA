from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.diagnostics import check_reproducibility  # noqa: E402


def test_reproducibility_is_stable_for_python_numpy_and_torch_cpu() -> None:
    first = check_reproducibility(seed=1729)
    second = check_reproducibility(seed=1729)

    assert first.status == "PASS"
    assert second.status == "PASS"
    assert first.details["samples"] == second.details["samples"]
    assert first.details["repeat_samples"] == second.details["repeat_samples"]
    assert first.details["all_match"] is True


def test_reproducibility_check_does_not_pollute_global_rng_states() -> None:
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    result = check_reproducibility(seed=1729)

    python_after = random.getstate()
    numpy_after = np.random.get_state()
    torch_after = torch.random.get_rng_state()
    assert result.status == "PASS"
    assert python_before == python_after
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch_after)
