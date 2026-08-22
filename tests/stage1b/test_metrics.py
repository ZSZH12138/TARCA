from __future__ import annotations

import math

import pytest
import torch

from tarca.stage1b.metrics import gaussian_crps, gaussian_nll, paired_bootstrap_interval


def test_standard_normal_crps_at_mean_matches_closed_form() -> None:
    value = gaussian_crps(
        mean=torch.tensor(0.0),
        scale=torch.tensor(1.0),
        target=torch.tensor(0.0),
    )

    assert float(value) == pytest.approx((math.sqrt(2.0) - 1.0) / math.sqrt(math.pi))


def test_gaussian_nll_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_paired_bootstrap_is_deterministic_and_uses_whole_units() -> None:
    improvements = torch.tensor([0.05, 0.07, 0.09, 0.11], dtype=torch.float64)

    first = paired_bootstrap_interval(
        improvements,
        replicates=2000,
        confidence_level=0.95,
        seed=1001,
    )
    second = paired_bootstrap_interval(
        improvements,
        replicates=2000,
        confidence_level=0.95,
        seed=1001,
    )

    assert first == second
    assert first.lower > 0
    assert first.unit_count == 4

