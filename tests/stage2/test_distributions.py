from __future__ import annotations

import torch

from tarca.stage2.distributions import gaussian_quantiles, residual_scale


def test_residual_scale_is_rms_with_floor_and_elementwise_ceiling() -> None:
    residuals = torch.tensor(
        [
            [[0.0, 4.0], [3.0, 0.0]],
            [[0.0, 0.0], [4.0, 0.0]],
        ]
    )
    ceiling = torch.tensor([[5.0, 2.0], [3.0, 5.0]])

    scale = residual_scale(residuals, floor=0.25, ceiling=ceiling)

    expected = torch.tensor([[0.25, 2.0], [3.0, 0.25]], dtype=torch.float64)
    assert torch.equal(scale, expected)


def test_gaussian_quantiles_are_aligned_and_non_crossing() -> None:
    mean = torch.zeros((2, 3, 2), dtype=torch.float32)
    scale = torch.full_like(mean, 2.0)

    quantiles = gaussian_quantiles(mean, scale, (0.025, 0.1, 0.9, 0.975))

    assert tuple(quantiles) == (0.025, 0.1, 0.9, 0.975)
    assert all(value.shape == mean.shape for value in quantiles.values())
    assert bool((quantiles[0.025] < quantiles[0.1]).all())
    assert bool((quantiles[0.1] < quantiles[0.9]).all())
    assert bool((quantiles[0.9] < quantiles[0.975]).all())

