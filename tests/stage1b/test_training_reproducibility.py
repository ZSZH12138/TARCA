from __future__ import annotations

import torch

from tarca.stage1b.neural import SmallITransformer
from tarca.stage1b.training import train_candidate


def test_same_seed_produces_identical_receipt_and_predictions() -> None:
    generator = torch.Generator().manual_seed(902)
    train_x = torch.randn(24, 8, 3, generator=generator)
    train_y = train_x[:, -1:, :].repeat(1, 2, 1) ** 2
    tune_x = torch.randn(8, 8, 3, generator=generator)
    tune_y = tune_x[:, -1:, :].repeat(1, 2, 1) ** 2

    def model() -> SmallITransformer:
        return SmallITransformer(
            history_length=8,
            horizon=2,
            input_dimension=3,
            d_model=12,
            n_layers=1,
            n_heads=3,
            dropout=0.0,
        )

    first = train_candidate(
        model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
    )
    second = train_candidate(
        model(),
        train_x,
        train_y,
        tune_x,
        tune_y,
        seed=903,
        batch_size=8,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
    )

    assert first.receipt == second.receipt
    torch.testing.assert_close(
        first.model.forward_distribution(tune_x).mean,
        second.model.forward_distribution(tune_x).mean,
        rtol=0.0,
        atol=0.0,
    )
