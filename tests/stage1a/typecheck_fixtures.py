from __future__ import annotations

import torch

from tarca.contracts import ForecastDistribution, ForecastPredictor, WindowBatch


class CorrectPredictor:
    adapter_name = "typed-fake"
    model_hash = "a" * 64
    is_frozen = True

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        if batch.y is None:
            raise ValueError("typed fake requires labels")
        return ForecastDistribution(
            mean=batch.y,
            scale=torch.ones_like(batch.y),
            quantiles={},
            logits=None,
            samples=None,
            window_id=batch.window_id,
            target_names=batch.target_names,
        )


def accepts_predictor(value: ForecastPredictor) -> None:
    del value


accepts_predictor(CorrectPredictor())
