from __future__ import annotations

from datetime import UTC, datetime, timedelta

import torch

from tarca.contracts import WindowBatch


def window_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    prefix: str = "window",
) -> WindowBatch:
    batch_size, history, dimension = x.shape
    horizon = y.shape[1]
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    feature_start = tuple(origin for _ in range(batch_size))
    feature_end = tuple(origin + timedelta(hours=history - 1) for _ in range(batch_size))
    prediction_start = tuple(origin + timedelta(hours=history) for _ in range(batch_size))
    label_end = tuple(
        origin + timedelta(hours=history + horizon - 1) for _ in range(batch_size)
    )
    forecast_time = tuple(
        tuple(origin + timedelta(hours=history + step) for step in range(horizon))
        for _ in range(batch_size)
    )
    names = tuple(f"x{index}" for index in range(dimension))
    return WindowBatch(
        x=x,
        y=y,
        observed_covariates=None,
        known_future_covariates=None,
        x_observed_mask=None,
        y_observed_mask=None,
        observed_covariates_mask=None,
        known_future_covariates_mask=None,
        regime=None,
        window_id=tuple(f"{prefix}-{index}" for index in range(batch_size)),
        input_feature_names=names,
        target_names=names,
        observed_covariate_names=(),
        known_future_covariate_names=(),
        feature_start=feature_start,
        feature_end=feature_end,
        prediction_start=prediction_start,
        label_end=label_end,
        forecast_time=forecast_time,
        metadata={"partition": "TRAIN"},
    )

