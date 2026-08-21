from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest
import torch

from tarca.contracts import WindowBatch, validate_window_batch


def _valid_batch() -> WindowBatch:
    start = datetime(2026, 8, 21, tzinfo=UTC)
    x = torch.tensor(
        [[[1.0, 10.0], [2.0, 11.0]], [[3.0, 12.0], [4.0, 13.0]]],
        requires_grad=True,
    )
    y = torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]], requires_grad=True)
    observed_covariates = torch.tensor([[[10.0], [11.0]], [[12.0], [13.0]]], requires_grad=True)
    known_future_covariates = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]], requires_grad=True)
    return WindowBatch(
        x=x,
        y=y,
        observed_covariates=observed_covariates,
        known_future_covariates=known_future_covariates,
        x_observed_mask=torch.ones_like(x, dtype=torch.bool),
        y_observed_mask=torch.ones_like(y, dtype=torch.bool),
        observed_covariates_mask=torch.ones_like(observed_covariates, dtype=torch.bool),
        known_future_covariates_mask=torch.ones_like(known_future_covariates, dtype=torch.bool),
        regime=torch.tensor([0, 1], dtype=torch.int64),
        window_id=("window-0", "window-1"),
        input_feature_names=("load", "temperature"),
        target_names=("load",),
        observed_covariate_names=("temperature",),
        known_future_covariate_names=("holiday",),
        feature_start=(start, start + timedelta(days=1)),
        feature_end=(start + timedelta(hours=1), start + timedelta(days=1, hours=1)),
        prediction_start=(
            start + timedelta(hours=2),
            start + timedelta(days=1, hours=2),
        ),
        label_end=(
            start + timedelta(hours=3),
            start + timedelta(days=1, hours=3),
        ),
        forecast_time=(
            (start + timedelta(hours=2), start + timedelta(hours=3)),
            (
                start + timedelta(days=1, hours=2),
                start + timedelta(days=1, hours=3),
            ),
        ),
        metadata={"partition": "TRAIN", "source": "stage1a-test"},
    )


def test_window_validation_preserves_tensor_identity_and_gradient_state() -> None:
    batch = _valid_batch()
    tensors = (
        batch.x,
        batch.y,
        batch.observed_covariates,
        batch.known_future_covariates,
    )
    before = tuple(
        (tensor, tensor.data_ptr(), tensor.dtype, tensor.device, tensor.requires_grad)
        for tensor in tensors
        if tensor is not None
    )

    result = validate_window_batch(batch)

    assert result is batch
    after = tuple(
        (tensor, tensor.data_ptr(), tensor.dtype, tensor.device, tensor.requires_grad)
        for tensor in tensors
        if tensor is not None
    )
    assert after == before


def test_window_validation_rejects_zero_forecast_horizon() -> None:
    batch = _valid_batch()
    empty_forecast_time = tuple(() for _ in batch.window_id)

    with pytest.raises(ValueError, match="forecast horizon must be positive"):
        validate_window_batch(
            replace(
                batch,
                y=None,
                y_observed_mask=None,
                known_future_covariates=None,
                known_future_covariates_mask=None,
                known_future_covariate_names=(),
                forecast_time=empty_forecast_time,
            )
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"y": torch.ones((2, 3, 1))}, "shape"),
        ({"y": torch.ones((2, 2, 1), dtype=torch.float64)}, "dtype"),
        ({"x_observed_mask": torch.ones((2, 2, 2))}, "bool"),
        ({"known_future_covariate_names": ("load",)}, "must not overlap"),
    ),
)
def test_window_validation_rejects_incompatible_shapes_and_semantics(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_window_batch(replace(_valid_batch(), **change))


def test_window_validation_rejects_nonfinite_values() -> None:
    batch = _valid_batch()
    invalid_x = batch.x.detach().clone()
    invalid_x[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        validate_window_batch(replace(batch, x=invalid_x))


@pytest.mark.parametrize(
    "change",
    (
        {"y": torch.ones((2, 2))},
        {"observed_covariates": torch.ones((2, 2))},
        {"known_future_covariates": torch.ones((2, 2))},
    ),
)
def test_window_validation_rejects_non_rank_three_data(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="rank-3"):
        validate_window_batch(replace(_valid_batch(), **change))


def test_window_validation_rejects_non_utc_or_non_monotonic_time() -> None:
    batch = _valid_batch()
    non_utc = datetime(2026, 8, 21, tzinfo=timezone(timedelta(hours=8)))
    with pytest.raises(ValueError, match="UTC"):
        validate_window_batch(replace(batch, feature_start=(non_utc, non_utc)))

    reversed_times = tuple(tuple(reversed(times)) for times in batch.forecast_time)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_window_batch(replace(batch, forecast_time=reversed_times))


def test_window_validation_rejects_duplicate_window_ids() -> None:
    with pytest.raises(ValueError, match="window_id must be unique"):
        validate_window_batch(replace(_valid_batch(), window_id=("same", "same")))
