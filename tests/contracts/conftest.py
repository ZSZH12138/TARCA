from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_valid_window_batch_kwargs(**overrides: object) -> dict[str, object]:
    """Build an independently specified valid WindowBatch input."""
    batch_size, history, horizon = 2, 3, 2
    feature_start = datetime(2025, 1, 1, tzinfo=UTC)
    feature_end = feature_start + timedelta(hours=2)
    prediction_start = feature_end + timedelta(hours=1)
    label_end = prediction_start + timedelta(hours=1)

    kwargs: dict[str, object] = {
        "x": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
        ),
        "y": torch.tensor([[[13.0], [14.0]], [[15.0], [16.0]]]),
        "observed_covariates": torch.ones((batch_size, history, 1)),
        "known_future_covariates": torch.ones((batch_size, horizon, 1)),
        "x_mask": torch.ones((batch_size, history, 2), dtype=torch.bool),
        "y_mask": torch.ones((batch_size, horizon, 1), dtype=torch.bool),
        "observed_covariates_mask": torch.ones((batch_size, history, 1), dtype=torch.bool),
        "known_future_covariates_mask": torch.ones((batch_size, horizon, 1), dtype=torch.bool),
        "regime": torch.tensor([0, 1], dtype=torch.int64),
        "window_id": ("window-a", "window-b"),
        "feature_names": ("signal_a", "signal_b"),
        "target_names": ("target",),
        "observed_covariate_names": ("observed",),
        "known_future_covariate_names": ("future",),
        "feature_start": (feature_start, feature_start),
        "feature_end": (feature_end, feature_end),
        "prediction_start": (prediction_start, prediction_start),
        "label_end": (label_end, label_end),
        "forecast_time": (
            (prediction_start, label_end),
            (prediction_start, label_end),
        ),
        "metadata": {"source": "fixture", "nested": {"values": [1, None, True]}},
    }
    return {**kwargs, **overrides}
