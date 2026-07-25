from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum

import pytest
import torch
from conftest import make_valid_window_batch_kwargs

from tarca.contracts.data import WindowBatch
from tarca.contracts.types import InterventionKind, RegimeRelation, RunStatus, SplitPartition
from tarca.contracts.validation import validate_json_metadata
from tarca.contracts.version import CONTRACT_SCHEMA_VERSION


def test_contract_primitives_expose_the_versioned_str_enums() -> None:
    assert CONTRACT_SCHEMA_VERSION == "1.0.0"
    assert all(
        issubclass(enum, StrEnum)
        for enum in (
            SplitPartition,
            RegimeRelation,
            InterventionKind,
            RunStatus,
        )
    )
    assert [member.value for member in SplitPartition] == ["train", "validation", "test"]
    assert [member.value for member in RegimeRelation] == ["same", "cross", "unknown"]
    assert [member.value for member in InterventionKind] == ["full_swap", "subspace_swap"]
    assert [member.value for member in RunStatus] == ["pending", "running", "completed", "failed"]


@pytest.mark.parametrize(
    ("current_field", "legacy_field"),
    [
        ("x_observed_mask", "x_mask"),
        ("y_observed_mask", "y_mask"),
        ("input_feature_names", "feature_names"),
    ],
)
def test_window_batch_rejects_legacy_public_field_names(
    current_field: str, legacy_field: str
) -> None:
    kwargs = make_valid_window_batch_kwargs()
    kwargs[legacy_field] = kwargs.pop(current_field)

    with pytest.raises(TypeError, match=legacy_field):
        WindowBatch(**kwargs)


def test_metadata_validation_accepts_json_values_and_preserves_tensor_identity_for_rejection() -> (
    None
):
    metadata = validate_json_metadata({"number": 1.5, "nested": [None, False, {"text": "ok"}]})
    assert metadata["number"] == 1.5
    assert tuple(metadata["nested"]) == (None, False, {"text": "ok"})
    tensor = torch.tensor(1.0)
    with pytest.raises(ValueError, match=r"metadata\.payload: tensors are not JSON-compatible"):
        validate_json_metadata({"payload": tensor})
    assert tensor.item() == 1.0


def test_metadata_validation_reports_indexed_tensor_paths_without_mutating_input() -> None:
    tensor = torch.tensor([1.0], requires_grad=True)
    metadata = {"payload": ["valid", tensor]}
    expected = (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )

    with pytest.raises(
        ValueError, match=r"metadata\.payload\[1\]: tensors are not JSON-compatible"
    ):
        validate_json_metadata(metadata)

    assert metadata["payload"][1] is tensor
    assert (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    ) == expected


@pytest.mark.parametrize(
    ("metadata", "path"),
    [
        ({"bad": math.inf}, r"metadata\.bad"),
        ({1: "value"}, r"metadata: mapping keys must be strings"),
        ({"nested": {"bad": object()}}, r"metadata\.nested\.bad"),
    ],
)
def test_metadata_validation_rejects_invalid_values_with_a_field_path(
    metadata: object, path: str
) -> None:
    with pytest.raises(ValueError, match=path):
        validate_json_metadata(metadata)


def test_window_batch_accepts_valid_inputs_without_changing_tensor_identity_or_layout() -> None:
    kwargs = make_valid_window_batch_kwargs()
    x = kwargs["x"]
    assert isinstance(x, torch.Tensor)
    x.requires_grad_(True)
    expected = (id(x), x.device, x.dtype, x.shape, x.stride(), x.requires_grad)

    batch = WindowBatch(**kwargs)

    assert batch.x is x
    assert (
        id(batch.x),
        batch.x.device,
        batch.x.dtype,
        batch.x.shape,
        batch.x.stride(),
        batch.x.requires_grad,
    ) == expected
    assert batch.metadata["source"] == "fixture"
    assert tuple(batch.metadata["nested"]["values"]) == (1, None, True)
    with pytest.raises(FrozenInstanceError):
        batch.window_id = ("replacement",)


def test_window_batch_preserves_every_provided_tensor_unchanged() -> None:
    kwargs = make_valid_window_batch_kwargs()
    tensor_fields = (
        "x",
        "y",
        "observed_covariates",
        "known_future_covariates",
        "x_observed_mask",
        "y_observed_mask",
        "observed_covariates_mask",
        "known_future_covariates_mask",
        "regime",
    )
    tensors = {field_name: kwargs[field_name] for field_name in tensor_fields}
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    for field_name in ("x", "y", "observed_covariates", "known_future_covariates"):
        tensor = tensors[field_name]
        assert isinstance(tensor, torch.Tensor)
        tensor.requires_grad_(True)
    expected = {
        field_name: (
            id(tensor),
            tensor.device,
            tensor.dtype,
            tensor.shape,
            tensor.stride(),
            tensor.requires_grad,
        )
        for field_name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }

    batch = WindowBatch(**kwargs)

    for field_name, original in tensors.items():
        assert isinstance(original, torch.Tensor)
        validated = getattr(batch, field_name)
        assert validated is original
        assert (
            id(validated),
            validated.device,
            validated.dtype,
            validated.shape,
            validated.stride(),
            validated.requires_grad,
        ) == expected[field_name]


def test_window_batch_rejection_preserves_every_input_tensor_unchanged() -> None:
    kwargs = make_valid_window_batch_kwargs()
    invalid_x = kwargs["x"]
    assert isinstance(invalid_x, torch.Tensor)
    invalid_x[0, 0, 0] = float("inf")
    tensor_fields = (
        "x",
        "y",
        "observed_covariates",
        "known_future_covariates",
        "x_observed_mask",
        "y_observed_mask",
        "observed_covariates_mask",
        "known_future_covariates_mask",
        "regime",
    )
    tensors = {field_name: kwargs[field_name] for field_name in tensor_fields}
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    for field_name in ("x", "y", "observed_covariates", "known_future_covariates"):
        tensor = tensors[field_name]
        assert isinstance(tensor, torch.Tensor)
        tensor.requires_grad_(True)
    expected = {
        field_name: (
            id(tensor),
            tensor.device,
            tensor.dtype,
            tensor.shape,
            tensor.stride(),
            tensor.requires_grad,
        )
        for field_name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }

    with pytest.raises(ValueError, match=r"x: values must be finite"):
        WindowBatch(**kwargs)

    for field_name, original in tensors.items():
        assert isinstance(original, torch.Tensor)
        assert kwargs[field_name] is original
        assert (
            id(original),
            original.device,
            original.dtype,
            original.shape,
            original.stride(),
            original.requires_grad,
        ) == expected[field_name]


@pytest.mark.parametrize(
    ("field_name", "tensor"),
    [
        ("x", torch.ones((2, 3, 2), device="meta", requires_grad=True)),
        ("y", torch.ones((2, 2, 1), device="meta", requires_grad=True)),
        (
            "observed_covariates",
            torch.ones((2, 3, 1), device="meta", requires_grad=True),
        ),
        ("regime", torch.ones(2, dtype=torch.int64, device="meta")),
    ],
)
def test_window_batch_rejects_unmaterialized_tensors_without_changing_them(
    field_name: str, tensor: torch.Tensor
) -> None:
    kwargs = make_valid_window_batch_kwargs(**{field_name: tensor})
    expected = (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )

    with pytest.raises(
        ValueError,
        match=rf"{field_name}: expected a materialized non-meta tensor",
    ):
        WindowBatch(**kwargs)

    assert kwargs[field_name] is tensor
    assert (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    ) == expected


@pytest.mark.parametrize(
    ("x", "reason"),
    [
        (torch.ones((2, 3), dtype=torch.float32), r"x: expected rank 3"),
        (torch.ones((2, 3, 1), dtype=torch.int64), r"x: expected a floating tensor"),
        (torch.tensor([[[float("nan")]]]), r"x: values must be finite"),
        (torch.ones((2, 0, 1)), r"x: dimensions must all be positive"),
    ],
)
def test_window_batch_rejects_invalid_x_tensor(x: torch.Tensor, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        WindowBatch(**make_valid_window_batch_kwargs(x=x))


@pytest.mark.parametrize(
    "overrides",
    [
        {"y": torch.ones((1, 2, 1))},
        {"observed_covariates": torch.ones((2, 2, 1))},
        {
            "known_future_covariates": torch.ones((2, 1, 1)),
            "known_future_covariates_mask": torch.ones((2, 1, 1), dtype=torch.bool),
        },
    ],
)
def test_window_batch_rejects_mismatched_tensor_dimensions(overrides: dict[str, object]) -> None:
    with pytest.raises(
        ValueError,
        match=r"(y|observed_covariates|known_future_covariates|horizon):",
    ):
        WindowBatch(**make_valid_window_batch_kwargs(**overrides))


def test_window_batch_derives_horizon_from_forecast_time_when_targets_are_absent() -> None:
    kwargs = make_valid_window_batch_kwargs(
        y=None,
        y_observed_mask=None,
        target_names=(),
        known_future_covariates=None,
        known_future_covariates_mask=None,
        known_future_covariate_names=(),
    )
    batch = WindowBatch(**kwargs)
    assert len(batch.forecast_time[0]) == 2


def test_window_batch_rejects_missing_or_contradictory_horizon() -> None:
    with pytest.raises(ValueError, match=r"horizon"):
        WindowBatch(
            **make_valid_window_batch_kwargs(
                y=None,
                y_observed_mask=None,
                target_names=(),
                known_future_covariates=None,
                known_future_covariates_mask=None,
                known_future_covariate_names=(),
                forecast_time=((), ()),
            )
        )
    with pytest.raises(ValueError, match=r"horizon"):
        WindowBatch(
            **make_valid_window_batch_kwargs(
                forecast_time=(
                    (datetime(2025, 1, 1, 3, tzinfo=UTC),),
                    (datetime(2025, 1, 1, 3, tzinfo=UTC),),
                )
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_feature_names": ("signal_a", "signal_a")},
        {"target_names": ("target", "extra")},
        {"known_future_covariate_names": ("target",)},
        {"window_id": ("window-a", "window-a")},
        {"window_id": ("window-a", "")},
    ],
)
def test_window_batch_rejects_invalid_names_and_window_ids(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"(names|window_id)"):
        WindowBatch(**make_valid_window_batch_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"x_observed_mask": None},
        {"x_observed_mask": torch.ones((2, 3, 2), dtype=torch.int64)},
        {"x_observed_mask": torch.ones((2, 3, 1), dtype=torch.bool)},
        {
            "y": None,
            "target_names": (),
            "y_observed_mask": torch.ones((2, 2, 1), dtype=torch.bool),
        },
    ],
)
def test_window_batch_enforces_mask_presence_dtype_and_shape(overrides: dict[str, object]) -> None:
    if overrides == {"x_observed_mask": None}:
        batch = WindowBatch(**make_valid_window_batch_kwargs(**overrides))
        assert batch.x_observed_mask is None
    else:
        with pytest.raises(ValueError, match=r"mask"):
            WindowBatch(**make_valid_window_batch_kwargs(**overrides))


def test_window_batch_rejects_nonfinite_values_even_when_masked_out() -> None:
    x = make_valid_window_batch_kwargs()["x"]
    assert isinstance(x, torch.Tensor)
    x[0, 0, 0] = float("inf")
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 0, 0] = False
    with pytest.raises(ValueError, match=r"x: values must be finite"):
        WindowBatch(**make_valid_window_batch_kwargs(x=x, x_observed_mask=mask))


@pytest.mark.parametrize(
    "regime",
    [
        torch.ones((2, 1), dtype=torch.int64),
        torch.ones(2, dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int64),
    ],
)
def test_window_batch_validates_regime_shape_and_integer_dtype(regime: torch.Tensor) -> None:
    if regime.dtype == torch.int64 and regime.ndim == 1:
        assert WindowBatch(**make_valid_window_batch_kwargs(regime=regime)).regime is regime
    else:
        with pytest.raises(ValueError, match=r"regime"):
            WindowBatch(**make_valid_window_batch_kwargs(regime=regime))


def test_window_batch_rejects_non_utc_and_invalid_boundary_order() -> None:
    with pytest.raises(ValueError, match=r"feature_start\[0\].*UTC"):
        WindowBatch(**make_valid_window_batch_kwargs(feature_start=(datetime(2025, 1, 1),) * 2))
    non_utc = datetime(2025, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match=r"feature_start\[0\].*UTC"):
        WindowBatch(**make_valid_window_batch_kwargs(feature_start=(non_utc,) * 2))
    late_feature_end = datetime(2025, 1, 1, 4, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"boundary order"):
        WindowBatch(**make_valid_window_batch_kwargs(feature_end=(late_feature_end,) * 2))


def test_window_batch_requires_strictly_increasing_forecast_times_within_prediction_interval() -> (
    None
):
    prediction_start = datetime(2025, 1, 1, 3, tzinfo=UTC)
    duplicate_times = ((prediction_start, prediction_start),) * 2
    with pytest.raises(ValueError, match=r"forecast_time\[0\]"):
        WindowBatch(**make_valid_window_batch_kwargs(forecast_time=duplicate_times))
    too_late = datetime(2025, 1, 1, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match=r"forecast_time\[0\]"):
        WindowBatch(
            **make_valid_window_batch_kwargs(forecast_time=((prediction_start, too_late),) * 2)
        )
