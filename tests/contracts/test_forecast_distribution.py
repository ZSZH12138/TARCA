from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from fractions import Fraction

import pytest
import torch

from tarca.contracts.forecast import ForecastDistribution


def _rank_three_tensor(offset: float = 0.0) -> torch.Tensor:
    tensor = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3).add(offset).transpose(1, 2)
    tensor.requires_grad_(True)
    return tensor


def _logits_tensor() -> torch.Tensor:
    tensor = torch.arange(48, dtype=torch.float64).reshape(2, 3, 4, 2).transpose(2, 3)
    tensor.requires_grad_(True)
    return tensor


def _samples_tensor() -> torch.Tensor:
    tensor = torch.arange(36, dtype=torch.float64).reshape(3, 2, 2, 3).transpose(2, 3)
    tensor.requires_grad_(True)
    return tensor


def _sparse_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.sparse_coo_tensor(
        torch.zeros((len(shape), 1), dtype=torch.int64),
        torch.ones(1, dtype=torch.float64),
        shape,
        check_invariants=True,
    )
    tensor.requires_grad_(True)
    return tensor


def _valid_forecast_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "mean": _rank_three_tensor(),
        "scale": _rank_three_tensor(1.0),
        "quantiles": {
            0.9: _rank_three_tensor(2.0),
            0.1: _rank_three_tensor(-2.0),
            0.5: _rank_three_tensor(),
        },
        "logits": _logits_tensor(),
        "samples": _samples_tensor(),
        "window_id": ("window-a", "window-b"),
        "target_names": ("target-a", "target-b"),
    }
    return {**kwargs, **overrides}


def _tensor_fields(kwargs: Mapping[str, object]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for field_name in ("mean", "scale", "logits", "samples"):
        value = kwargs[field_name]
        if isinstance(value, torch.Tensor):
            tensors[field_name] = value
    quantiles = kwargs["quantiles"]
    if isinstance(quantiles, Mapping):
        for level, value in quantiles.items():
            if isinstance(value, torch.Tensor):
                tensors[f"quantiles[{level!r}]"] = value
    return tensors


def _snapshot(tensor: torch.Tensor) -> tuple[tuple[object, ...], torch.Tensor]:
    properties = (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )
    return properties, tensor.detach().clone()


def _assert_unchanged(
    tensor: torch.Tensor, snapshot: tuple[tuple[object, ...], torch.Tensor]
) -> None:
    properties, values = snapshot
    assert (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    ) == properties
    assert torch.equal(tensor.detach(), values)


def _sparse_snapshot(
    tensor: torch.Tensor,
) -> tuple[tuple[object, ...], torch.Tensor, torch.Tensor]:
    properties = (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.requires_grad,
        tensor.layout,
    )
    return properties, tensor._indices().clone(), tensor._values().detach().clone()


def _assert_sparse_unchanged(
    tensor: torch.Tensor,
    snapshot: tuple[tuple[object, ...], torch.Tensor, torch.Tensor],
) -> None:
    properties, indices, values = snapshot
    assert (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.requires_grad,
        tensor.layout,
    ) == properties
    assert torch.equal(tensor._indices(), indices)
    assert torch.equal(tensor._values().detach(), values)


def test_forecast_distribution_has_exact_frozen_field_surface() -> None:
    assert [field.name for field in fields(ForecastDistribution)] == [
        "mean",
        "scale",
        "quantiles",
        "logits",
        "samples",
        "window_id",
        "target_names",
    ]
    distribution = ForecastDistribution(**_valid_forecast_kwargs())
    with pytest.raises(FrozenInstanceError):
        distribution.target_names = ("replacement",)


def test_forecast_distribution_preserves_every_tensor_identity_and_layout() -> None:
    kwargs = _valid_forecast_kwargs()
    tensors = _tensor_fields(kwargs)
    snapshots = {name: _snapshot(tensor) for name, tensor in tensors.items()}

    distribution = ForecastDistribution(**kwargs)

    assert distribution.mean is tensors["mean"]
    assert distribution.scale is tensors["scale"]
    assert distribution.logits is tensors["logits"]
    assert distribution.samples is tensors["samples"]
    for level, tensor in distribution.quantiles.items():
        assert tensor is tensors[f"quantiles[{level!r}]"]
    for name, tensor in tensors.items():
        _assert_unchanged(tensor, snapshots[name])


def test_forecast_distribution_preserves_every_tensor_on_rejection() -> None:
    kwargs = _valid_forecast_kwargs()
    scale = kwargs["scale"]
    assert isinstance(scale, torch.Tensor)
    with torch.no_grad():
        scale[0, 0, 0] = 0.0
    tensors = _tensor_fields(kwargs)
    snapshots = {name: _snapshot(tensor) for name, tensor in tensors.items()}

    with pytest.raises(ValueError, match=r"scale"):
        ForecastDistribution(**kwargs)

    for name, tensor in tensors.items():
        _assert_unchanged(tensor, snapshots[name])


@pytest.mark.parametrize(
    ("mean", "reason"),
    [
        ("not-a-tensor", r"mean"),
        (torch.ones((2, 3), dtype=torch.float64), r"mean"),
        (torch.ones((2, 3, 2), dtype=torch.int64), r"mean"),
        (torch.full((2, 3, 2), math.nan, dtype=torch.float64), r"mean"),
        (torch.ones((2, 0, 2), dtype=torch.float64), r"mean"),
        (torch.ones((2, 3, 2), dtype=torch.float64, device="meta"), r"mean"),
    ],
)
def test_forecast_distribution_rejects_invalid_mean(mean: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        ForecastDistribution(**_valid_forecast_kwargs(mean=mean))


@pytest.mark.parametrize(
    ("field_name", "shape", "message"),
    [
        ("mean", (2, 3, 2), r"mean: values must support finite validation"),
        ("scale", (2, 3, 2), r"scale: values must support finite validation"),
        (
            "quantiles",
            (2, 3, 2),
            r"quantiles\[0\.5\]: values must support finite validation",
        ),
        ("logits", (2, 3, 2, 4), r"logits: values must support finite validation"),
        ("samples", (3, 2, 3, 2), r"samples: values must support finite validation"),
    ],
)
def test_forecast_distribution_fails_closed_when_sparse_values_cannot_be_validated(
    field_name: str,
    shape: tuple[int, ...],
    message: str,
) -> None:
    sparse = _sparse_tensor(shape)
    snapshot = _sparse_snapshot(sparse)
    kwargs = _valid_forecast_kwargs()
    if field_name == "quantiles":
        kwargs[field_name] = {0.5: sparse}
    else:
        kwargs[field_name] = sparse

    with pytest.raises(ValueError, match=message):
        ForecastDistribution(**kwargs)

    _assert_sparse_unchanged(sparse, snapshot)


def test_forecast_distribution_preserves_dense_finite_validation_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = RuntimeError("dense forecast backend failure")

    def fail_finite_validation(_value: object) -> None:
        raise sentinel

    monkeypatch.setattr(torch, "isfinite", fail_finite_validation)

    with pytest.raises(RuntimeError) as captured:
        ForecastDistribution(**_valid_forecast_kwargs())

    assert captured.value is sentinel


@pytest.mark.parametrize(
    "scale",
    [
        "not-a-tensor",
        torch.ones((2, 3, 1), dtype=torch.float64),
        torch.ones((2, 3, 2), dtype=torch.float32),
        torch.full((2, 3, 2), math.inf, dtype=torch.float64),
        torch.zeros((2, 3, 2), dtype=torch.float64),
        -torch.ones((2, 3, 2), dtype=torch.float64),
        torch.ones((2, 3, 2), dtype=torch.float64, device="meta"),
    ],
)
def test_forecast_distribution_rejects_invalid_scale(scale: object) -> None:
    with pytest.raises(ValueError, match=r"scale"):
        ForecastDistribution(**_valid_forecast_kwargs(scale=scale))


@pytest.mark.parametrize(
    "quantiles",
    [
        [],
        {True: torch.ones((2, 3, 2), dtype=torch.float64)},
        {"0.5": torch.ones((2, 3, 2), dtype=torch.float64)},
        {math.nan: torch.ones((2, 3, 2), dtype=torch.float64)},
        {math.inf: torch.ones((2, 3, 2), dtype=torch.float64)},
        {0.0: torch.ones((2, 3, 2), dtype=torch.float64)},
        {1.0: torch.ones((2, 3, 2), dtype=torch.float64)},
    ],
)
def test_forecast_distribution_rejects_invalid_quantile_levels(quantiles: object) -> None:
    with pytest.raises(ValueError, match=r"quantiles"):
        ForecastDistribution(**_valid_forecast_kwargs(quantiles=quantiles))


@pytest.mark.parametrize(
    "quantile",
    [
        "not-a-tensor",
        torch.ones((2, 3), dtype=torch.float64),
        torch.ones((2, 3, 1), dtype=torch.float64),
        torch.ones((2, 3, 2), dtype=torch.float32),
        torch.full((2, 3, 2), math.nan, dtype=torch.float64),
        torch.ones((2, 3, 2), dtype=torch.float64, device="meta"),
    ],
)
def test_forecast_distribution_rejects_invalid_quantile_tensors(quantile: object) -> None:
    with pytest.raises(ValueError, match=r"quantiles"):
        ForecastDistribution(**_valid_forecast_kwargs(quantiles={0.5: quantile}))


def test_forecast_distribution_allows_empty_quantiles() -> None:
    distribution = ForecastDistribution(**_valid_forecast_kwargs(quantiles={}))
    assert not distribution.quantiles


def test_forecast_distribution_normalizes_real_quantile_levels_to_float() -> None:
    quantile = _rank_three_tensor()

    distribution = ForecastDistribution(
        **_valid_forecast_kwargs(quantiles={Fraction(1, 2): quantile})
    )

    assert tuple(distribution.quantiles) == (0.5,)
    assert type(next(iter(distribution.quantiles))) is float
    assert distribution.quantiles[0.5] is quantile


def test_forecast_distribution_rejects_levels_that_collide_after_float_normalization() -> None:
    quantiles = {
        Fraction(1, 10): _rank_three_tensor(-1.0),
        0.1: _rank_three_tensor(1.0),
    }

    with pytest.raises(ValueError, match=r"quantiles"):
        ForecastDistribution(**_valid_forecast_kwargs(quantiles=quantiles))


def test_forecast_distribution_checks_quantiles_in_numeric_level_order() -> None:
    lower = _rank_three_tensor(-2.0)
    middle = _rank_three_tensor()
    upper = _rank_three_tensor(2.0)
    valid = ForecastDistribution(
        **_valid_forecast_kwargs(quantiles={0.9: upper, 0.1: lower, 0.5: middle})
    )
    assert valid.quantiles[0.1] is lower

    crossing_upper = _rank_three_tensor(2.0)
    with torch.no_grad():
        crossing_upper[0, 0, 0] = lower[0, 0, 0] - 1.0
    with pytest.raises(ValueError, match=r"quantiles"):
        ForecastDistribution(**_valid_forecast_kwargs(quantiles={0.9: crossing_upper, 0.1: lower}))


@pytest.mark.parametrize(
    "logits",
    [
        "not-a-tensor",
        torch.ones((2, 3, 2), dtype=torch.float64),
        torch.ones((2, 3, 2, 1), dtype=torch.float64),
        torch.ones((1, 3, 2, 4), dtype=torch.float64),
        torch.ones((2, 3, 2, 4), dtype=torch.float32),
        torch.full((2, 3, 2, 4), math.nan, dtype=torch.float64),
        torch.ones((2, 3, 2, 4), dtype=torch.float64, device="meta"),
    ],
)
def test_forecast_distribution_rejects_invalid_logits(logits: object) -> None:
    with pytest.raises(ValueError, match=r"logits"):
        ForecastDistribution(**_valid_forecast_kwargs(logits=logits))


@pytest.mark.parametrize(
    "samples",
    [
        "not-a-tensor",
        torch.ones((2, 3, 2), dtype=torch.float64),
        torch.ones((0, 2, 3, 2), dtype=torch.float64),
        torch.ones((3, 1, 3, 2), dtype=torch.float64),
        torch.ones((3, 2, 3, 2), dtype=torch.float32),
        torch.full((3, 2, 3, 2), math.inf, dtype=torch.float64),
        torch.ones((3, 2, 3, 2), dtype=torch.float64, device="meta"),
    ],
)
def test_forecast_distribution_rejects_invalid_samples(samples: object) -> None:
    with pytest.raises(ValueError, match=r"samples"):
        ForecastDistribution(**_valid_forecast_kwargs(samples=samples))


@pytest.mark.parametrize(
    "window_id",
    [
        ("window-a",),
        ("window-a", ""),
        ("window-a", "window-a"),
        ("window-a", 2),
        "window-a",
    ],
)
def test_forecast_distribution_rejects_invalid_window_ids(window_id: object) -> None:
    with pytest.raises(ValueError, match=r"window_id"):
        ForecastDistribution(**_valid_forecast_kwargs(window_id=window_id))


def test_forecast_distribution_allows_absent_optional_payloads() -> None:
    distribution = ForecastDistribution(
        **_valid_forecast_kwargs(
            scale=None,
            logits=None,
            samples=None,
            window_id=None,
        )
    )
    assert distribution.scale is None
    assert distribution.logits is None
    assert distribution.samples is None
    assert distribution.window_id is None


@pytest.mark.parametrize(
    "target_names",
    [
        ("target-a",),
        ("target-a", ""),
        ("target-a", "target-a"),
        ("target-a", 2),
        "target-a",
    ],
)
def test_forecast_distribution_rejects_invalid_target_names(target_names: object) -> None:
    with pytest.raises(ValueError, match=r"target_names"):
        ForecastDistribution(**_valid_forecast_kwargs(target_names=target_names))
