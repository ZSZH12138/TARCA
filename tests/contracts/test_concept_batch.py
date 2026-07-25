from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from tarca.contracts.concepts import ConceptBatch


def _values_tensor() -> torch.Tensor:
    tensor = torch.arange(6, dtype=torch.float64).reshape(3, 2).transpose(0, 1)
    tensor.requires_grad_(True)
    return tensor


def _valid_mask_tensor() -> torch.Tensor:
    return torch.tensor(
        [[True, False], [True, True], [False, True]],
        dtype=torch.bool,
    ).transpose(0, 1)


def _sparse_values_tensor() -> torch.Tensor:
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]], dtype=torch.int64),
        torch.ones(1, dtype=torch.float64),
        (2, 3),
        check_invariants=True,
    )
    tensor.requires_grad_(True)
    return tensor


def _valid_concept_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "values": _values_tensor(),
        "valid_mask": _valid_mask_tensor(),
        "names": ("level", "trend", "volatility"),
        "window_id": ("window-a", "window-b"),
        "computed_from_history_only": True,
        "definition_version": "concepts-v1",
    }
    return {**kwargs, **overrides}


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


def test_concept_batch_has_exact_frozen_field_surface() -> None:
    assert [field.name for field in fields(ConceptBatch)] == [
        "values",
        "valid_mask",
        "names",
        "window_id",
        "computed_from_history_only",
        "definition_version",
    ]
    batch = ConceptBatch(**_valid_concept_kwargs())
    with pytest.raises(FrozenInstanceError):
        batch.names = ("replacement",)


def test_concept_batch_preserves_both_tensor_identities_and_layouts() -> None:
    kwargs = _valid_concept_kwargs()
    values = kwargs["values"]
    valid_mask = kwargs["valid_mask"]
    assert isinstance(values, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    values_snapshot = _snapshot(values)
    mask_snapshot = _snapshot(valid_mask)

    batch = ConceptBatch(**kwargs)

    assert batch.values is values
    assert batch.valid_mask is valid_mask
    _assert_unchanged(batch.values, values_snapshot)
    _assert_unchanged(batch.valid_mask, mask_snapshot)


def test_concept_batch_preserves_both_tensors_on_rejection() -> None:
    kwargs = _valid_concept_kwargs()
    values = kwargs["values"]
    valid_mask = kwargs["valid_mask"]
    assert isinstance(values, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    invalid_mask = valid_mask.to(dtype=torch.int64)
    kwargs["valid_mask"] = invalid_mask
    values_snapshot = _snapshot(values)
    mask_snapshot = _snapshot(invalid_mask)

    with pytest.raises(ValueError, match=r"valid_mask"):
        ConceptBatch(**kwargs)

    _assert_unchanged(values, values_snapshot)
    _assert_unchanged(invalid_mask, mask_snapshot)


@pytest.mark.parametrize(
    "values",
    [
        "not-a-tensor",
        torch.ones((2, 3, 1), dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.int64),
        torch.full((2, 3), math.nan, dtype=torch.float64),
        torch.ones((2, 0), dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64, device="meta"),
    ],
)
def test_concept_batch_rejects_invalid_values(values: object) -> None:
    with pytest.raises(ValueError, match=r"values"):
        ConceptBatch(**_valid_concept_kwargs(values=values))


def test_concept_batch_fails_closed_when_sparse_values_cannot_be_validated() -> None:
    values = _sparse_values_tensor()
    snapshot = _sparse_snapshot(values)

    with pytest.raises(ValueError, match=r"values: values must support finite validation"):
        ConceptBatch(**_valid_concept_kwargs(values=values))

    _assert_sparse_unchanged(values, snapshot)


@pytest.mark.parametrize(
    "valid_mask",
    [
        "not-a-tensor",
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((2, 3), dtype=torch.int64),
        torch.ones((2, 3), dtype=torch.bool, device="meta"),
    ],
)
def test_concept_batch_rejects_invalid_mask(valid_mask: object) -> None:
    with pytest.raises(ValueError, match=r"valid_mask"):
        ConceptBatch(**_valid_concept_kwargs(valid_mask=valid_mask))


@pytest.mark.parametrize(
    "names",
    [
        ("level", "trend"),
        ("level", "", "volatility"),
        ("level", "level", "volatility"),
        ("level", "trend", 3),
        "level",
    ],
)
def test_concept_batch_rejects_invalid_names(names: object) -> None:
    with pytest.raises(ValueError, match=r"names"):
        ConceptBatch(**_valid_concept_kwargs(names=names))


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
def test_concept_batch_rejects_invalid_window_ids(window_id: object) -> None:
    with pytest.raises(ValueError, match=r"window_id"):
        ConceptBatch(**_valid_concept_kwargs(window_id=window_id))


@pytest.mark.parametrize("flag", [1, 0, None, "true", torch.tensor(True)])
def test_concept_batch_requires_an_exact_bool_history_flag(flag: object) -> None:
    with pytest.raises(ValueError, match=r"computed_from_history_only"):
        ConceptBatch(**_valid_concept_kwargs(computed_from_history_only=flag))


@pytest.mark.parametrize("flag", [True, False])
def test_concept_batch_accepts_either_explicit_bool_history_flag(flag: bool) -> None:
    batch = ConceptBatch(**_valid_concept_kwargs(computed_from_history_only=flag))
    assert batch.computed_from_history_only is flag


def test_concept_batch_requires_history_flag_argument() -> None:
    kwargs = _valid_concept_kwargs()
    del kwargs["computed_from_history_only"]
    with pytest.raises(TypeError, match=r"computed_from_history_only"):
        ConceptBatch(**kwargs)


@pytest.mark.parametrize("definition_version", ["", "   ", 1, None])
def test_concept_batch_rejects_invalid_definition_version(definition_version: object) -> None:
    with pytest.raises(ValueError, match=r"definition_version"):
        ConceptBatch(**_valid_concept_kwargs(definition_version=definition_version))
