from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from tarca.contracts.interventions import (
    InterventionSite,
    InterventionSpec,
    basis_orthonormality_tolerance,
    validate_spec_against_site,
)
from tarca.contracts.types import InterventionKind


def _valid_site_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "site_name": "encoder.hidden",
        "layer": 2,
        "tensor_rank": 4,
        "batch_axis": 0,
        "variable_axis": 1,
        "patch_axis": 2,
        "feature_axis": 3,
        "shape_template": (2, 3, 4, 5),
    }
    return {**kwargs, **overrides}


def _valid_spec_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "site_name": "encoder.hidden",
        "layer": 2,
        "variable_index": 1,
        "patch_index": 2,
        "lag": 0,
        "subspace_basis": None,
        "intervention_kind": InterventionKind.FULL_SWAP,
    }
    return {**kwargs, **overrides}


def _basis_tensor() -> torch.Tensor:
    basis = torch.eye(5, dtype=torch.float64)[:, :2]
    basis.requires_grad_(True)
    assert not basis.is_contiguous()
    return basis


def _tensor_snapshot(tensor: torch.Tensor) -> tuple[tuple[object, ...], torch.Tensor]:
    properties = (
        id(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )
    return properties, tensor.detach().clone()


def _assert_tensor_unchanged(
    tensor: torch.Tensor,
    snapshot: tuple[tuple[object, ...], torch.Tensor],
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


def test_intervention_site_has_exact_frozen_field_surface() -> None:
    assert [field.name for field in fields(InterventionSite)] == [
        "site_name",
        "layer",
        "tensor_rank",
        "batch_axis",
        "variable_axis",
        "patch_axis",
        "feature_axis",
        "shape_template",
    ]
    site = InterventionSite(**_valid_site_kwargs())
    assert not hasattr(site, "__dict__")
    with pytest.raises(FrozenInstanceError):
        site.site_name = "replacement"


@pytest.mark.parametrize("site_name", ["", "   ", 1, None])
def test_intervention_site_rejects_invalid_site_name(site_name: object) -> None:
    with pytest.raises(ValueError, match=r"site_name"):
        InterventionSite(**_valid_site_kwargs(site_name=site_name))


@pytest.mark.parametrize("tensor_rank", [True, False, 0, -1, 1.5, None])
def test_intervention_site_requires_a_positive_integer_rank(tensor_rank: object) -> None:
    with pytest.raises(ValueError, match=r"tensor_rank"):
        InterventionSite(**_valid_site_kwargs(tensor_rank=tensor_rank))


@pytest.mark.parametrize(
    "shape_template",
    [
        [2, 3, 4, 5],
        (2, 3, 4),
        (2, 3, 4, 5, 6),
        (2, 0, 4, 5),
        (2, -1, 4, 5),
        (2, True, 4, 5),
        (2, 3.5, 4, 5),
    ],
)
def test_intervention_site_rejects_invalid_shape_template(shape_template: object) -> None:
    with pytest.raises(ValueError, match=r"shape_template"):
        InterventionSite(**_valid_site_kwargs(shape_template=shape_template))


def test_intervention_site_accepts_unknown_positive_shape_dimensions() -> None:
    site = InterventionSite(**_valid_site_kwargs(shape_template=(None, 3, None, 5)))
    assert site.shape_template == (None, 3, None, 5)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("batch_axis", True),
        ("batch_axis", -1),
        ("batch_axis", 4),
        ("variable_axis", False),
        ("variable_axis", -1),
        ("variable_axis", 4),
        ("patch_axis", True),
        ("patch_axis", -1),
        ("patch_axis", 4),
        ("feature_axis", False),
        ("feature_axis", -1),
        ("feature_axis", 4),
    ],
)
def test_intervention_site_rejects_invalid_axes(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=field_name):
        InterventionSite(**_valid_site_kwargs(**{field_name: value}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_axis": 0},
        {"variable_axis": 0},
        {"patch_axis": 0},
        {"patch_axis": 1},
        {"feature_axis": 1},
        {"feature_axis": 2},
    ],
)
def test_intervention_site_requires_pairwise_distinct_axes(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"axis"):
        InterventionSite(**_valid_site_kwargs(**overrides))


def test_intervention_site_allows_absent_optional_axes() -> None:
    site = InterventionSite(
        **_valid_site_kwargs(
            tensor_rank=2,
            variable_axis=None,
            patch_axis=None,
            feature_axis=1,
            shape_template=(None, 5),
        )
    )
    assert site.variable_axis is None
    assert site.patch_axis is None


@pytest.mark.parametrize("layer", [True, False, -1, 1.5, "2"])
def test_intervention_site_rejects_invalid_optional_layer(layer: object) -> None:
    with pytest.raises(ValueError, match=r"layer"):
        InterventionSite(**_valid_site_kwargs(layer=layer))


def test_intervention_site_allows_absent_layer() -> None:
    assert InterventionSite(**_valid_site_kwargs(layer=None)).layer is None


def test_intervention_spec_has_exact_frozen_field_surface_and_allows_every_lag_sign() -> None:
    assert [field.name for field in fields(InterventionSpec)] == [
        "site_name",
        "layer",
        "variable_index",
        "patch_index",
        "lag",
        "subspace_basis",
        "intervention_kind",
    ]
    for lag in (-7, 0, 9):
        spec = InterventionSpec(**_valid_spec_kwargs(lag=lag))
        assert spec.lag == lag
    assert not hasattr(spec, "__dict__")
    with pytest.raises(FrozenInstanceError):
        spec.lag = 1


@pytest.mark.parametrize("site_name", ["", "   ", 1, None])
def test_intervention_spec_rejects_invalid_site_name(site_name: object) -> None:
    with pytest.raises(ValueError, match=r"site_name"):
        InterventionSpec(**_valid_spec_kwargs(site_name=site_name))


@pytest.mark.parametrize("field_name", ["layer", "variable_index", "patch_index"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, "1"])
def test_intervention_spec_rejects_invalid_optional_indices(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        InterventionSpec(**_valid_spec_kwargs(**{field_name: value}))


@pytest.mark.parametrize("field_name", ["layer", "variable_index", "patch_index"])
def test_intervention_spec_allows_absent_optional_indices(field_name: str) -> None:
    spec = InterventionSpec(**_valid_spec_kwargs(**{field_name: None}))
    assert getattr(spec, field_name) is None


@pytest.mark.parametrize("lag", [True, False, 1.0, "1", None])
def test_intervention_spec_rejects_non_integer_lag(lag: object) -> None:
    with pytest.raises(ValueError, match=r"lag"):
        InterventionSpec(**_valid_spec_kwargs(lag=lag))


@pytest.mark.parametrize("kind", ["full_swap", "subspace_swap", 1, None])
def test_intervention_spec_does_not_coerce_intervention_kind(kind: object) -> None:
    with pytest.raises(ValueError, match=r"intervention_kind"):
        InterventionSpec(**_valid_spec_kwargs(intervention_kind=kind))


def test_full_swap_rejects_a_subspace_basis() -> None:
    with pytest.raises(ValueError, match=r"subspace_basis"):
        InterventionSpec(**_valid_spec_kwargs(subspace_basis=_basis_tensor()))


def test_subspace_swap_requires_a_basis() -> None:
    with pytest.raises(ValueError, match=r"subspace_basis"):
        InterventionSpec(
            **_valid_spec_kwargs(
                intervention_kind=InterventionKind.SUBSPACE_SWAP,
                subspace_basis=None,
            )
        )


@pytest.mark.parametrize(
    "basis",
    [
        "not-a-tensor",
        torch.ones(3, dtype=torch.float64),
        torch.ones((2, 2, 1), dtype=torch.float64),
        torch.ones((3, 2), dtype=torch.int64),
        torch.full((3, 2), math.nan, dtype=torch.float64),
        torch.full((3, 2), math.inf, dtype=torch.float64),
        torch.ones((3, 0), dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
        torch.ones((3, 2), dtype=torch.float64, device="meta"),
    ],
)
def test_subspace_swap_rejects_invalid_basis_structure(basis: object) -> None:
    with pytest.raises(ValueError, match=r"subspace_basis"):
        InterventionSpec(
            **_valid_spec_kwargs(
                intervention_kind=InterventionKind.SUBSPACE_SWAP,
                subspace_basis=basis,
            )
        )


def test_subspace_swap_preserves_basis_identity_layout_and_autograd_state() -> None:
    basis = _basis_tensor()
    snapshot = _tensor_snapshot(basis)

    spec = InterventionSpec(
        **_valid_spec_kwargs(
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
            subspace_basis=basis,
        )
    )

    assert spec.subspace_basis is basis
    _assert_tensor_unchanged(basis, snapshot)


def test_subspace_swap_preserves_basis_on_orthonormality_rejection() -> None:
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    ).transpose(0, 1)
    snapshot = _tensor_snapshot(basis)

    with pytest.raises(ValueError, match=r"subspace_basis.*orthonormal"):
        InterventionSpec(
            **_valid_spec_kwargs(
                intervention_kind=InterventionKind.SUBSPACE_SWAP,
                subspace_basis=basis,
            )
        )

    _assert_tensor_unchanged(basis, snapshot)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_basis_orthonormality_tolerance_is_dtype_aware(dtype: torch.dtype) -> None:
    assert basis_orthonormality_tolerance(dtype) == max(1e-7, 8 * torch.finfo(dtype).eps)


def test_subspace_swap_uses_engineering_tolerance_for_both_allclose_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tolerance = basis_orthonormality_tolerance(torch.float64)
    observed_tolerances: list[tuple[float, float]] = []
    real_allclose = torch.allclose

    def recording_allclose(
        input_tensor: torch.Tensor,
        other_tensor: torch.Tensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        observed_tolerances.append((atol, rtol))
        return real_allclose(
            input_tensor,
            other_tensor,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    monkeypatch.setattr(torch, "allclose", recording_allclose)
    within_tolerance = torch.tensor(
        [[math.sqrt(1.0 + tolerance)]],
        dtype=torch.float64,
    )
    outside_tolerance = torch.tensor(
        [[math.sqrt(1.0 + 4.0 * tolerance)]],
        dtype=torch.float64,
    )

    accepted = InterventionSpec(
        **_valid_spec_kwargs(
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
            subspace_basis=within_tolerance,
        )
    )
    assert accepted.subspace_basis is within_tolerance
    assert observed_tolerances == [(tolerance, tolerance)]
    with pytest.raises(ValueError, match=r"subspace_basis.*orthonormal"):
        InterventionSpec(
            **_valid_spec_kwargs(
                intervention_kind=InterventionKind.SUBSPACE_SWAP,
                subspace_basis=outside_tolerance,
            )
        )
    assert observed_tolerances == [
        (tolerance, tolerance),
        (tolerance, tolerance),
    ]


def test_validate_spec_against_site_accepts_fully_aligned_contracts() -> None:
    site = InterventionSite(**_valid_site_kwargs())
    basis = _basis_tensor()
    spec = InterventionSpec(
        **_valid_spec_kwargs(
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
            subspace_basis=basis,
        )
    )
    assert validate_spec_against_site(spec, site) is None


@pytest.mark.parametrize(
    ("spec_overrides", "error_field"),
    [
        ({"site_name": "decoder.hidden"}, "site_name"),
        ({"layer": 3}, "layer"),
        ({"layer": None}, "layer"),
    ],
)
def test_validate_spec_against_site_rejects_redundant_identity_mismatches(
    spec_overrides: dict[str, object],
    error_field: str,
) -> None:
    site = InterventionSite(**_valid_site_kwargs())
    spec = InterventionSpec(**_valid_spec_kwargs(**spec_overrides))
    with pytest.raises(ValueError, match=error_field):
        validate_spec_against_site(spec, site)


@pytest.mark.parametrize(
    ("axis_field", "index_field"),
    [
        ("variable_axis", "variable_index"),
        ("patch_axis", "patch_index"),
    ],
)
def test_validate_spec_against_site_rejects_index_without_corresponding_axis(
    axis_field: str,
    index_field: str,
) -> None:
    site = InterventionSite(**_valid_site_kwargs(**{axis_field: None}))
    spec = InterventionSpec(**_valid_spec_kwargs(**{index_field: 0}))
    with pytest.raises(ValueError, match=index_field):
        validate_spec_against_site(spec, site)


@pytest.mark.parametrize(
    ("index_field", "out_of_range"),
    [
        ("variable_index", 3),
        ("patch_index", 4),
    ],
)
def test_validate_spec_against_site_rejects_known_dimension_index_overflow(
    index_field: str,
    out_of_range: int,
) -> None:
    site = InterventionSite(**_valid_site_kwargs())
    spec = InterventionSpec(**_valid_spec_kwargs(**{index_field: out_of_range}))
    with pytest.raises(ValueError, match=index_field):
        validate_spec_against_site(spec, site)


@pytest.mark.parametrize(
    ("axis", "index_field"),
    [
        (1, "variable_index"),
        (2, "patch_index"),
    ],
)
def test_validate_spec_against_site_allows_index_when_shape_dimension_is_unknown(
    axis: int,
    index_field: str,
) -> None:
    shape_template = [2, 3, 4, 5]
    shape_template[axis] = None
    site = InterventionSite(**_valid_site_kwargs(shape_template=tuple(shape_template)))
    spec = InterventionSpec(**_valid_spec_kwargs(**{index_field: 10_000}))
    assert validate_spec_against_site(spec, site) is None


def test_validate_spec_against_site_rejects_basis_feature_dimension_mismatch() -> None:
    site = InterventionSite(**_valid_site_kwargs())
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    spec = InterventionSpec(
        **_valid_spec_kwargs(
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
            subspace_basis=basis,
        )
    )
    with pytest.raises(ValueError, match=r"subspace_basis.*feature"):
        validate_spec_against_site(spec, site)


def test_validate_spec_against_site_allows_basis_when_feature_dimension_is_unknown() -> None:
    site = InterventionSite(**_valid_site_kwargs(shape_template=(2, 3, 4, None)))
    basis = torch.eye(20, dtype=torch.float64)[:, :2]
    spec = InterventionSpec(
        **_valid_spec_kwargs(
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
            subspace_basis=basis,
        )
    )
    assert validate_spec_against_site(spec, site) is None
