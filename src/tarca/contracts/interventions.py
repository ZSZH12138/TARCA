"""Validated intervention site and intervention request contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .types import InterventionKind


@dataclass(frozen=True, slots=True)
class InterventionSite:
    """A model tensor location that can be captured or intervened on."""

    site_name: str
    layer: int | None
    tensor_rank: int
    batch_axis: int
    variable_axis: int | None
    patch_axis: int | None
    feature_axis: int
    shape_template: tuple[int | None, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.site_name, "site_name")
        _validate_optional_non_negative_integer(self.layer, "layer")
        rank = _validate_positive_integer(self.tensor_rank, "tensor_rank")
        _validate_shape_template(self.shape_template, rank)

        axes = (
            ("batch_axis", self.batch_axis, False),
            ("variable_axis", self.variable_axis, True),
            ("patch_axis", self.patch_axis, True),
            ("feature_axis", self.feature_axis, False),
        )
        occupied_axes: dict[int, str] = {}
        for field_name, value, optional in axes:
            axis = _validate_axis(value, field_name, rank, optional=optional)
            if axis is None:
                continue
            if axis in occupied_axes:
                raise ValueError(f"{field_name}: axis must be distinct from {occupied_axes[axis]}")
            occupied_axes[axis] = field_name


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """A validated request for a full or subspace activation swap."""

    site_name: str
    layer: int | None
    variable_index: int | None
    patch_index: int | None
    lag: int
    subspace_basis: Tensor | None
    intervention_kind: InterventionKind

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.site_name, "site_name")
        _validate_optional_non_negative_integer(self.layer, "layer")
        _validate_optional_non_negative_integer(self.variable_index, "variable_index")
        _validate_optional_non_negative_integer(self.patch_index, "patch_index")
        _validate_integer(self.lag, "lag")
        if not isinstance(self.intervention_kind, InterventionKind):
            raise ValueError("intervention_kind: expected an InterventionKind member")

        if self.intervention_kind is InterventionKind.FULL_SWAP:
            if self.subspace_basis is not None:
                raise ValueError("subspace_basis: FULL_SWAP must not carry a basis")
            return

        if self.subspace_basis is None:
            raise ValueError("subspace_basis: SUBSPACE_SWAP requires a basis")
        _validate_subspace_basis(self.subspace_basis)


def basis_orthonormality_tolerance(dtype: torch.dtype) -> float:
    """Return a dtype-aware engineering tolerance for basis validation.

    This numerical allowance handles floating-point roundoff only. It is not a
    scientific success threshold and must not be used as one.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype: expected a floating torch.dtype")
    return max(1e-7, 8 * torch.finfo(dtype).eps)


def validate_spec_against_site(spec: InterventionSpec, site: InterventionSite) -> None:
    """Validate conditions that require both a spec and its declared site."""
    if spec.site_name != site.site_name:
        raise ValueError("site_name: spec must exactly match the intervention site")
    if spec.layer != site.layer:
        raise ValueError("layer: spec must exactly match the intervention site")

    _validate_index_against_site_axis(
        index=spec.variable_index,
        index_field="variable_index",
        axis=site.variable_axis,
        axis_field="variable_axis",
        shape_template=site.shape_template,
    )
    _validate_index_against_site_axis(
        index=spec.patch_index,
        index_field="patch_index",
        axis=site.patch_axis,
        axis_field="patch_axis",
        shape_template=site.shape_template,
    )

    basis = spec.subspace_basis
    feature_dimension = site.shape_template[site.feature_axis]
    if basis is not None and feature_dimension is not None and basis.shape[0] != feature_dimension:
        raise ValueError(
            "subspace_basis: first dimension must match the site's known feature dimension"
        )


def _validate_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}: expected a non-empty string")


def _validate_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name}: expected an integer, excluding bool")
    return value


def _validate_positive_integer(value: object, field_name: str) -> int:
    integer = _validate_integer(value, field_name)
    if integer < 1:
        raise ValueError(f"{field_name}: expected an integer at least 1")
    return integer


def _validate_optional_non_negative_integer(value: object, field_name: str) -> None:
    if value is None:
        return
    integer = _validate_integer(value, field_name)
    if integer < 0:
        raise ValueError(f"{field_name}: expected a non-negative integer")


def _validate_shape_template(value: object, rank: int) -> None:
    if not isinstance(value, tuple):
        raise ValueError("shape_template: expected a tuple")
    if len(value) != rank:
        raise ValueError("shape_template: length must exactly match tensor_rank")
    for index, dimension in enumerate(value):
        if dimension is None:
            continue
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"shape_template[{index}]: expected None or a positive integer")


def _validate_axis(
    value: object,
    field_name: str,
    rank: int,
    *,
    optional: bool,
) -> int | None:
    if optional and value is None:
        return None
    axis = _validate_integer(value, field_name)
    if not 0 <= axis < rank:
        raise ValueError(f"{field_name}: expected an axis within tensor_rank")
    return axis


def _validate_subspace_basis(basis: object) -> None:
    if not isinstance(basis, Tensor):
        raise ValueError("subspace_basis: expected a torch.Tensor")
    if basis.ndim != 2:
        raise ValueError("subspace_basis: expected rank 2")
    if not torch.is_floating_point(basis):
        raise ValueError("subspace_basis: expected a floating tensor")
    if basis.device.type == "meta":
        raise ValueError("subspace_basis: values must be materialized and finite")
    try:
        finite = bool(torch.isfinite(basis).all())
    except (NotImplementedError, RuntimeError) as error:
        raise ValueError("subspace_basis: values must support finite validation") from error
    if not finite:
        raise ValueError("subspace_basis: values must be finite")

    rows, columns = basis.shape
    if columns < 1:
        raise ValueError("subspace_basis: must have at least one column")
    if columns > rows:
        raise ValueError("subspace_basis: columns must not outnumber rows")

    tolerance = basis_orthonormality_tolerance(basis.dtype)
    try:
        gram = basis.transpose(0, 1) @ basis
        identity = torch.eye(columns, dtype=basis.dtype, device=basis.device)
        orthonormal = torch.allclose(gram, identity, atol=tolerance, rtol=tolerance)
    except (NotImplementedError, RuntimeError) as error:
        raise ValueError("subspace_basis: values must support orthonormality validation") from error
    if not orthonormal:
        raise ValueError("subspace_basis: columns must be numerically orthonormal")


def _validate_index_against_site_axis(
    *,
    index: int | None,
    index_field: str,
    axis: int | None,
    axis_field: str,
    shape_template: tuple[int | None, ...],
) -> None:
    if index is None:
        return
    if axis is None:
        raise ValueError(f"{index_field}: site has no {axis_field}")
    dimension = shape_template[axis]
    if dimension is not None and index >= dimension:
        raise ValueError(f"{index_field}: out of range for the site's known dimension")
