from __future__ import annotations

import pytest
import torch

from tarca.contracts import (
    InterventionKind,
    InterventionSpec,
    MechanisticModelAdapter,
    validate_forecast_distribution,
)
from tarca.stage1b.neural import ITransformerReference, PatchTSTReference

from .model_helpers import window_batch


def _patchtst() -> PatchTSTReference:
    return PatchTSTReference(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        patch_length=4,
        patch_stride=2,
    )


def _itransformer() -> ITransformerReference:
    return ITransformerReference(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
    )


@pytest.mark.parametrize("factory", [_patchtst, _itransformer])
def test_reference_predictor_exposes_stable_intervention_sites(factory) -> None:  # type: ignore[no-untyped-def]
    model = factory()
    sites = model.list_intervention_sites()
    assert isinstance(model, MechanisticModelAdapter)
    assert sites
    assert len({site.site_name for site in sites}) == len(sites)


def test_patchtst_uses_channel_independent_overlapping_patch_tokens() -> None:
    model = _patchtst().freeze()
    first_x = torch.arange(64, dtype=torch.float32).reshape(2, 8, 4)
    second_x = first_x.clone()
    second_x[:, :, 1] = torch.tensor([0, 1, 4, 9, 16, 25, 36, 49])
    first = window_batch(first_x, torch.zeros(2, 3, 4), prefix="first")
    second = window_batch(second_x, torch.zeros(2, 3, 4), prefix="second")
    input_site = model.list_intervention_sites()[0]
    first_tokens = model.capture(first, (input_site,))[input_site.site_name]
    second_tokens = model.capture(second, (input_site,))[input_site.site_name]
    assert input_site.tensor_rank == 4
    assert input_site.variable_axis == 1
    assert input_site.patch_axis == 2
    assert first_tokens.shape == (2, 4, 3, 16)
    torch.testing.assert_close(first_tokens[:, 0], second_tokens[:, 0])
    assert not torch.equal(first_tokens[:, 1], second_tokens[:, 1])


@pytest.mark.parametrize("factory", [_patchtst, _itransformer])
def test_window_normalization_is_affine_equivariant(factory) -> None:  # type: ignore[no-untyped-def]
    model = factory().freeze()
    history = torch.randn(2, 8, 4)
    base = model.forward_distribution(history)
    shifted = model.forward_distribution(history * 2.5 + 7.0)
    # Official-style normalization includes an epsilon inside sqrt, so the
    # affine relationship is numerical rather than algebraically exact.
    torch.testing.assert_close(shifted.mean, base.mean * 2.5 + 7.0, atol=1e-4, rtol=1e-4)
    assert base.scale is not None and shifted.scale is not None
    torch.testing.assert_close(shifted.scale, base.scale * 2.5, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("factory", [_patchtst, _itransformer])
def test_forecast_has_finite_mean_and_input_dependent_positive_scale(factory) -> None:  # type: ignore[no-untyped-def]
    model = factory().freeze()
    first = model.forward_distribution(torch.randn(2, 8, 4))
    second = model.forward_distribution(torch.randn(2, 8, 4) ** 3)
    validate_forecast_distribution(first)
    assert first.scale is not None and second.scale is not None
    assert bool((first.scale > 0).all())
    assert not torch.equal(first.scale, second.scale)


def test_source_swap_changes_forecast_without_mutating_frozen_weights() -> None:
    model = _itransformer().freeze()
    base = window_batch(torch.zeros(2, 8, 4), torch.zeros(2, 3, 4), prefix="base")
    source = window_batch(torch.randn(2, 8, 4), torch.zeros(2, 3, 4), prefix="source")
    site = model.list_intervention_sites()[-1]
    spec = InterventionSpec(
        site_name=site.site_name,
        layer=site.layer,
        variable_index=None,
        patch_index=None,
        lag=0,
        subspace_basis=None,
        intervention_kind=InterventionKind.FULL_SWAP,
    )
    before = model.model_hash
    intervened = model.intervene(base, source, spec)
    ordinary = model.predict_distribution(base)
    assert model.model_hash == before
    assert not torch.equal(intervened.mean, ordinary.mean)
