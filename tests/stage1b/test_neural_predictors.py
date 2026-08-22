from __future__ import annotations

import pytest
import torch

from tarca.contracts import (
    InterventionKind,
    InterventionSpec,
    MechanisticModelAdapter,
    validate_forecast_distribution,
)
from tarca.stage1b.neural import SmallITransformer, SmallPatchTST

from .model_helpers import window_batch


@pytest.mark.parametrize("model_type", [SmallPatchTST, SmallITransformer])
def test_neural_predictor_exposes_stable_intervention_sites(model_type: type) -> None:
    model = model_type(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )

    sites = model.list_intervention_sites()

    assert isinstance(model, MechanisticModelAdapter)
    assert sites
    assert len({site.site_name for site in sites}) == len(sites)


@pytest.mark.parametrize("model_type", [SmallPatchTST, SmallITransformer])
def test_neural_forecast_has_finite_mean_and_positive_scale(model_type: type) -> None:
    model = model_type(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    ).freeze()
    batch = window_batch(torch.randn(2, 8, 4), torch.randn(2, 3, 4))

    forecast = validate_forecast_distribution(model.predict_distribution(batch))

    assert forecast.scale is not None
    assert bool((forecast.scale > 0).all())


def test_source_swap_changes_forecast_without_mutating_frozen_weights() -> None:
    model = SmallITransformer(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    ).freeze()
    base = window_batch(torch.zeros(2, 8, 4), torch.zeros(2, 3, 4), prefix="base")
    source = window_batch(torch.ones(2, 8, 4), torch.zeros(2, 3, 4), prefix="source")
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
