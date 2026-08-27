from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tarca.contracts import InterventionKind, InterventionSpec
from tarca.stage1b.modeling import (
    OfficialITransformerPredictor,
    default_itransformer_source_context,
)
from tarca.stage1b.neural import ITransformerReference

from .model_helpers import window_batch


def _model(*, dropout: float = 0.0) -> OfficialITransformerPredictor:
    return OfficialITransformerPredictor(
        history_length=8,
        horizon=3,
        input_dimension=4,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=dropout,
    )


@pytest.mark.official_source
def test_itransformer_mean_is_upstream_mean() -> None:
    torch.manual_seed(104729)
    model = _model().eval()
    histories = torch.randn(2, 8, 4)
    with torch.inference_mode():
        upstream = model.mean_backbone(histories, None, None, None)
        adapted = model.forward_distribution(histories)
    torch.testing.assert_close(adapted.mean, upstream, atol=0.0, rtol=0.0)


def test_itransformer_alias_scale_device_and_sites() -> None:
    model = _model().freeze()
    assert ITransformerReference is OfficialITransformerPredictor
    assert model.supports_cross_variable_claim is True
    histories = torch.randn(2, 8, 4)
    distribution = model.forward_distribution(histories)
    assert distribution.mean.device == histories.device
    assert distribution.scale is not None
    assert distribution.scale.device == histories.device
    assert bool(torch.isfinite(distribution.scale).all())
    assert bool((distribution.scale > 0).all())
    sites = model.list_intervention_sites()
    assert tuple(site.site_name for site in sites) == (
        "encoder.input",
        "encoder.layer.0",
        "encoder.representation",
    )
    assert all(site.shape_template == (None, 4, 16) for site in sites)


def test_itransformer_rejects_an_altered_source_receipt() -> None:
    altered = replace(
        default_itransformer_source_context(),
        receipt_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="verified receipt"):
        OfficialITransformerPredictor(
            history_length=8,
            horizon=3,
            input_dimension=4,
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            source=altered,
        )


def test_itransformer_source_context_uses_the_configured_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_cache = tmp_path / "offline-source-cache"
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", str(configured_cache))
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "offline-capsule")

    with pytest.raises(RuntimeError, match=re.escape(str(configured_cache / "itransformer"))):
        _model()


def test_itransformer_capture_and_swaps_are_operable_without_weight_mutation() -> None:
    model = _model().freeze()
    base = window_batch(torch.randn(2, 8, 4), torch.zeros(2, 3, 4), prefix="base")
    source = window_batch(torch.randn(2, 8, 4), torch.zeros(2, 3, 4), prefix="source")
    ordinary = model.predict_distribution(base)
    before = model.model_hash

    for site in model.list_intervention_sites():
        captured = model.capture(base, (site,))[site.site_name]
        assert captured.shape == (2, 4, 16)
        identity = model.intervene(
            base,
            base,
            InterventionSpec(
                site_name=site.site_name,
                layer=site.layer,
                variable_index=None,
                patch_index=None,
                lag=0,
                subspace_basis=None,
                intervention_kind=InterventionKind.FULL_SWAP,
            ),
        )
        torch.testing.assert_close(identity.mean, ordinary.mean, atol=0.0, rtol=0.0)
        torch.testing.assert_close(identity.scale, ordinary.scale, atol=0.0, rtol=0.0)

    representation = model.list_intervention_sites()[-1]
    full = model.intervene(
        base,
        source,
        InterventionSpec(
            site_name=representation.site_name,
            layer=representation.layer,
            variable_index=1,
            patch_index=None,
            lag=0,
            subspace_basis=None,
            intervention_kind=InterventionKind.FULL_SWAP,
        ),
    )
    subspace = model.intervene(
        base,
        source,
        InterventionSpec(
            site_name=representation.site_name,
            layer=representation.layer,
            variable_index=1,
            patch_index=None,
            lag=0,
            subspace_basis=torch.eye(16),
            intervention_kind=InterventionKind.SUBSPACE_SWAP,
        ),
    )
    assert not torch.equal(full.mean, ordinary.mean)
    torch.testing.assert_close(subspace.mean, full.mean, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(subspace.scale, full.scale, atol=1e-6, rtol=0.0)
    assert model.model_hash == before
