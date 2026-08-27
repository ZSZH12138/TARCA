from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tarca.contracts import InterventionKind, InterventionSpec
from tarca.stage1b.modeling import (
    OfficialPatchTSTPredictor,
    default_patchtst_source_context,
)
from tarca.stage1b.neural import PatchTSTReference

from .model_helpers import window_batch


def _official_patchtst(seed: int = 104729) -> OfficialPatchTSTPredictor:
    torch.manual_seed(seed)
    return OfficialPatchTSTPredictor(
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
        source=default_patchtst_source_context(),
    )


@pytest.mark.official_source
def test_patchtst_mean_is_exact_upstream_mean() -> None:
    model = _official_patchtst().eval()
    histories = torch.randn(2, 8, 4)

    with torch.inference_mode():
        upstream = model.mean_backbone(histories.permute(0, 2, 1)).permute(0, 2, 1)
        adapted = model.forward_distribution(histories)

    torch.testing.assert_close(adapted.mean, upstream, atol=0.0, rtol=0.0)
    assert adapted.scale is not None
    assert bool((adapted.scale > 0).all())


def test_patchtst_is_official_alias_without_cross_variable_claim() -> None:
    assert PatchTSTReference is OfficialPatchTSTPredictor
    assert _official_patchtst().supports_cross_variable_claim is False


def test_patchtst_rejects_unverified_source_receipt_identity() -> None:
    source = default_patchtst_source_context()

    with pytest.raises(ValueError, match="verified receipt"):
        OfficialPatchTSTPredictor(
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
            source=replace(source, receipt_sha256="0" * 64),
        )


def test_patchtst_source_context_uses_the_configured_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_cache = tmp_path / "offline-source-cache"
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_CACHE_ROOT", str(configured_cache))
    monkeypatch.setenv("TARCA_STAGE1B_SOURCE_MODE", "offline-capsule")

    with pytest.raises(RuntimeError, match=re.escape(str(configured_cache / "patchtst"))):
        default_patchtst_source_context()


def test_patchtst_reset_covers_official_nonmodule_parameters() -> None:
    first = _official_patchtst(7)
    second = _official_patchtst(9)

    first.reset_for_training(130363)
    second.reset_for_training(130363)

    assert first.model_hash == second.model_hash


@pytest.mark.official_source
def test_patchtst_registered_identity_swap_preserves_output_and_weights() -> None:
    model = _official_patchtst().freeze()
    batch = window_batch(
        torch.randn(2, 8, 4),
        torch.zeros(2, 3, 4),
        prefix="identity",
    )
    before_hash = model.model_hash
    ordinary = model.predict_distribution(batch)

    for site in model.list_intervention_sites():
        spec = InterventionSpec(
            site_name=site.site_name,
            layer=site.layer,
            variable_index=None,
            patch_index=None,
            lag=0,
            subspace_basis=None,
            intervention_kind=InterventionKind.FULL_SWAP,
        )
        intervened = model.intervene(batch, batch, spec)

        assert model.model_hash == before_hash
        torch.testing.assert_close(intervened.mean, ordinary.mean, atol=0.0, rtol=0.0)
        assert intervened.scale is not None and ordinary.scale is not None
        torch.testing.assert_close(intervened.scale, ordinary.scale, atol=0.0, rtol=0.0)


def test_patchtst_swap_rejects_out_of_range_registered_axis() -> None:
    model = _official_patchtst().freeze()
    batch = window_batch(
        torch.randn(2, 8, 4),
        torch.zeros(2, 3, 4),
        prefix="bounds",
    )
    site = model.list_intervention_sites()[-1]
    spec = InterventionSpec(
        site_name=site.site_name,
        layer=site.layer,
        variable_index=4,
        patch_index=None,
        lag=0,
        subspace_basis=None,
        intervention_kind=InterventionKind.FULL_SWAP,
    )

    with pytest.raises(ValueError, match="outside"):
        model.intervene(batch, batch, spec)
