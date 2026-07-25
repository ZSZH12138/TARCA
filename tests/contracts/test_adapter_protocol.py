from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import get_type_hints

import pytest
import torch
from torch import Tensor

from tarca.contracts.adapters import ForecastModelAdapter
from tarca.contracts.data import WindowBatch
from tarca.contracts.forecast import ForecastDistribution
from tarca.contracts.interventions import InterventionSite, InterventionSpec
from tarca.contracts.types import InterventionKind


class _FakeAdapter:
    def __init__(
        self,
        distribution: ForecastDistribution,
        site: InterventionSite,
        captured: Tensor,
    ) -> None:
        self._distribution = distribution
        self._site = site
        self._captured = captured

    @property
    def adapter_name(self) -> str:
        return "fake-adapter"

    @property
    def model_hash(self) -> str:
        return "sha256:test-model"

    @property
    def is_frozen(self) -> bool:
        return True

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        return self._distribution

    def list_intervention_sites(self) -> tuple[InterventionSite, ...]:
        return (self._site,)

    def capture(
        self,
        batch: WindowBatch,
        sites: tuple[InterventionSite, ...],
    ) -> Mapping[str, Tensor]:
        return {site.site_name: self._captured for site in sites}

    def intervene(
        self,
        base: WindowBatch,
        source: WindowBatch,
        spec: InterventionSpec,
    ) -> ForecastDistribution:
        return self._distribution


def _site() -> InterventionSite:
    return InterventionSite(
        site_name="encoder.hidden",
        layer=1,
        tensor_rank=2,
        batch_axis=0,
        variable_axis=None,
        patch_axis=None,
        feature_axis=1,
        shape_template=(2, 3),
    )


def _spec() -> InterventionSpec:
    return InterventionSpec(
        site_name="encoder.hidden",
        layer=1,
        variable_index=None,
        patch_index=None,
        lag=-2,
        subspace_basis=None,
        intervention_kind=InterventionKind.FULL_SWAP,
    )


def _distribution() -> ForecastDistribution:
    return ForecastDistribution(
        mean=torch.ones((2, 1, 1), dtype=torch.float64),
        scale=None,
        quantiles={},
        logits=None,
        samples=None,
        window_id=("window-a", "window-b"),
        target_names=("target",),
    )


def test_forecast_model_adapter_has_exact_property_annotations() -> None:
    expected = {
        "adapter_name": str,
        "model_hash": str,
        "is_frozen": bool,
    }
    for property_name, return_type in expected.items():
        descriptor = inspect.getattr_static(ForecastModelAdapter, property_name)
        assert isinstance(descriptor, property)
        assert descriptor.fget is not None
        assert get_type_hints(descriptor.fget) == {"return": return_type}
        assert tuple(inspect.signature(descriptor.fget).parameters) == ("self",)


def test_forecast_model_adapter_has_exact_method_signatures_and_annotations() -> None:
    expected = {
        "predict_distribution": (
            ("self", "batch"),
            {"batch": WindowBatch, "return": ForecastDistribution},
        ),
        "list_intervention_sites": (
            ("self",),
            {"return": tuple[InterventionSite, ...]},
        ),
        "capture": (
            ("self", "batch", "sites"),
            {
                "batch": WindowBatch,
                "sites": tuple[InterventionSite, ...],
                "return": Mapping[str, Tensor],
            },
        ),
        "intervene": (
            ("self", "base", "source", "spec"),
            {
                "base": WindowBatch,
                "source": WindowBatch,
                "spec": InterventionSpec,
                "return": ForecastDistribution,
            },
        ),
    }
    for method_name, (parameters, expected_annotations) in expected.items():
        method = inspect.getattr_static(ForecastModelAdapter, method_name)
        assert inspect.isfunction(method)
        assert tuple(inspect.signature(method).parameters) == parameters
        assert get_type_hints(method) == expected_annotations


def test_forecast_model_adapter_is_not_runtime_checkable() -> None:
    fake = _FakeAdapter(_distribution(), _site(), torch.ones((2, 3)))
    with pytest.raises(TypeError, match=r"runtime_checkable"):
        isinstance(fake, ForecastModelAdapter)


def test_local_fake_exercises_the_protocol_surface_without_a_real_adapter() -> None:
    distribution = _distribution()
    site = _site()
    captured = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    fake = _FakeAdapter(distribution, site, captured)
    batch = object()
    assert fake.adapter_name == "fake-adapter"
    assert fake.model_hash == "sha256:test-model"
    assert fake.is_frozen is True
    assert fake.predict_distribution(batch) is distribution  # type: ignore[arg-type]
    assert fake.list_intervention_sites() == (site,)
    assert fake.capture(batch, (site,)) == {site.site_name: captured}  # type: ignore[arg-type]
    assert fake.intervene(batch, batch, _spec()) is distribution  # type: ignore[arg-type]
