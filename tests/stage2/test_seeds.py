from __future__ import annotations

import importlib

import pytest


def _seeds_module():
    return importlib.import_module("tarca.stage2.seeds")


@pytest.mark.parametrize(
    ("namespace", "expected"),
    (
        ("tarca/stage2_probabilistic_forecasting_v1/dev-data/0", 669591429),
        ("tarca/stage2_probabilistic_forecasting_v1/dev-data/1", 1840764098),
        ("tarca/stage2_probabilistic_forecasting_v1/dev-data/2", 1185077341),
        ("tarca/stage2_probabilistic_forecasting_v1/model-init/0", 1797287582),
        ("tarca/stage2_probabilistic_forecasting_v1/model-init/1", 883082243),
        ("tarca/stage2_probabilistic_forecasting_v1/model-init/2", 1933050005),
        ("tarca/stage2_probabilistic_forecasting_v1/bootstrap/0", 172657089),
    ),
)
def test_namespaced_seed_derivation_is_frozen(namespace: str, expected: int) -> None:
    assert _seeds_module().derive_namespaced_seed(namespace) == expected


def test_blank_seed_namespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        _seeds_module().derive_namespaced_seed("  ")


def test_seed_isolation_rejects_cross_group_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _seeds_module().validate_seed_isolation(
            development_seeds=(11, 12, 13),
            initialization_seeds=(21, 22, 11),
            excluded_seeds=(1729, 2718),
        )


def test_seed_isolation_rejects_duplicate_group_members() -> None:
    with pytest.raises(ValueError, match="unique"):
        _seeds_module().validate_seed_isolation(
            development_seeds=(11, 11, 13),
            initialization_seeds=(21, 22, 23),
            excluded_seeds=(1729, 2718),
        )
