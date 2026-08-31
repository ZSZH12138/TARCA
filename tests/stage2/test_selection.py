from __future__ import annotations

import pytest

from tarca.stage2.selection import (
    ValidationScore,
    select_primary_initialization,
    select_strongest_linear,
)


def test_strongest_linear_uses_only_validation_crps() -> None:
    selected = select_strongest_linear({"VAR": 0.31, "DLINEAR": 0.29})

    assert selected.model_id == "DLINEAR"
    assert selected.validation_score == 0.29


def test_strongest_linear_tie_breaks_dlinear_before_var() -> None:
    assert select_strongest_linear({"VAR": 0.3, "DLINEAR": 0.3}).model_id == "DLINEAR"


def test_primary_initialization_tie_breaks_by_frozen_seed_order() -> None:
    selected = select_primary_initialization(
        "ITRANSFORMER",
        {33: 0.22, 11: 0.22, 22: 0.24},
        seed_order=(11, 22, 33),
    )

    assert selected.seed == 11


def test_selection_rejects_non_validation_artifact_reference() -> None:
    with pytest.raises(ValueError, match="VALIDATION"):
        select_strongest_linear(
            (
                ValidationScore("VAR", None, 0.31, "TEST/var.json"),
                ValidationScore("DLINEAR", None, 0.29, "VALIDATION/dlinear.json"),
            )
        )

