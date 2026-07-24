from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.stage0.diagnostics as diagnostics  # noqa: E402
from tarca.stage0.models import DoctorReport  # noqa: E402


def test_pot_smoke_produces_valid_transport_plan() -> None:
    result = diagnostics.check_pot()

    assert result.status == "PASS"
    assert tuple(result.details["shape"]) == (3, 3)
    assert result.details["finite"] is True
    assert result.details["nonnegative"] is True
    assert result.details["row_marginal_error"] <= 1e-5
    assert result.details["column_marginal_error"] <= 1e-5
    assert result.details["max_marginal_error"] <= 1e-5
    assert len(result.details["transport_plan"]) == 3


def test_pot_smoke_reports_failure_for_invalid_solver_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_plan = np.array([[np.nan, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    monkeypatch.setattr(diagnostics.ot, "sinkhorn", lambda *_args, **_kwargs: invalid_plan)

    result = diagnostics.check_pot()

    assert result.status == "FAIL"
    assert result.details["finite"] is False
    assert result.details["nonnegative"] is False
    assert result.remediation is not None
    payload = json.loads(diagnostics.report_to_json(DoctorReport(results=(result,))))
    assert payload["results"][0]["details"]["transport_plan"][0][0] is None
