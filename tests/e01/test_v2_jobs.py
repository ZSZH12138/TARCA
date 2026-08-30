from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from tarca.e01.v2_carry_forward import verify_e01_b_carry_forward
from tarca.e01.v2_config import load_e01_v2_config
from tarca.e01.v2_jobs import (
    build_e01_v2_final_report,
    build_v2_seed_effects,
    e01_v2_executor_registry,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_e01_v2_config(ROOT / "configs/e01/e01_v2.yaml")


def test_v2_generator_requests_only_one_frozen_seed_and_all_conditions() -> None:
    calls: list[dict[str, object]] = []

    def fake_simulator(**arguments: object):
        calls.append(dict(arguments))
        return SimpleNamespace(values=torch.ones(2, CONFIG.horizons[-1], dtype=torch.float64))

    seed = CONFIG.formal_seeds[0]
    effects = build_v2_seed_effects(
        CONFIG,
        seed,
        device="cpu",
        batch_size=2048,
        simulator=fake_simulator,
    )

    assert tuple(effects) == CONFIG.conditions
    assert len(calls) == 5
    assert {call["seed"] for call in calls} == {seed}
    assert {call["sample_count"] for call in calls} == {8192}
    assert {call["condition"] for call in calls} == set(CONFIG.conditions)


def _passing_report(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "world_id": "analytic_delayed_control_v1",
        "sample_size_statistics": {
            "32": {"point_estimate": 1.0, "mcse": 0.04},
            "8192": {
                "point_estimate": 1.0,
                "mcse": 0.003,
                "lower": 0.994,
                "upper": 1.006,
                "half_width": 0.006,
                "covers_truth": True,
            },
        },
        "mcse_ratio": 0.075,
        "recovered_lag": 3,
        "identity_bitwise_zero": True,
        "controls": {
            control: {"pass": True, "win_fraction": 0.9, "gap_lower": 0.1}
            for control in ("WRONG_SCM", "WRONG_LAG", "RANDOM_CONCEPT")
        },
    }


def test_v2_final_report_requires_all_seed_reports_and_binds_carry_forward() -> None:
    reports = tuple(_passing_report(seed) for seed in CONFIG.formal_seeds)
    carry_forward = verify_e01_b_carry_forward(ROOT, CONFIG)

    final = build_e01_v2_final_report(CONFIG, reports, carry_forward)

    assert final["schema_version"] == "tarca-e01-final-report-v2"
    assert final["experiment_id"] == "e01_scm_truth_v2"
    assert final["gate"]["status"] == "PASS"
    assert final["e01_b_carry_forward"]["receipt_sha256"] == carry_forward.receipt_sha256
    assert final["failed_seed_policy"] == "NO_SILENT_DELETION"


def test_v2_executor_registry_is_a_three_key_allowlist() -> None:
    registry = e01_v2_executor_registry(ROOT)

    assert registry.keys == ("e01.v2.aggregate", "e01.v2.analyze", "e01.v2.generate")
