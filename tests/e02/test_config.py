from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

E02_CONFIG_PATH = Path("configs/e02/e02_v1.yaml")


def _config_module():
    return importlib.import_module("tarca.e02.config")


def test_repository_e02_config_has_frozen_gate() -> None:
    config = _config_module().load_e02_config(E02_CONFIG_PATH)

    assert config.protocol_id == "TARCA-E2E-STAGE-PROTOCOL-2.0"
    assert config.experiment_id == "e02_predictor_validity_v1"
    assert config.formal_seeds == (1729, 2718, 3141, 5772, 8111)
    assert config.trajectories_seen_per_seed == 12
    assert config.trajectories_unseen_per_seed == 12
    assert config.bootstrap.replicates == 5000
    assert config.bootstrap.seed == 172657089
    assert config.bootstrap.confidence == 0.90
    assert config.gate.minimum_crps_skill == 0.02
    assert config.gate.minimum_positive_data_seeds == 3
    assert config.gate.minimum_positive_initializations == 2


def test_e02_guardrails_are_exact() -> None:
    gate = _config_module().load_e02_config(E02_CONFIG_PATH).gate

    assert gate.unseen_skill_floor == -0.05
    assert gate.relative_nll_tolerance == 0.05
    assert gate.relative_mae_tolerance == 0.05
    assert gate.overall_coverage_error_max == 0.05
    assert gate.regime_coverage_error_max == 0.10
    assert gate.secondary_horizon_skill_floor == -0.10
    assert gate.required_completed_trajectories == 120


def test_e02_config_rejects_a_lower_primary_gate(tmp_path: Path) -> None:
    payload = yaml.safe_load(E02_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["gate"]["minimum_crps_skill"] = 0.01
    path = tmp_path / "lowered.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="gate thresholds"):
        _config_module().load_e02_config(path)


def test_runtime_changes_do_not_change_e02_scientific_hash(tmp_path: Path) -> None:
    payload = yaml.safe_load(E02_CONFIG_PATH.read_text(encoding="utf-8"))
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    payload["runtime_profile"]["monitor_port"] = 9876
    second_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    first = _config_module().load_e02_config(first_path)
    second = _config_module().load_e02_config(second_path)

    assert first.scientific_hash() == second.scientific_hash()
    assert first.runtime_hash() != second.runtime_hash()
