from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarca.e01.v2_config import E01V2Config, load_e01_v2_config
from tarca.e01.v2_seeds import derive_v2_seeds

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e01/e01_v2.yaml"
V1_SEEDS = (1729, 2718, 3141, 5772, 8111)


def test_v2_seed_derivation_is_deterministic_unique_and_disjoint_from_v1() -> None:
    first = derive_v2_seeds("tarca/e01-v2/formal-test", 50, V1_SEEDS)
    second = derive_v2_seeds("tarca/e01-v2/formal-test", 50, V1_SEEDS)

    assert first == second
    assert len(first) == len(set(first)) == 50
    assert all(0 < seed < 2**31 for seed in first)
    assert set(first).isdisjoint(V1_SEEDS)
    assert derive_v2_seeds("tarca/e01-v2/validation", 50, (*V1_SEEDS, *first)) != first


def test_repository_v2_config_freezes_calibrated_test_design() -> None:
    config = load_e01_v2_config(CONFIG)

    assert config.schema_version == "2.0.0"
    assert config.experiment_id == "e01_scm_truth_v2"
    assert config.formal_partition == "TEST"
    assert config.formal_seed_config.namespace == "tarca/e01-v2/formal-test"
    assert len(config.formal_seeds) == 50
    assert set(config.formal_seeds).isdisjoint(V1_SEEDS)
    assert config.sample_sizes == (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    assert tuple(world.study for world in config.worlds) == ("E01_A",)
    assert config.gates.required_seed_count == 45
    assert config.gates.confidence == pytest.approx(0.95)
    assert config.gates.mcse_ratio_max == pytest.approx(0.25)
    assert config.gates.interval_half_width_max == pytest.approx(0.01)
    assert config.gates.aggregate_multiplier_bias_max == pytest.approx(0.005)
    assert config.gates.endpoint_error_ratio_is_gate is False


def test_v2_config_rejects_gate_or_seed_drift() -> None:
    payload = load_e01_v2_config(CONFIG).model_dump(mode="json")
    payload["formal_seed_config"]["count"] = 49
    payload["gates"]["required_seed_count"] = 44

    with pytest.raises(ValidationError, match="Input should be 50"):
        E01V2Config.model_validate(payload)


def test_runtime_profile_does_not_change_v2_scientific_identity() -> None:
    config = load_e01_v2_config(CONFIG)
    payload = config.model_dump(mode="json")
    payload["runtime_profile"]["expected_ram_gib"] = 120
    changed = E01V2Config.model_validate(payload)

    assert changed.scientific_hash() == config.scientific_hash()
    assert changed.runtime_hash() != config.runtime_hash()
