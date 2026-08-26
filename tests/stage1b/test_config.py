from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tarca.stage1b.config import (
    SourceAuthorizationPolicy,
    SourceCodeUsage,
    load_qualification_config,
    load_world_suite,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _source(source_id: str = "published") -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": "Published dynamics",
        "repository_url": "https://github.com/example/published.git",
        "paper_url": "https://doi.org/10.0000/example",
        "commit": "a" * 40,
        "license_id": "UNDECLARED",
        "code_usage": "DIRECT_OFFICIAL_CODE_AND_DATA",
        "authorization_policy": "USER_AUTHORIZED_NO_LICENSE_BLOCK",
        "authorization_id": "stage1b-v2-user-direct-official-use-2026-08-26",
        "assets": [
            {
                "asset_id": "world_source",
                "relative_path": "world.py",
                "sha256": "c" * 64,
                "required_for": ["REPRODUCTION", "ORACLE"],
            }
        ],
        "evidence_files": [
            {
                "url": "https://raw.githubusercontent.com/example/published/commit/world.py",
                "sha256": "b" * 64,
            }
        ],
    }


def _primary(world_id: str, family_id: str) -> dict[str, object]:
    return {
        "world_id": world_id,
        "family_id": family_id,
        "role": "PRIMARY_MECHANISTIC",
        "source_id": "published",
        "adapter": "LORENZ96",
        "dimension": 20,
        "latent_dimension": 0,
        "concepts": ["forcing", "propagation", "scale"],
        "concept_pairs": [
            {
                "pair_id": "trend_primary",
                "concept": "trend",
                "parameter_family": "forcing",
                "factual_parameter_ref": "official_f10",
                "counterfactual_parameter_ref": "official_f40",
                "factual_value": 10.0,
                "counterfactual_value": 40.0,
                "shared_initial_state": True,
                "shared_future_noise": True,
                "evidence_asset_ids": ["world_source"],
            },
            {
                "pair_id": "scale_primary",
                "concept": "scale",
                "parameter_family": "measurement_noise",
                "factual_parameter_ref": "official_clean",
                "counterfactual_parameter_ref": "official_noisy",
                "factual_value": 0.0,
                "counterfactual_value": 0.1,
                "shared_initial_state": True,
                "shared_future_noise": True,
                "evidence_asset_ids": ["world_source"],
            },
        ],
        "downstream_mappings": ["weather_regime", "financial_spillover"],
        "truth_capabilities": {
            "shared_future_noise": True,
            "graph": True,
            "signed_graph": True,
            "causal_lag": True,
            "regime": True,
            "source_pairs": True,
            "negative_controls": True,
        },
        "graph": {"kind": "LORENZ96", "directed": True},
        "generator": {
            "integration_step": 0.01,
            "observation_interval": 0.1,
            "forcing": 10.0,
            "initial_perturbation": 0.01,
        },
        "regimes": [
            {
                "regime_id": "seen_clean",
                "split_role": "SEEN",
                "changed_parameter": "measurement_noise",
                "parameters": {"measurement_noise": 0.0},
            },
            {
                "regime_id": "unseen_noisy",
                "split_role": "UNSEEN",
                "changed_parameter": "measurement_noise",
                "parameters": {"measurement_noise": 0.1},
            },
        ],
    }


def _suite() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "suite_id": "stage1b-worlds-v2",
        "sources": [_source()],
        "worlds": [
            _primary("lorenz96_v2", "lorenz96_single_scale"),
            _primary("lorenz96_twoscale_v2", "lorenz96_two_scale"),
        ],
    }


def _qualification() -> dict[str, object]:
    model_common = {
        "n_layers": 3,
        "dropout": 0.1,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 20,
        "learning_rate": 0.0001,
        "d_ff": 256,
    }
    return {
        "schema_version": "2.0.0",
        "qualification_id": "stage1b-qualification-v2",
        "partitions": ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"],
        "qualification_seeds": [104729, 130363, 155921],
        "reserved_formal_seeds": [1729, 2718, 3141, 5772, 8111],
        "history_length": 64,
        "horizon": 24,
        "horizon_groups": [[1, 6], [7, 12], [13, 24]],
        "trajectory_length": 512,
        "warmup_steps": 128,
        "trajectories_per_partition": {
            "QUAL_TRAIN": 24,
            "QUAL_TUNE": 8,
            "QUAL_SEEN": 12,
            "QUAL_UNSEEN": 12,
        },
        "models": [
            {
                **model_common,
                "model_id": "patchtst_reference",
                "adapter": "PATCHTST_REFERENCE",
                "d_model": 128,
                "n_heads": 16,
                "patch_length": 16,
                "patch_stride": 8,
                "revin": True,
            },
            {
                **model_common,
                "model_id": "itransformer_reference",
                "adapter": "ITRANSFORMER_REFERENCE",
                "d_model": 512,
                "n_heads": 8,
                "revin": True,
            },
        ],
        "var_search": {"lag_orders": [1, 2, 4, 8, 16], "ridge": [1e-6, 0.001, 0.1]},
        "gate": {
            "primary_metric": "CRPS",
            "bootstrap_replicates": 2000,
            "confidence_level": 0.95,
            "guardrail_relative_tolerance": 0.05,
            "minimum_primary_families": 1,
            "minimum_comparison_units": 40,
            "minimum_win_rate": 0.65,
            "minimum_skill_score": 0.0,
            "require_seen_and_unseen_majority": True,
        },
    }


def test_world_suite_rejects_primary_without_shared_future_noise(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["truth_capabilities"]["shared_future_noise"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="shared future noise"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_world_suite_requires_two_registered_primary_families(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][1]["family_id"] = "lorenz96_single_scale"  # type: ignore[index]
    with pytest.raises(ValidationError, match="two independent primary families"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_primary_world_requires_evidenced_trend_and_scale_pairs(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["concept_pairs"] = []  # type: ignore[index]

    with pytest.raises(ValidationError, match="trend and scale"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_concept_pair_rejects_unregistered_evidence_asset(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["concept_pairs"][0]["evidence_asset_ids"] = [  # type: ignore[index]
        "invented_asset"
    ]

    with pytest.raises(ValidationError, match="evidence assets"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_concept_pair_rejects_unofficial_parameter_family(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["concept_pairs"][0]["parameter_family"] = (  # type: ignore[index]
        "invented_parameter"
    )

    with pytest.raises(ValidationError, match="parameter family"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_scale_concept_pair_rejects_negative_scale(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["concept_pairs"][1]["factual_value"] = -0.1  # type: ignore[index]

    with pytest.raises(ValidationError, match="nonnegative"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_user_authorized_direct_source_is_valid(tmp_path: Path) -> None:
    payload = _suite()
    suite = load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))

    source = suite.source("published")
    assert source.code_usage is SourceCodeUsage.DIRECT_OFFICIAL_CODE_AND_DATA
    assert source.authorization_policy is SourceAuthorizationPolicy.USER_AUTHORIZED_NO_LICENSE_BLOCK


def test_direct_source_requires_authorization_id(tmp_path: Path) -> None:
    payload = _suite()
    payload["sources"][0]["authorization_id"] = ""  # type: ignore[index]

    with pytest.raises(ValidationError, match="authorization"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_source_asset_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _suite()
    payload["sources"][0]["assets"][0]["relative_path"] = "../world.py"  # type: ignore[index]

    with pytest.raises(ValidationError, match="relative path"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_source_id_rejects_path_components(tmp_path: Path) -> None:
    payload = _suite()
    payload["sources"][0]["source_id"] = "../published"  # type: ignore[index]

    with pytest.raises(ValidationError, match="source_id"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_source_repository_url_rejects_embedded_credentials(tmp_path: Path) -> None:
    payload = _suite()
    payload["sources"][0]["repository_url"] = (  # type: ignore[index]
        "https://token@example.com/published.git"
    )

    with pytest.raises(ValidationError, match="credentials"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_v1_schema_is_not_an_active_contract(tmp_path: Path) -> None:
    payload = _suite()
    payload["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError, match=r"2\.0\.0"):
        load_world_suite(_write_yaml(tmp_path / "worlds.yaml", payload))


def test_qualification_config_rejects_formal_test_partition(tmp_path: Path) -> None:
    payload = _qualification()
    payload["partitions"] = ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "TEST"]
    with pytest.raises(ValidationError, match="qualification-only partitions"):
        load_qualification_config(_write_yaml(tmp_path / "qualification.yaml", payload))


def test_qualification_seeds_must_not_overlap_formal_seeds(tmp_path: Path) -> None:
    payload = _qualification()
    payload["qualification_seeds"] = [104729, 130363, 1729]
    with pytest.raises(ValidationError, match="must not overlap"):
        load_qualification_config(_write_yaml(tmp_path / "qualification.yaml", payload))


def test_gate_requires_at_least_40_blind_comparison_units(tmp_path: Path) -> None:
    payload = _qualification()
    payload["gate"]["minimum_comparison_units"] = 39  # type: ignore[index]
    with pytest.raises(ValidationError, match="greater than or equal to 40"):
        load_qualification_config(_write_yaml(tmp_path / "qualification.yaml", payload))


def test_repository_v2_configs_load_with_published_parameters() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    qualification = load_qualification_config(
        REPOSITORY_ROOT / "configs/stage1b/qualification_v2.yaml"
    )
    single = suite.world("lorenz96_f10_v2")
    two_scale = suite.world("lorenz96_twoscale_v2")
    ecology = suite.world("gvar_predator_prey_v2")
    assert single.generator_map()["forcing"] == pytest.approx(10.0)
    assert single.dimension == 20
    assert single.supporting_source_ids == ("neural_gc",)
    assert two_scale.generator_map()["fast_variables_per_slow"] == pytest.approx(32.0)
    assert two_scale.latent_dimension == 256
    assert ecology.generator_map()["beta"] == pytest.approx(0.2)
    assert ecology.generator_map()["sigma"] == pytest.approx(0.1)
    assert ecology.regimes[0].changed_parameter == "dynamic_noise_scale"
    assert ecology.boundary_policy == "DECLARED_ZERO_CLIP"
    assert qualification.gate.minimum_comparison_units == 40
    assert qualification.gate.minimum_win_rate == pytest.approx(0.65)
    assert {model.adapter for model in qualification.models} == {
        "PATCHTST_REFERENCE",
        "ITRANSFORMER_REFERENCE",
    }


def test_world_lookup_rejects_unknown_world() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    with pytest.raises(KeyError, match="unknown_world"):
        suite.world("unknown_world")
