from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tarca.stage1b.runner import (
    QualificationBoundaryError,
    run_hardware_probe,
    run_qualification,
    validate_qualification_receipt_boundaries,
)

from .receipt_helpers import passing_receipt


def test_runner_receipt_never_exposes_formal_partition_or_experiment() -> None:
    receipt = passing_receipt()
    validated = validate_qualification_receipt_boundaries(receipt)
    assert "TEST" not in validated["partition_names"]
    assert validated["experiment_ids"] == []


def test_runner_rejects_formal_partition_in_receipt() -> None:
    receipt = passing_receipt()
    receipt["partition_names"] = ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "TEST"]
    with pytest.raises(QualificationBoundaryError, match="qualification partitions"):
        validate_qualification_receipt_boundaries(receipt)


def test_runner_rejects_e02_identifier_in_receipt() -> None:
    receipt = passing_receipt()
    receipt["experiment_ids"] = ["E02"]
    with pytest.raises(QualificationBoundaryError, match="formal experiment"):
        validate_qualification_receipt_boundaries(receipt)


def _truth() -> dict[str, bool]:
    return {
        "shared_future_noise": True,
        "graph": True,
        "signed_graph": True,
        "causal_lag": True,
        "regime": True,
        "source_pairs": True,
        "negative_controls": True,
    }


def _world(world_id: str, family_id: str) -> dict[str, object]:
    return {
        "world_id": world_id,
        "family_id": family_id,
        "role": "PRIMARY_MECHANISTIC",
        "source_id": "published",
        "adapter": "CORRECTED_CML",
        "dimension": 4,
        "latent_dimension": 0,
        "concepts": ["local_nonlinearity", "propagation"],
        "downstream_mappings": ["network", "spillover"],
        "truth_capabilities": _truth(),
        "graph": {"kind": "RING", "directed": False},
        "generator": {"alpha": 2.0, "sigma": 0.0, "observations_per_step": 1},
        "regimes": [
            {
                "regime_id": "seen",
                "split_role": "SEEN",
                "changed_parameter": "epsilon",
                "parameters": {"epsilon": 0.3},
            },
            {
                "regime_id": "unseen",
                "split_role": "UNSEEN",
                "changed_parameter": "epsilon",
                "parameters": {"epsilon": 0.1},
            },
        ],
    }


def _worlds_payload() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "suite_id": "tiny-stage1b-worlds-v2",
        "sources": [
            {
                "source_id": "published",
                "title": "Published test equation",
                "repository_url": "https://github.com/example/published.git",
                "paper_url": "https://doi.org/10.0000/example",
                "commit": "a" * 40,
                "license_id": "REFERENCE_ONLY",
                "code_usage": "REIMPLEMENTED_EQUATIONS",
                "evidence_files": [{"url": "https://example.org/evidence.py", "sha256": "b" * 64}],
            }
        ],
        "worlds": [_world("tiny_cml_a", "family_a"), _world("tiny_cml_b", "family_b")],
    }


def _qualification_payload() -> dict[str, object]:
    common = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 16,
        "dropout": 0.0,
        "batch_size": 64,
        "max_epochs": 2,
        "patience": 1,
        "learning_rate": 0.001,
        "revin": True,
    }
    return {
        "schema_version": "2.0.0",
        "qualification_id": "tiny-qualification-v2",
        "partitions": ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"],
        "qualification_seeds": [101, 103, 107],
        "reserved_formal_seeds": [201, 203],
        "history_length": 8,
        "horizon": 4,
        "horizon_groups": [[1, 2], [3, 4]],
        "trajectory_length": 24,
        "warmup_steps": 4,
        "trajectories_per_partition": {
            "QUAL_TRAIN": 1,
            "QUAL_TUNE": 1,
            "QUAL_SEEN": 1,
            "QUAL_UNSEEN": 1,
        },
        "models": [
            {
                **common,
                "model_id": "patch",
                "adapter": "PATCHTST_REFERENCE",
                "patch_length": 4,
                "patch_stride": 2,
            },
            {**common, "model_id": "inverted", "adapter": "ITRANSFORMER_REFERENCE"},
        ],
        "var_search": {"lag_orders": [1, 2], "ridge": [0.001]},
        "gate": {
            "primary_metric": "CRPS",
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "guardrail_relative_tolerance": 0.05,
            "minimum_primary_families": 1,
            "minimum_comparison_units": 40,
            "minimum_win_rate": 0.65,
            "minimum_skill_score": 0.0,
            "require_seen_and_unseen_majority": True,
        },
    }


def test_tiny_v2_qualification_runs_without_formal_surface(tmp_path: Path) -> None:
    worlds_path = tmp_path / "worlds.yaml"
    qualification_path = tmp_path / "qualification.yaml"
    worlds_path.write_text(yaml.safe_dump(_worlds_payload(), sort_keys=False), encoding="utf-8")
    qualification_path.write_text(
        yaml.safe_dump(_qualification_payload(), sort_keys=False), encoding="utf-8"
    )
    artifact_root = tmp_path / "artifacts"
    probe = run_hardware_probe(worlds_path, qualification_path, artifact_root / "runtime")
    receipt = run_qualification(worlds_path, qualification_path, artifact_root)
    assert probe["decision"]["feasible"] is True
    assert receipt["partition_names"] == ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"]
    assert receipt["experiment_ids"] == []
    assert len(receipt["world_decisions"]) == 2
    assert len(receipt["training_receipts"]) == 12
    assert (artifact_root / "qualification_v2_summary.json").is_file()
