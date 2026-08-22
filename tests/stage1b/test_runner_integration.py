from __future__ import annotations

import hashlib
import subprocess
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


def _source_repo(root: Path) -> tuple[Path, str, str]:
    source_root = root / "interfere-source"
    source_root.mkdir()
    license_payload = b"MIT test license\n"
    (source_root / "LICENSE").write_bytes(license_payload)
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    subprocess.run(["git", "-C", str(source_root), "add", "LICENSE"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=TARCA Test",
            "-c",
            "user.email=tarca-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source_root, commit, hashlib.sha256(license_payload).hexdigest()


def _truth() -> dict[str, bool]:
    return {
        "shared_future_noise": True,
        "graph": True,
        "causal_lag": True,
        "regime": True,
        "source_pairs": True,
        "negative_controls": True,
    }


def _worlds_payload(commit: str, license_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "suite_id": "tiny-stage1b-worlds",
        "sources": [
            {
                "source_id": "interfere",
                "repository_url": "https://github.com/djpasseyjr/interfere.git",
                "commit": commit,
                "package_version": "1.0.2",
                "license_id": "MIT",
                "license_path": "LICENSE",
                "license_sha256": license_sha256,
            }
        ],
        "worlds": [
            {
                "world_id": "tiny_control",
                "family_id": "linear_control",
                "role": "CONTROL_LINEAR",
                "source_id": "interfere",
                "adapter": "INTERFERE_VARMA",
                "dimension": 4,
                "concepts": ["linear_memory"],
                "downstream_mappings": ["baseline_fairness"],
                "truth_capabilities": _truth(),
                "graph": {"kind": "RING", "directed": True},
                "generator": {"spectral_radius": 0.7, "innovation_scale": 0.05},
                "regimes": [
                    {
                        "regime_id": "linear_seen",
                        "split_role": "SEEN",
                        "parameters": {"coefficient_scale": 1.0},
                    },
                    {
                        "regime_id": "linear_unseen",
                        "split_role": "UNSEEN",
                        "parameters": {"coefficient_scale": 0.9},
                    },
                ],
            },
            {
                "world_id": "tiny_cml",
                "family_id": "network",
                "role": "PRIMARY_MECHANISTIC",
                "source_id": "interfere",
                "adapter": "INTERFERE_CML",
                "dimension": 4,
                "concepts": ["persistence", "propagation"],
                "downstream_mappings": ["network", "spillover"],
                "truth_capabilities": _truth(),
                "graph": {"kind": "RING", "directed": False},
                "generator": {"alpha": 1.45, "sigma": 0.0, "tsteps_btw_obs": 1},
                "regimes": [
                    {
                        "regime_id": "cml_seen",
                        "split_role": "SEEN",
                        "parameters": {"eps": 0.2},
                    },
                    {
                        "regime_id": "cml_unseen",
                        "split_role": "UNSEEN",
                        "parameters": {"eps": 0.5},
                    },
                ],
            },
            {
                "world_id": "tiny_lv",
                "family_id": "ecology",
                "role": "PRIMARY_MECHANISTIC",
                "source_id": "interfere",
                "adapter": "INTERFERE_LOTKA_VOLTERRA_SDE",
                "dimension": 4,
                "concepts": ["growth", "diffusion"],
                "downstream_mappings": ["population", "volatility"],
                "truth_capabilities": _truth(),
                "graph": {"kind": "RING", "directed": True},
                "generator": {
                    "growth_min": 0.75,
                    "growth_max": 1.15,
                    "capacity": 1.5,
                    "initial_min": 0.25,
                    "initial_max": 0.75,
                    "time_step": 0.05,
                },
                "regimes": [
                    {
                        "regime_id": "lv_seen",
                        "split_role": "SEEN",
                        "parameters": {
                            "growth_scale": 0.9,
                            "interaction_scale": 0.08,
                            "sigma": 0.01,
                        },
                    },
                    {
                        "regime_id": "lv_unseen",
                        "split_role": "UNSEEN",
                        "parameters": {
                            "growth_scale": 1.1,
                            "interaction_scale": 0.15,
                            "sigma": 0.03,
                        },
                    },
                ],
            },
        ],
    }


def _qualification_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "qualification_id": "tiny-qualification-v1",
        "partitions": ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"],
        "qualification_seeds": [101, 103, 107],
        "reserved_formal_seeds": [201, 203],
        "history_length": 4,
        "horizon": 2,
        "horizon_groups": [[1, 1], [2, 2]],
        "trajectory_length": 12,
        "warmup_steps": 4,
        "trajectories_per_partition": {
            "QUAL_TRAIN": 1,
            "QUAL_TUNE": 1,
            "QUAL_SEEN": 1,
            "QUAL_UNSEEN": 1,
        },
        "models": {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "batch_size": 16,
            "max_epochs": 2,
            "patience": 1,
            "learning_rate": 0.001,
        },
        "var_search": {"lag_orders": [1, 2], "ridge": [0.001]},
        "gate": {
            "primary_metric": "CRPS",
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "guardrail_relative_tolerance": 0.02,
            "minimum_primary_families": 2,
        },
    }


def test_tiny_real_qualification_runs_end_to_end_without_formal_surface(
    tmp_path: Path,
) -> None:
    source_root, commit, license_sha256 = _source_repo(tmp_path)
    worlds_path = tmp_path / "worlds.yaml"
    qualification_path = tmp_path / "qualification.yaml"
    worlds_path.write_text(
        yaml.safe_dump(_worlds_payload(commit, license_sha256), sort_keys=False),
        encoding="utf-8",
    )
    qualification_path.write_text(
        yaml.safe_dump(_qualification_payload(), sort_keys=False),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"

    probe = run_hardware_probe(
        worlds_path,
        qualification_path,
        source_root,
        artifact_root / "runtime",
    )
    receipt = run_qualification(
        worlds_path,
        qualification_path,
        source_root,
        artifact_root,
    )

    assert probe["decision"]["feasible"] is True
    assert receipt["partition_names"] == [
        "QUAL_TRAIN",
        "QUAL_TUNE",
        "QUAL_SEEN",
        "QUAL_UNSEEN",
    ]
    assert receipt["experiment_ids"] == []
    assert len(receipt["world_decisions"]) == 3
    assert len(receipt["training_receipts"]) == 18
    assert (artifact_root / "qualification_v1_summary.json").is_file()
