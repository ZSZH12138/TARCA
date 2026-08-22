from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tarca.stage1b.config import (
    SourceLockError,
    load_qualification_config,
    load_world_suite,
    verify_source_lock,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _source() -> dict[str, object]:
    return {
        "source_id": "interfere",
        "repository_url": "https://github.com/djpasseyjr/interfere.git",
        "commit": "a" * 40,
        "package_version": "1.0.2",
        "license_id": "MIT",
        "license_path": "LICENSE",
        "license_sha256": "b" * 64,
    }


def _primary(world_id: str, family_id: str) -> dict[str, object]:
    return {
        "world_id": world_id,
        "family_id": family_id,
        "role": "PRIMARY_MECHANISTIC",
        "source_id": "interfere",
        "adapter": "INTERFERE_CML",
        "dimension": 8,
        "concepts": ["persistence", "propagation"],
        "downstream_mappings": ["nonfinancial_network", "financial_spillover"],
        "truth_capabilities": {
            "shared_future_noise": True,
            "graph": True,
            "causal_lag": True,
            "regime": True,
            "source_pairs": True,
            "negative_controls": True,
        },
        "graph": {"kind": "RING", "directed": True},
        "generator": {"alpha": 1.45, "sigma": 0.0},
        "regimes": [
            {"regime_id": "seen_low", "split_role": "SEEN", "parameters": {"eps": 0.2}},
            {
                "regime_id": "unseen_high",
                "split_role": "UNSEEN",
                "parameters": {"eps": 0.5},
            },
        ],
    }


def _suite() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "suite_id": "stage1b-worlds-v1",
        "sources": [_source()],
        "worlds": [
            _primary("network_cml_v1", "network"),
            _primary("ecology_lv_sde_v1", "ecology"),
        ],
    }


def _qualification() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "qualification_id": "stage1b-qualification-v1",
        "partitions": ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "QUAL_UNSEEN"],
        "qualification_seeds": [104729, 130363, 155921],
        "reserved_formal_seeds": [1729, 2718, 3141, 5772, 8111],
        "history_length": 32,
        "horizon": 8,
        "horizon_groups": [[1, 2], [3, 5], [6, 8]],
        "trajectory_length": 256,
        "warmup_steps": 64,
        "trajectories_per_partition": {
            "QUAL_TRAIN": 24,
            "QUAL_TUNE": 8,
            "QUAL_SEEN": 12,
            "QUAL_UNSEEN": 12,
        },
        "models": {
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "dropout": 0.1,
            "batch_size": 64,
            "max_epochs": 30,
            "patience": 5,
            "learning_rate": 0.001,
        },
        "var_search": {"lag_orders": [1, 2, 4, 8], "ridge": [1e-6, 0.001, 0.1]},
        "gate": {
            "primary_metric": "CRPS",
            "bootstrap_replicates": 2000,
            "confidence_level": 0.95,
            "guardrail_relative_tolerance": 0.02,
            "minimum_primary_families": 2,
        },
    }


def test_world_suite_rejects_primary_without_shared_future_noise(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][0]["truth_capabilities"]["shared_future_noise"] = False  # type: ignore[index]
    config_path = _write_yaml(tmp_path / "worlds.yaml", payload)

    with pytest.raises(ValidationError, match="shared future noise"):
        load_world_suite(config_path)


def test_world_suite_requires_two_independent_primary_families(tmp_path: Path) -> None:
    payload = _suite()
    payload["worlds"][1]["family_id"] = "network"  # type: ignore[index]
    config_path = _write_yaml(tmp_path / "worlds.yaml", payload)

    with pytest.raises(ValidationError, match="two independent primary families"):
        load_world_suite(config_path)


def test_qualification_config_rejects_formal_test_partition(tmp_path: Path) -> None:
    payload = _qualification()
    payload["partitions"] = ["QUAL_TRAIN", "QUAL_TUNE", "QUAL_SEEN", "TEST"]
    config_path = _write_yaml(tmp_path / "qualification.yaml", payload)

    with pytest.raises(ValidationError, match="qualification-only partitions"):
        load_qualification_config(config_path)


def test_qualification_seeds_must_not_overlap_formal_seeds(tmp_path: Path) -> None:
    payload = _qualification()
    payload["qualification_seeds"] = [104729, 130363, 1729]
    config_path = _write_yaml(tmp_path / "qualification.yaml", payload)

    with pytest.raises(ValidationError, match="must not overlap"):
        load_qualification_config(config_path)


def test_source_lock_accepts_exact_commit_and_license(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    license_bytes = b"MIT test license\n"
    (source_root / "LICENSE").write_bytes(license_bytes)
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

    evidence = verify_source_lock(source_root, commit, _sha256(license_bytes), "LICENSE")

    assert evidence.commit == commit
    assert evidence.license_sha256 == _sha256(license_bytes)


def test_source_lock_rejects_wrong_commit(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)

    with pytest.raises(SourceLockError, match="commit"):
        verify_source_lock(source_root, "0" * 40, _sha256(b"MIT\n"), "LICENSE")


def test_repository_v1_configs_load_with_expected_frozen_contracts() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")
    qualification = load_qualification_config(
        REPOSITORY_ROOT / "configs/stage1b/qualification_v1.yaml"
    )

    network = suite.world("network_cml_v1")
    ecology = suite.world("ecology_lv_sde_v1")
    assert network.generator_map()["alpha"] == pytest.approx(1.45)
    assert ecology.regimes[0].parameter_map()["growth_scale"] == pytest.approx(0.85)
    assert qualification.partitions == (
        "QUAL_TRAIN",
        "QUAL_TUNE",
        "QUAL_SEEN",
        "QUAL_UNSEEN",
    )
    assert set(qualification.qualification_seeds).isdisjoint(
        qualification.reserved_formal_seeds
    )


def test_world_lookup_rejects_unknown_world() -> None:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v1.yaml")

    with pytest.raises(KeyError, match="unknown_world"):
        suite.world("unknown_world")


def test_source_lock_rejects_license_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "LICENSE").write_text("MIT original\n", encoding="utf-8")
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
    (source_root / "LICENSE").write_text("MIT changed\n", encoding="utf-8")

    with pytest.raises(SourceLockError, match="license hash"):
        verify_source_lock(source_root, commit, _sha256(b"MIT original\n"), "LICENSE")
