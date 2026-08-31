from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

STAGE2_CONFIG_PATH = Path("configs/stage2/stage2_v1.yaml")


def _config_module():
    return importlib.import_module("tarca.stage2.config")


def test_repository_stage2_config_has_frozen_identity() -> None:
    config = _config_module().load_stage2_config(STAGE2_CONFIG_PATH)

    assert config.protocol_id == "TARCA-E2E-STAGE-PROTOCOL-2.0"
    assert config.experiment_id == "stage2_probabilistic_forecasting_v1"
    assert config.upstream.stage1b_manifest_sha256 == (
        "d1b4d09260bcc41b3b94a020474ee0b5e9f9dd5f0f498bb96510228141f44b25"
    )
    assert config.upstream.e01_receipt_sha256 == (
        "16de7fc103b8f1589eec07deaebfb66fbf7ea603046020e4778bb52458c3ae14"
    )
    assert config.data.development_seeds == (669591429, 1840764098, 1185077341)
    assert config.training.initialization_seeds == (1797287582, 883082243, 1933050005)
    assert config.data.history == 64
    assert config.data.horizon == 24
    assert config.data.trajectory_length == 512
    assert config.data.train_trajectories_per_seed == 24
    assert config.data.validation_trajectories_per_seed == 8


def test_stage2_sources_and_models_are_exact() -> None:
    config = _config_module().load_stage2_config(STAGE2_CONFIG_PATH)

    assert tuple(source.source_id for source in config.sources) == (
        "dlinear",
        "itransformer",
        "patchtst",
        "scoring_rules_l96",
    )
    dlinear = config.source("dlinear")
    assert dlinear.commit == "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
    assert dlinear.assets[0].sha256 == (
        "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc"
    )
    assert tuple(model.model_id for model in config.models) == (
        "LAST_VALUE",
        "SEASONAL_NAIVE",
        "VAR",
        "DLINEAR",
        "PATCHTST",
        "ITRANSFORMER",
    )


def test_runtime_changes_do_not_change_stage2_scientific_hash(tmp_path: Path) -> None:
    payload = yaml.safe_load(STAGE2_CONFIG_PATH.read_text(encoding="utf-8"))
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    payload["runtime_profile"]["dataloader_workers_per_gpu_job"] = 2
    second_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    first = _config_module().load_stage2_config(first_path)
    second = _config_module().load_stage2_config(second_path)

    assert first.scientific_hash() == second.scientific_hash()
    assert first.runtime_hash() != second.runtime_hash()


def test_stage2_config_rejects_reserved_seed_overlap(tmp_path: Path) -> None:
    payload = yaml.safe_load(STAGE2_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["data"]["development_seeds"][0] = 1729
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="seed isolation"):
        _config_module().load_stage2_config(path)


def test_model_parameters_reject_non_json_values() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        _config_module().Stage2ModelConfig.model_validate(
            {"model_id": "LAST_VALUE", "adapter": "LAST_VALUE_GAUSSIAN", "parameters": {"bad": {1}}}
        )
