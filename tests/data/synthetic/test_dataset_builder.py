from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch
import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tarca.data.synthetic.dataset_builder as dataset_builder_module  # noqa: E402
from tarca.contracts import (  # noqa: E402
    DataManifest,
    SplitPartition,
    WindowBatch,
    validate_disjoint_window_partitions,
)
from tarca.data.synthetic.dataset_builder import (  # noqa: E402
    PersistedSyntheticDataset,
    PhysicalSplit,
    SyntheticConfig,
    SyntheticDataset,
    build_synthetic_dataset,
    load_synthetic_config,
    persist_synthetic_dataset,
)
from tarca.data.synthetic.nonlinear_var import (  # noqa: E402
    RegimeDynamics,
    deterministic_transition,
)
from tarca.data.synthetic.regimes import spawn_random_streams  # noqa: E402

CONFIG_ROOT = PROJECT_ROOT / "configs" / "synthetic"
PHYSICAL_SPLIT_NAMES = (
    "train",
    "validation",
    "test_seen_regime",
    "test_unseen_regime",
)
REQUIRED_FILES = {
    "config_resolved.yaml",
    "manifest.json",
    "checksums.json",
    "truth.npz",
    "windows_train.arrow",
    "windows_validation.arrow",
    "windows_test_seen_regime.arrow",
    "windows_test_unseen_regime.arrow",
    "normalization.json",
}
REQUIRED_TRUTH = {
    "x_complete",
    "trend",
    "scale",
    "regime_sequence",
    "exogenous",
    "observation_noise",
    "trend_noise",
    "scale_noise",
    "shock_sequence",
    "missing_mask",
    "resolved_true_delay",
    "parameter_variant",
    "base_log_scale_schedule",
    "trend_delay_schedule",
    "scale_loading_schedule",
    "linear_matrices",
    "nonlinear_matrices",
    "exogenous_matrices",
    "seen_base_log_scales",
    "unseen_base_log_scales",
    "true_graph",
    "replay_initial_history",
    "replay_trend_history",
    "nonlinear_matrix_schedule",
    "exogenous_matrix_schedule",
    "nonlinear_strength_schedule",
    "nonlinear_delay_schedule",
    "raw_spectral_radius_schedule",
    "spectral_scale_factor_schedule",
    "final_spectral_radius_schedule",
    "true_graph_schedule",
    "trend_loading",
    "observation_scale_floor",
    "stability_target",
}
EXPECTED_ARROW_FIELDS = (
    ("window_id", pa.string(), False),
    ("x", pa.large_list(pa.large_list(pa.float64())), False),
    ("y", pa.large_list(pa.large_list(pa.float64())), True),
    ("observed_covariates", pa.large_list(pa.large_list(pa.float64())), True),
    ("known_future_covariates", pa.large_list(pa.large_list(pa.float64())), True),
    ("x_observed_mask", pa.large_list(pa.large_list(pa.bool_())), True),
    ("y_observed_mask", pa.large_list(pa.large_list(pa.bool_())), True),
    ("observed_covariates_mask", pa.large_list(pa.large_list(pa.bool_())), True),
    ("known_future_covariates_mask", pa.large_list(pa.large_list(pa.bool_())), True),
    ("regime", pa.int64(), True),
    ("input_feature_names", pa.large_list(pa.string()), False),
    ("target_names", pa.large_list(pa.string()), False),
    ("observed_covariate_names", pa.large_list(pa.string()), False),
    ("known_future_covariate_names", pa.large_list(pa.string()), False),
    ("feature_start", pa.timestamp("us", tz="UTC"), False),
    ("feature_end", pa.timestamp("us", tz="UTC"), False),
    ("prediction_start", pa.timestamp("us", tz="UTC"), False),
    ("label_end", pa.timestamp("us", tz="UTC"), False),
    ("forecast_time", pa.large_list(pa.timestamp("us", tz="UTC")), False),
    ("metadata_json", pa.string(), False),
)


@pytest.fixture(scope="module")
def easy_config() -> SyntheticConfig:
    return load_synthetic_config(CONFIG_ROOT / "synthetic_easy.yaml")


@pytest.fixture(scope="module")
def easy_dataset(easy_config: SyntheticConfig) -> SyntheticDataset:
    return build_synthetic_dataset(easy_config)


def _split(dataset: SyntheticDataset, name: str) -> PhysicalSplit:
    return next(split for split in dataset.physical_splits if split.name == name)


def _origin(window_id: str) -> int:
    return int(window_id.rsplit(":", maxsplit=1)[1])


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ("cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        return
    link.symlink_to(target, target_is_directory=True)


def test_dataset_builder_public_surface_is_importable() -> None:
    assert SyntheticConfig is not None
    assert PhysicalSplit is not None
    assert SyntheticDataset is not None
    assert PersistedSyntheticDataset is not None
    assert callable(load_synthetic_config)
    assert callable(build_synthetic_dataset)
    assert callable(persist_synthetic_dataset)


@pytest.mark.parametrize(
    (
        "filename",
        "expected_name",
        "dimension",
        "history",
        "horizon",
        "regimes",
        "delay",
        "seed",
        "burn_in",
        "steps",
        "mc_samples",
        "pairs",
        "snr",
        "overlap",
        "missing_rate",
        "missingness",
    ),
    [
        (
            "synthetic_easy.yaml",
            "synthetic_easy",
            4,
            48,
            12,
            2,
            2,
            20260725,
            256,
            4096,
            256,
            16,
            "high",
            "low",
            0.0,
            "none",
        ),
        (
            "synthetic_medium.yaml",
            "synthetic_medium",
            8,
            96,
            24,
            3,
            (1, 4),
            20260726,
            384,
            8192,
            128,
            8,
            "medium",
            "medium",
            0.05,
            "mcar",
        ),
        (
            "synthetic_hard.yaml",
            "synthetic_hard",
            16,
            192,
            48,
            4,
            (0, 8),
            20260727,
            512,
            12288,
            64,
            4,
            "low",
            "high",
            0.15,
            "block",
        ),
    ],
)
def test_authoritative_config_core_values_are_preserved_exactly(
    filename: str,
    expected_name: str,
    dimension: int,
    history: int,
    horizon: int,
    regimes: int,
    delay: int | tuple[int, int],
    seed: int,
    burn_in: int,
    steps: int,
    mc_samples: int,
    pairs: int,
    snr: str,
    overlap: str,
    missing_rate: float,
    missingness: str,
) -> None:
    config = load_synthetic_config(CONFIG_ROOT / filename)

    assert (
        config.name,
        config.D,
        config.L,
        config.H,
        config.regimes,
        config.true_delay,
    ) == (expected_name, dimension, history, horizon, regimes, delay)
    assert (
        config.root_seed,
        config.burn_in,
        config.total_steps,
        config.mc_samples_smoke,
        config.oracle_pairs_smoke,
    ) == (seed, burn_in, steps, mc_samples, pairs)
    assert (config.snr, config.concept_overlap) == (snr, overlap)
    assert (config.missing_rate, config.missingness_kind) == (missing_rate, missingness)
    assert config.unseen_shift_kind == "parameter_shift"
    if isinstance(delay, tuple):
        assert type(delay) is tuple


def test_config_and_nested_records_are_strict_frozen_and_extra_forbid(
    easy_config: SyntheticConfig,
    tmp_path: Path,
) -> None:
    assert easy_config.model_config["strict"] is True
    assert easy_config.model_config["frozen"] is True
    assert easy_config.model_config["extra"] == "forbid"
    assert easy_config.generation.model_config == easy_config.model_config
    assert easy_config.normalization.model_config == easy_config.model_config

    payload = easy_config.model_dump(mode="python")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SyntheticConfig.model_validate(payload | {"unexpected": 1})
    with pytest.raises(ValidationError, match="root_seed"):
        SyntheticConfig.model_validate(payload | {"root_seed": True})
    with pytest.raises(ValidationError, match="frozen_instance"):
        easy_config.root_seed = 1

    for invalid in (float("nan"), float("inf")):
        generation = payload["generation"] | {"unseen_parameter_shift": invalid}
        with pytest.raises(ValidationError, match="unseen_parameter_shift"):
            SyntheticConfig.model_validate(payload | {"generation": generation})

    invalid_yaml = tmp_path / "extra.yaml"
    invalid_yaml.write_text(
        (CONFIG_ROOT / "synthetic_easy.yaml").read_text(encoding="utf-8") + "\nunexpected: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_synthetic_config(invalid_yaml)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"true_delay": 12}, "true_delay"),
        ({"true_delay": [1, 4, 7]}, "true_delay"),
        ({"true_delay": [True, 2]}, "true_delay"),
        ({"true_delay": [4, 1]}, "true_delay"),
        ({"L": 1}, "L"),
        ({"total_steps": 100}, "physical split"),
        ({"mc_samples_smoke": 257}, "mc_samples_smoke"),
        ({"oracle_pairs_smoke": 17}, "oracle_pairs_smoke"),
        ({"missing_rate": 0.1}, "missing_rate"),
        ({"burn_in": 1}, "burn_in"),
    ],
)
def test_config_fails_fast_on_invalid_delays_resources_and_geometry(
    easy_config: SyntheticConfig,
    mutation: dict[str, object],
    message: str,
) -> None:
    payload = easy_config.model_dump(mode="python") | mutation

    with pytest.raises(ValidationError, match=message):
        SyntheticConfig.model_validate(payload)


def test_true_delay_before_validator_refuses_non_yaml_coercions(
    easy_config: SyntheticConfig,
) -> None:
    payload = easy_config.model_dump(mode="python")
    for delay in ((1.0, 2), ("1", 2), np.array([1, 2]), [1.0, 2]):
        with pytest.raises(ValidationError, match="true_delay"):
            SyntheticConfig.model_validate(payload | {"true_delay": delay})


def test_build_is_deterministic_and_changed_seed_changes_identity(
    easy_config: SyntheticConfig,
    easy_dataset: SyntheticDataset,
) -> None:
    repeated = build_synthetic_dataset(easy_config)
    changed = build_synthetic_dataset(easy_config.model_copy(update={"root_seed": 20260724}))

    assert repeated.config_hash == easy_dataset.config_hash
    assert repeated.dataset_hash == easy_dataset.dataset_hash
    assert tuple(split.split_hash for split in repeated.physical_splits) == tuple(
        split.split_hash for split in easy_dataset.physical_splits
    )
    for name in sorted(easy_dataset.truth):
        assert repeated.truth[name].tobytes() == easy_dataset.truth[name].tobytes()
    assert changed.config_hash != easy_dataset.config_hash
    assert changed.dataset_hash != easy_dataset.dataset_hash
    assert changed.truth["x_complete"].tobytes() != easy_dataset.truth["x_complete"].tobytes()


def test_truth_is_complete_finite_read_only_and_uses_all_named_streams(
    easy_dataset: SyntheticDataset,
) -> None:
    truth = easy_dataset.truth

    assert REQUIRED_TRUTH <= set(truth)
    assert truth["x_complete"].shape == (4096, 4)
    assert truth["trend"].shape == (4097,)
    assert truth["scale"].shape == (4097,)
    assert truth["regime_sequence"].shape == (4096,)
    assert truth["exogenous"].shape == (4096, 1)
    assert truth["observation_noise"].shape == (4096, 4)
    assert truth["missing_mask"].shape == (4096, 4)
    assert truth["missing_mask"].dtype == np.bool_
    assert truth["resolved_true_delay"].tolist() == [2, 2]
    assert all(not array.flags.writeable for array in truth.values())
    assert all(np.all(np.isfinite(array)) for array in truth.values())
    with pytest.raises(TypeError):
        truth["new"] = np.zeros(1)  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        truth["x_complete"][0, 0] = 0.0
    assert tuple(stream.name for stream in easy_dataset.synthetic_provenance.random_streams) == (
        "regime_transitions",
        "trend_innovations",
        "scale_innovations",
        "exogenous_variables",
        "observation_innovations",
        "sparse_shocks",
        "missingness",
        "parameter_generation",
        "counterfactual_mc_bank",
        "random_concept_negative_control",
    )


def test_truth_contains_exact_post_burn_replay_state_and_dynamics(
    easy_dataset: SyntheticDataset,
) -> None:
    truth = easy_dataset.truth
    assert truth["replay_initial_history"].shape == (2, 4)
    assert truth["replay_trend_history"].shape == (2,)
    assert truth["linear_matrix_schedule"].shape == (4096, 4, 4)
    assert truth["nonlinear_matrix_schedule"].shape == (4096, 4, 4)
    assert truth["exogenous_matrix_schedule"].shape == (4096, 4, 1)
    for name in (
        "nonlinear_strength_schedule",
        "base_log_scale_schedule",
        "nonlinear_delay_schedule",
        "trend_delay_schedule",
        "scale_loading_schedule",
        "raw_spectral_radius_schedule",
        "spectral_scale_factor_schedule",
        "final_spectral_radius_schedule",
    ):
        assert truth[name].shape == (4096,)
    assert truth["true_graph_schedule"].shape == (4096, 4, 4)
    assert truth["observation_scale_floor"].shape == ()
    assert truth["stability_target"].shape == ()
    np.testing.assert_array_equal(
        truth["replay_initial_history"][-1],
        truth["x_complete"][0],
    )

    dynamics = RegimeDynamics(
        regime_label=int(truth["regime_sequence"][0]),
        linear_matrix=truth["linear_matrix_schedule"][0],
        nonlinear_matrix=truth["nonlinear_matrix_schedule"][0],
        exogenous_matrix=truth["exogenous_matrix_schedule"][0],
        nonlinear_strength=float(truth["nonlinear_strength_schedule"][0]),
        base_log_scale=float(truth["base_log_scale_schedule"][0]),
        scale_loading=float(truth["scale_loading_schedule"][0]),
        nonlinear_delay=int(truth["nonlinear_delay_schedule"][0]),
        trend_delay=int(truth["trend_delay_schedule"][0]),
        raw_spectral_radius=float(truth["raw_spectral_radius_schedule"][0]),
        spectral_scale_factor=float(truth["spectral_scale_factor_schedule"][0]),
        final_spectral_radius=float(truth["final_spectral_radius_schedule"][0]),
        stability_target=float(truth["stability_target"]),
        true_graph=truth["true_graph_schedule"][0],
    )
    next_state = deterministic_transition(
        state_history=truth["replay_initial_history"],
        trend_history=np.concatenate((truth["replay_trend_history"], truth["trend"][:1])),
        scale_state=float(truth["scale"][0]),
        exogenous_input=truth["exogenous"][0],
        observation_innovation=truth["observation_noise"][0],
        shock=truth["shock_sequence"][0],
        dynamics=dynamics,
        trend_loading=truth["trend_loading"],
        observation_scale_floor=float(truth["observation_scale_floor"]),
    )
    np.testing.assert_array_equal(next_state, truth["x_complete"][1])


def test_concept_overlap_controls_actual_loading_support() -> None:
    expected_support = {"low": 1, "medium": 4, "high": 16}
    for filename in (
        "synthetic_easy.yaml",
        "synthetic_medium.yaml",
        "synthetic_hard.yaml",
    ):
        config = load_synthetic_config(CONFIG_ROOT / filename)
        stream = spawn_random_streams(config.root_seed)["parameter_generation"].generator
        _, _, parameters = dataset_builder_module._generate_parameters(config, stream)
        assert (
            np.count_nonzero(parameters["trend_loading"])
            == expected_support[config.concept_overlap]
        )


def test_physical_splits_are_continuous_disjoint_and_windows_never_cross(
    easy_dataset: SyntheticDataset,
) -> None:
    splits = easy_dataset.physical_splits
    expected = (
        ("train", 0, 2457, 2398),
        ("validation", 2457, 3276, 760),
        ("test_seen_regime", 3276, 3686, 351),
        ("test_unseen_regime", 3686, 4096, 351),
    )

    assert tuple(
        (item.name, item.start, item.stop, len(item.batch.window_id)) for item in splits
    ) == (expected)
    all_ids = tuple(window for item in splits for window in item.batch.window_id)
    assert len(all_ids) == len(set(all_ids))
    for item in splits:
        origins = tuple(_origin(window_id) for window_id in item.batch.window_id)
        assert origins[0] == item.start + easy_dataset.config.L - 1
        assert origins[-1] + easy_dataset.config.H < item.stop
        assert all(left < right for left, right in pairwise(origins))

    validate_disjoint_window_partitions(
        {
            SplitPartition.TRAIN: splits[0].batch.window_id,
            SplitPartition.VALIDATION: splits[1].batch.window_id,
            SplitPartition.TEST: (*splits[2].batch.window_id, *splits[3].batch.window_id),
        }
    )


def test_window_batches_are_contract_valid_and_regime_is_the_origin_regime(
    easy_dataset: SyntheticDataset,
) -> None:
    for split in easy_dataset.physical_splits:
        batch = split.batch
        assert isinstance(batch, WindowBatch)
        assert batch.x.shape[1:] == (48, 4)
        assert batch.y is not None and batch.y.shape[1:] == (12, 4)
        assert batch.observed_covariates is not None
        assert batch.observed_covariates.shape[1:] == (48, 1)
        assert batch.known_future_covariates is not None
        assert batch.known_future_covariates.shape[1:] == (12, 1)
        assert batch.regime is not None
        origins = np.array([_origin(window_id) for window_id in batch.window_id])
        expected_regime = torch.from_numpy(easy_dataset.truth["regime_sequence"][origins].copy())
        assert torch.equal(batch.regime, expected_regime)
        assert bool(torch.isfinite(batch.x).all())
        assert bool(torch.isfinite(batch.y).all())
        assert batch.x_observed_mask is not None
        assert batch.x_observed_mask.dtype == torch.bool
        assert batch.y_observed_mask is not None
        assert batch.y_observed_mask.dtype == torch.bool
        first_origin = _origin(batch.window_id[0])
        np.testing.assert_array_equal(
            batch.known_future_covariates[0, 0].numpy(),
            easy_dataset.truth["exogenous"][first_origin],
        )


def test_scaler_is_fit_on_complete_train_only_with_one_fallback(
    easy_dataset: SyntheticDataset,
) -> None:
    train_complete = easy_dataset.truth["x_complete"][:2457]
    expected_mean = np.mean(train_complete, axis=0)
    population_std = np.std(train_complete, axis=0)
    expected_scale = np.where(
        population_std > easy_dataset.normalization.epsilon,
        population_std,
        1.0,
    )

    np.testing.assert_array_equal(np.array(easy_dataset.normalization.mean), expected_mean)
    np.testing.assert_array_equal(np.array(easy_dataset.normalization.scale), expected_scale)
    assert (easy_dataset.normalization.fit_start, easy_dataset.normalization.fit_stop) == (
        0,
        2457,
    )
    constant = np.array([[1.0, 2.0], [1.0, 6.0]], dtype=np.float64)
    fallback = dataset_builder_module._fit_normalization(
        constant,
        epsilon=0.1,
        fit_start=0,
        fit_stop=2,
    )
    assert fallback.scale[0] == 1.0
    assert fallback.scale[1] == 2.0


def test_unseen_parameters_begin_only_at_post_burn_final_ten_percent(
    easy_dataset: SyntheticDataset,
) -> None:
    truth = easy_dataset.truth
    unseen_start = 3686

    assert np.all(truth["parameter_variant"][:unseen_start] == 0)
    assert np.all(truth["parameter_variant"][unseen_start:] == 1)
    assert not np.array_equal(
        truth["seen_base_log_scales"],
        truth["unseen_base_log_scales"],
    )
    for index in (0, 2457, 3276, 3685, 3686, 4095):
        regime = int(truth["regime_sequence"][index])
        parameter_bank = (
            truth["seen_base_log_scales"]
            if index < unseen_start
            else truth["unseen_base_log_scales"]
        )
        assert truth["base_log_scale_schedule"][index] == parameter_bank[regime]
    difference = easy_dataset.synthetic_provenance.seen_unseen_parameter_difference
    assert difference.base_log_scale_shift == easy_dataset.config.generation.unseen_parameter_shift
    assert difference.applies_from == unseen_start


def test_manifest_has_three_canonical_partitions_and_exact_test_aggregation(
    easy_dataset: SyntheticDataset,
) -> None:
    manifest = easy_dataset.data_manifest
    splits = easy_dataset.physical_splits
    expected_test_payload = {
        "test_seen_regime": splits[2].split_hash,
        "test_unseen_regime": splits[3].split_hash,
    }
    canonical = json.dumps(
        expected_test_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    expected_test_hash = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    assert isinstance(manifest, DataManifest)
    assert tuple(summary.partition for summary in manifest.splits) == (
        SplitPartition.TRAIN,
        SplitPartition.VALIDATION,
        SplitPartition.TEST,
    )
    assert manifest.splits[0].split_hash == splits[0].split_hash
    assert manifest.splits[1].split_hash == splits[1].split_hash
    assert manifest.splits[2].split_hash == expected_test_hash
    assert manifest.splits[2].count == len(splits[2].batch.window_id) + len(
        splits[3].batch.window_id
    )
    assert manifest.dataset_hash == easy_dataset.dataset_hash


def test_runtime_records_and_strict_provenance_are_immutable_round_trippable(
    easy_dataset: SyntheticDataset,
) -> None:
    provenance = easy_dataset.synthetic_provenance
    restored = type(provenance).model_validate_json(provenance.model_dump_json())

    assert restored == provenance
    assert provenance.model_config["strict"] is True
    assert provenance.model_config["frozen"] is True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        type(provenance).model_validate(
            provenance.model_dump(mode="python") | {"unexpected": "value"}
        )
    with pytest.raises(FrozenInstanceError):
        easy_dataset.dataset_hash = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        easy_dataset.physical_splits[0].start = 1  # type: ignore[misc]


# fmt: off
def test_persistence_round_trips_private_arrow_npz_manifest_and_checksums(
    easy_dataset: SyntheticDataset, tmp_path: Path) -> None:
    output_root = tmp_path / "dataset"
    persisted = persist_synthetic_dataset(easy_dataset, output_root)
    assert isinstance(persisted, PersistedSyntheticDataset)
    assert persisted.dataset_hash == easy_dataset.dataset_hash
    assert persisted.output_root == output_root.resolve()
    assert {path.name for path in persisted.files.values()} == REQUIRED_FILES
    assert {path.name for path in output_root.iterdir()} == REQUIRED_FILES
    assert set(persisted.checksums) == REQUIRED_FILES - {"checksums.json"}
    checksum_payload = json.loads(
        (output_root / "checksums.json").read_text(encoding="utf-8"))
    assert checksum_payload == dict(persisted.checksums)
    for filename, digest in persisted.checksums.items():
        assert digest == _sha256(output_root / filename)
    with np.load(output_root / "truth.npz", allow_pickle=False) as truth:
        assert REQUIRED_TRUTH <= set(truth.files)
        assert all(truth[name].dtype != object for name in truth.files)
        for name in REQUIRED_TRUTH:
            np.testing.assert_array_equal(truth[name], easy_dataset.truth[name])
    composite_type = dataset_builder_module._CompositeManifest
    composite = composite_type.model_validate_json(
        (output_root / "manifest.json").read_text(encoding="utf-8"))
    assert composite.data_manifest == easy_dataset.data_manifest
    assert composite.synthetic_provenance == easy_dataset.synthetic_provenance
    with pytest.raises(ValidationError, match="extra_forbidden"):
        composite_type.model_validate(composite.model_dump(mode="python") | {"unexpected": True})
    normalization = json.loads(
        (output_root / "normalization.json").read_text(encoding="utf-8"))
    assert normalization == easy_dataset.normalization.model_dump(mode="json")
    resolved_config = yaml.safe_load(
        (output_root / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved_config == easy_dataset.config.model_dump(mode="json")
    for split in easy_dataset.physical_splits:
        arrow_path = output_root / f"windows_{split.name}.arrow"
        assert _sha256(arrow_path) == split.split_hash
        with pa.ipc.open_file(arrow_path) as reader:
            assert reader.num_record_batches == 1
            schema = reader.schema
            assert tuple((field.name, field.type, field.nullable)
                for field in schema) == EXPECTED_ARROW_FIELDS
            assert tuple(schema.metadata or {}) == (b"contract_schema_version",
                b"physical_split", b"tensor_dtype", b"x_shape", b"y_shape")
            assert schema.metadata[b"contract_schema_version"] == b"1.0.0"
            assert schema.metadata[b"physical_split"] == split.name.encode()
            assert schema.metadata[b"x_shape"] == b"[48,4]"
            table = reader.read_all()
        round_tripped = dataset_builder_module._window_batch_from_arrow_table(
            table, physical_split=split.name)
        assert isinstance(round_tripped, WindowBatch)
        assert round_tripped.window_id == split.batch.window_id
        assert torch.equal(round_tripped.x, split.batch.x)
        assert round_tripped.y is not None and split.batch.y is not None
        assert torch.equal(round_tripped.y, split.batch.y)
        metadata = json.loads(table["metadata_json"][0].as_py())
        assert tuple(metadata) == ("contract_schema_version", "physical_split",
            "tensor_dtype", "x_shape", "y_shape")
        if split.name == "train":
            rows = table["metadata_json"].to_pylist()
            rows[-1] = "{}"
            drifted = table.set_column(19, table.schema.field(19), pa.array(rows))
            with pytest.raises(ValueError, match="metadata_json"):
                dataset_builder_module._window_batch_from_arrow_table(
                    drifted, physical_split=split.name)
            x_column = table["x"].combine_chunks()
            offsets = x_column.offsets.to_numpy(zero_copy_only=False).copy()
            offsets[1] -= 1
            ragged = pa.LargeListArray.from_arrays(pa.array(offsets), x_column.values)
            corrupt = table.set_column(1, table.schema.field(1), ragged)
            with pytest.raises(ValueError, match="x outer offsets"):
                dataset_builder_module._window_batch_from_arrow_table(
                    corrupt, physical_split=split.name)
def test_persistence_failure_cleans_only_owned_staging_and_publishes_nothing(
    easy_dataset: SyntheticDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "dataset"
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("owned by caller", encoding="utf-8")

    def fail_npz(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected npz failure")

    monkeypatch.setattr(dataset_builder_module.np, "savez", fail_npz)
    with pytest.raises(OSError, match="injected npz failure"):
        persist_synthetic_dataset(easy_dataset, output_root)

    assert not output_root.exists()
    assert unrelated.read_text(encoding="utf-8") == "owned by caller"
    assert not tuple(tmp_path.glob(".dataset.staging-*"))
    swapped: dict[str, Path] = {}
    def swap_npz(path: str | os.PathLike[str], *_args: object, **_kwargs: object) -> None:
        staging = Path(path).parent
        original = staging.with_name(f"{staging.name}.original")
        staging.rename(original)
        staging.mkdir()
        (staging / "sentinel").write_text("attacker-owned", encoding="utf-8")
        swapped.update(staging=staging, original=original)
        raise OSError("injected staging swap")
    monkeypatch.setattr(dataset_builder_module.np, "savez", swap_npz)
    with pytest.raises(OSError, match="staging swap"):
        persist_synthetic_dataset(easy_dataset, tmp_path / "swapped")
    assert (swapped["staging"] / "sentinel").read_text() == "attacker-owned"
    assert swapped["original"].is_dir() and not (tmp_path / "swapped").exists()


def test_persistence_rejects_existing_parent_missing_dotdot_and_reparse_paths(
    easy_dataset: SyntheticDataset,
    tmp_path: Path,
) -> None:
    altered = list(easy_dataset.data_manifest.splits)
    altered[0] = altered[0].model_copy(update={"count": altered[0].count + 1})
    forged_manifest = easy_dataset.data_manifest.model_copy(update={"splits": tuple(altered)})
    forged_provenance = easy_dataset.synthetic_provenance.model_copy(
        update={"research_status": "FORGED"})
    swapped_splits = (easy_dataset.physical_splits[1], easy_dataset.physical_splits[0],
        *easy_dataset.physical_splits[2:])
    for name, forged in (("hash", replace(easy_dataset, dataset_hash="sha256:" + "0" * 64)),
        ("manifest", replace(easy_dataset, data_manifest=forged_manifest)),
        ("order", replace(easy_dataset, physical_splits=swapped_splits)),
        ("provenance", replace(easy_dataset, synthetic_provenance=forged_provenance))):
        with pytest.raises(ValueError, match="identity mismatch"):
            persist_synthetic_dataset(forged, tmp_path / name)
        assert not (tmp_path / name).exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "caller.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        persist_synthetic_dataset(easy_dataset, existing)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"

    with pytest.raises(ValueError, match=r"parent.*exist"):
        persist_synthetic_dataset(easy_dataset, tmp_path / "missing" / "dataset")

    lexical_parent = tmp_path / "lexical"
    lexical_parent.mkdir()
    with pytest.raises(ValueError, match=r"dot|\\.\\."):
        persist_synthetic_dataset(
            easy_dataset,
            lexical_parent / ".." / "escaped",
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _make_directory_link(linked_parent, real_parent)
    with pytest.raises(ValueError, match=r"symlink|junction|reparse"):
        persist_synthetic_dataset(easy_dataset, linked_parent / "dataset")
    assert not (real_parent / "dataset").exists()
