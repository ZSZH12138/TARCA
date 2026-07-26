"""Behavior tests for synthetic truth validation and the E01 smoke."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from tarca.data.synthetic import dataset_builder as builder_module
from tarca.data.synthetic import validation as validation_module
from tarca.data.synthetic.dataset_builder import (
    PersistedSyntheticDataset,
    SyntheticDataset,
    build_synthetic_dataset,
    load_synthetic_config,
    persist_synthetic_dataset,
)
from tarca.data.synthetic.validation import (
    ConvergencePoint,
    E01EngineeringSmokeReport,
    ValidationIssue,
    ValidationMetric,
    ValidationReport,
    run_e01_engineering_smoke,
    validate_synthetic_dataset,
)

ROOT = Path(__file__).parents[3]
EASY_CONFIG = ROOT / "configs" / "synthetic" / "synthetic_easy.yaml"
MC_SIZES = (32, 64, 128, 256)
QUANTILES = np.array([0.1, 0.5, 0.9], dtype=np.float64)
GIB = 1024**3
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


@pytest.fixture(scope="module")
def easy_dataset() -> SyntheticDataset:
    return build_synthetic_dataset(load_synthetic_config(EASY_CONFIG))


@pytest.fixture
def persisted_easy(
    easy_dataset: SyntheticDataset,
    tmp_path: Path,
) -> PersistedSyntheticDataset:
    return persist_synthetic_dataset(easy_dataset, tmp_path / "published")


def _unsafe_replace(value: Any, **changes: object) -> Any:
    clone = copy.copy(value)
    for name, replacement in changes.items():
        object.__setattr__(clone, name, replacement)
    return clone


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _replace_truth(
    dataset: SyntheticDataset,
    key: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> SyntheticDataset:
    truth = dict(dataset.truth)
    truth[key] = transform(truth[key])
    return _unsafe_replace(dataset, truth=MappingProxyType(truth))


def _replace_truth_mapping(
    dataset: SyntheticDataset,
    transform: Callable[[dict[str, np.ndarray]], None],
) -> SyntheticDataset:
    truth = dict(dataset.truth)
    transform(truth)
    return _unsafe_replace(dataset, truth=MappingProxyType(truth))


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _update_checksum(
    persisted: PersistedSyntheticDataset,
    filename: str,
) -> PersistedSyntheticDataset:
    checksums = dict(persisted.checksums)
    checksums[filename] = _sha256(persisted.files[filename])
    persisted.files["checksums.json"].write_text(
        json.dumps(checksums, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return _unsafe_replace(persisted, checksums=MappingProxyType(checksums))


def _rewrite_checksum_index(
    persisted: PersistedSyntheticDataset,
    mode: str,
) -> PersistedSyntheticDataset:
    checksums = dict(persisted.checksums)
    if mode == "self":
        checksums["checksums.json"] = "sha256:" + "0" * 64
    elif mode == "missing":
        checksums.pop("truth.npz")
    else:
        checksums["ghost.bin"] = "sha256:" + "0" * 64
    persisted.files["checksums.json"].write_text(
        json.dumps(checksums, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return _unsafe_replace(persisted, checksums=MappingProxyType(checksums))


def _pass_report() -> ValidationReport:
    return ValidationReport(
        "VALIDATION_PASS",
        (),
        (ValidationMetric("truth", 1.0, "boolean", True),),
        False,
        DIGEST_A,
        DIGEST_B,
    )


def _e01_kwargs() -> dict[str, object]:
    points = tuple(ConvergencePoint(size, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0) for size in MC_SIZES)
    return {
        "status": "ENGINEERING_SMOKE_PASS",
        "validation": _pass_report(),
        "convergence": points,
        "trend_mean_rmse": 0.0,
        "scale_mean_rmse": 0.0,
        "scale_std_relative_error": 0.1,
        "conditional_variance_relative_error": 0.1,
        "quantile_normalized_rmse": 0.1,
        "true_delay": 2,
        "estimated_delay": 2,
        "delay_absolute_error": 0,
        "convergence_log_error_slope": -0.5,
        "correct_signature_distance": 0.1,
        "wrong_delay_signature_distance": 0.2,
        "wrong_scale_signature_distance": 0.2,
        "random_concept_signature_distance": 0.3,
        "estimator_variance": 0.0,
        "pair_count": 1,
        "mc_sample_sizes": MC_SIZES,
        "runtime_seconds": 1.0,
        "additional_memory_estimate_bytes": 1024,
        "memory_estimation_method": "static bound with 2x safety factor",
        "output_size_bytes": 0,
        "logical_cpu_count": 1,
        "available_memory_bytes": 1024,
        "gpu_available": False,
        "gpu_used": False,
        "root_seed": 20260725,
        "git_commit": "c" * 40,
        "config_hash": DIGEST_A,
        "data_hash": DIGEST_B,
        "issues": (),
    }


def test_public_api_imports() -> None:
    assert all(
        (
            ValidationIssue,
            ValidationMetric,
            ValidationReport,
            ConvergencePoint,
            E01EngineeringSmokeReport,
            validate_synthetic_dataset,
            run_e01_engineering_smoke,
        )
    )


def test_records_are_deeply_immutable_and_defensive() -> None:
    issues = [ValidationIssue("truth.shape", "truth.x_complete", "bad")]
    metrics = [ValidationMetric("shape", 0.0, "boolean", False, 1.0)]
    report = ValidationReport(
        "VALIDATION_FAIL",
        issues,  # type: ignore[arg-type]
        metrics,  # type: ignore[arg-type]
        False,
        DIGEST_A,
        DIGEST_B,
    )
    issues.clear()
    metrics.clear()
    assert isinstance(report.issues, tuple) and len(report.issues) == 1
    assert isinstance(report.metrics, tuple) and len(report.metrics) == 1
    with pytest.raises(FrozenInstanceError):
        report.status = "VALIDATION_PASS"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ValidationIssue("", "truth.x", "message"),
        lambda: ValidationMetric("metric", math.nan, "ratio", False),
        lambda: ValidationReport(
            "PASS",
            (),
            (),
            False,
            DIGEST_A,
            DIGEST_B,  # type: ignore[arg-type]
        ),
        lambda: ValidationReport("VALIDATION_PASS", (), (), False, "bad", DIGEST_B),
        lambda: ConvergencePoint(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        lambda: E01EngineeringSmokeReport(**{**_e01_kwargs(), "status": "E01_FORMAL_PASS"}),
        lambda: E01EngineeringSmokeReport(**{**_e01_kwargs(), "gpu_used": True}),
        lambda: E01EngineeringSmokeReport(
            **{**_e01_kwargs(), "convergence": _e01_kwargs()["convergence"][:-1]}
        ),
        lambda: E01EngineeringSmokeReport(**{**_e01_kwargs(), "logical_cpu_count": 0}),
    ],
)
def test_records_reject_invalid_or_formal_claims(
    factory: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_signature_distance_is_composite_rms_to_analytic_target() -> None:
    candidate = (
        np.array([1.0, 3.0], dtype=np.float64),
        np.array([2.0], dtype=np.float64),
        np.array([0.0, 4.0, 2.0], dtype=np.float64),
    )
    target = (
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    expected = math.sqrt((1.0 + 4.0 + 1.0 + 0.0 + 16.0 + 4.0) / 6.0)
    assert validation_module._signature_distance(candidate, target) == pytest.approx(expected)


def test_total_error_is_fixed_four_component_rms() -> None:
    expected = math.sqrt((0.1**2 + 0.2**2 + 0.3**2 + 0.4**2) / 4.0)
    assert validation_module._total_error(0.1, 0.2, 0.3, 0.4) == pytest.approx(expected)


def test_valid_easy_dataset_passes(easy_dataset: SyntheticDataset) -> None:
    report = validate_synthetic_dataset(easy_dataset)
    assert report.status == "VALIDATION_PASS"
    assert report.issues == ()
    assert report.persisted_checked is False
    assert report.config_hash == easy_dataset.config_hash
    assert report.data_hash == easy_dataset.dataset_hash
    assert all(metric.passed for metric in report.metrics)


def test_validation_is_independent_of_builder_private_verifiers(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("validation delegated to a builder private helper")

    for name in (
        "_split_boundaries",
        "_fit_normalization",
        "_build_split",
        "_validate_dataset",
        "_digest",
        "_canonical",
        "_CompositeManifest",
        "_window_batch_from_arrow_table",
        "_dataset_hash",
    ):
        monkeypatch.setattr(builder_module, name, forbidden)
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_PASS"


def test_validation_runs_dataset_coupled_paired_oracle(
    easy_dataset: SyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = validation_module.oracle.replay_paired_counterfactual

    def record_call(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(validation_module.oracle, "replay_paired_counterfactual", record_call)
    report = validate_synthetic_dataset(easy_dataset)
    assert report.status == "VALIDATION_PASS"
    assert len(calls) >= 3
    for call in calls:
        bank = call["noise_bank"]
        assert bank.observation_innovations.shape[1] == easy_dataset.config.D
        assert len(call["dynamics_schedule"]) == easy_dataset.config.H
    assert any(
        (intervention := call["intervention"]) is not None
        and intervention.concept == "scale"
        and intervention.source_value != call["current_scale"]
        for call in calls
    )
    assert any(metric.name == "oracle_invariants" and metric.passed for metric in report.metrics)


def test_validation_rejects_oracle_that_ignores_nontrivial_interventions(
    easy_dataset: SyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = validation_module.oracle.replay_paired_counterfactual

    def ignore_nontrivial_intervention(**kwargs: Any) -> object:
        intervention = kwargs["intervention"]
        current = (
            kwargs["current_trend"]
            if intervention is not None and intervention.concept == "trend"
            else kwargs["current_scale"]
        )
        effective = (
            None
            if intervention is not None and intervention.source_value != current
            else intervention
        )
        return original(**{**kwargs, "intervention": effective})

    monkeypatch.setattr(
        validation_module.oracle,
        "replay_paired_counterfactual",
        ignore_nontrivial_intervention,
    )
    report = validate_synthetic_dataset(easy_dataset)
    assert report.status == "VALIDATION_FAIL"
    assert "oracle.intervention_effect" in _codes(report)


@pytest.mark.parametrize(
    ("transform", "expected_code"),
    [
        (lambda a: np.asarray(a, dtype=np.float32), "truth.dtype"),
        (lambda a: _readonly(a[:-1]), "truth.shape"),
        (lambda a: np.array(a, copy=True), "truth.mutable"),
        (lambda a: np.full(a.shape, np.nan), "truth.nonfinite"),
    ],
)
def test_truth_array_contract_corruption_is_diagnosed(
    easy_dataset: SyntheticDataset,
    transform: Callable[[np.ndarray], np.ndarray],
    expected_code: str,
) -> None:
    report = validate_synthetic_dataset(_replace_truth(easy_dataset, "x_complete", transform))
    assert report.status == "VALIDATION_FAIL"
    assert expected_code in _codes(report)
    assert any(issue.location == "truth.x_complete" for issue in report.issues)


@pytest.mark.parametrize(
    ("mode", "expected_code"), [("missing", "truth.keys"), ("extra", "truth.keys")]
)
def test_truth_requires_exact_key_set(
    easy_dataset: SyntheticDataset,
    mode: str,
    expected_code: str,
) -> None:
    def alter(truth: dict[str, np.ndarray]) -> None:
        if mode == "missing":
            truth.pop("x_complete")
        else:
            truth["unexpected"] = _readonly(np.zeros(1, dtype=np.float64))

    report = validate_synthetic_dataset(_replace_truth_mapping(easy_dataset, alter))
    assert report.status == "VALIDATION_FAIL"
    assert expected_code in _codes(report)


@pytest.mark.parametrize(
    ("key", "index", "expected_code"),
    [
        ("trend", (1,), "latent.recurrence"),
        ("regime_sequence", (0,), "regime.replay"),
        ("linear_matrix_schedule", (0, 0, 0), "dynamics.schedule"),
        ("true_graph_schedule", (0, 0, 0), "dynamics.true_graph"),
        ("replay_initial_history", (-1, 0), "replay.factual"),
        ("missing_mask", (0, 0), "mask.semantics"),
        ("parameter_variant", (3685,), "dynamics.unseen_boundary"),
    ],
)
def test_semantic_truth_corruption_has_specific_issue(
    easy_dataset: SyntheticDataset,
    key: str,
    index: tuple[int, ...],
    expected_code: str,
) -> None:
    def corrupt(array: np.ndarray) -> np.ndarray:
        changed = np.array(array, copy=True)
        if changed.dtype == np.bool_:
            changed[index] = ~changed[index]
        elif np.issubdtype(changed.dtype, np.integer):
            changed[index] = int(changed[index]) + 1
        else:
            changed[index] += 0.25
        changed.setflags(write=False)
        return changed

    report = validate_synthetic_dataset(_replace_truth(easy_dataset, key, corrupt))
    assert report.status == "VALIDATION_FAIL"
    assert expected_code in _codes(report)


def test_mutable_torch_batch_is_reconstructed_and_rejected(
    easy_dataset: SyntheticDataset,
) -> None:
    split = easy_dataset.physical_splits[0]
    batch = _unsafe_replace(split.batch, x=split.batch.x.clone())
    batch.x[0, 0, 0] += 1.0
    changed = _unsafe_replace(split, batch=batch)
    dataset = _unsafe_replace(
        easy_dataset,
        physical_splits=(changed, *easy_dataset.physical_splits[1:]),
    )
    report = validate_synthetic_dataset(dataset)
    assert report.status == "VALIDATION_FAIL"
    assert "split.standardization" in _codes(report)


def test_train_scaler_is_recomputed_from_complete_train_truth(
    easy_dataset: SyntheticDataset,
) -> None:
    shifted = tuple(value + 0.5 for value in easy_dataset.normalization.mean)
    normalization = easy_dataset.normalization.model_copy(update={"mean": shifted})
    report = validate_synthetic_dataset(_unsafe_replace(easy_dataset, normalization=normalization))
    assert report.status == "VALIDATION_FAIL"
    assert "normalization.train_only" in _codes(report)


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [("dataset_hash", "hash.dataset"), ("provenance", "provenance.identity")],
)
def test_publication_identity_is_independently_recomputed(
    easy_dataset: SyntheticDataset,
    field: str,
    expected_code: str,
) -> None:
    if field == "dataset_hash":
        dataset = _unsafe_replace(easy_dataset, dataset_hash="0" * 64)
    else:
        provenance = easy_dataset.synthetic_provenance.model_copy(
            update={"root_seed": easy_dataset.config.root_seed + 1}
        )
        dataset = _unsafe_replace(easy_dataset, synthetic_provenance=provenance)
    report = validate_synthetic_dataset(dataset)
    assert report.status == "VALIDATION_FAIL"
    assert expected_code in _codes(report)


def test_valid_persisted_dataset_strictly_roundtrips(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_PASS"
    assert report.persisted_checked is True


def test_persisted_exact_file_set_rejects_extra(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    (persisted_easy.output_root / "unexpected.txt").write_text("bad", encoding="utf-8")
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.file_set" in _codes(report)
    assert any(issue.location == "file.unexpected.txt" for issue in report.issues)


@pytest.mark.parametrize("mode", ["self", "missing", "extra"])
def test_checksum_index_is_exactly_eight_nonself_entries(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    mode: str,
) -> None:
    changed = _rewrite_checksum_index(persisted_easy, mode)
    report = validate_synthetic_dataset(easy_dataset, persisted=changed)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.checksum_index" in _codes(report)


def test_checksum_payload_corruption_names_entry(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    persisted_easy.files["config_resolved.yaml"].write_text("name: bad\n", encoding="utf-8")
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.checksum" in _codes(report)
    assert any(issue.location == "checksum.config_resolved.yaml" for issue in report.issues)


def test_checksummed_npz_parse_failure_is_reported(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    persisted_easy.files["truth.npz"].write_bytes(b"not-an-npz")
    changed = _update_checksum(persisted_easy, "truth.npz")
    report = validate_synthetic_dataset(easy_dataset, persisted=changed)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.npz" in _codes(report)
    assert any(issue.location == "file.truth.npz" for issue in report.issues)


@pytest.mark.parametrize("mode", ["object", "missing"])
def test_checksummed_npz_rejects_pickle_or_key_drift(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    mode: str,
) -> None:
    path = persisted_easy.files["truth.npz"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if mode == "object":
        arrays["x_complete"] = np.array([object()], dtype=object)
    else:
        arrays.pop("x_complete")
    np.savez(path, **arrays)
    changed = _update_checksum(persisted_easy, "truth.npz")
    report = validate_synthetic_dataset(easy_dataset, persisted=changed)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.npz" in _codes(report)


def _rewrite_arrow(
    persisted: PersistedSyntheticDataset,
    *,
    two_batches: bool,
) -> PersistedSyntheticDataset:
    filename = "windows_train.arrow"
    path = persisted.files[filename]
    table = pa.ipc.open_file(pa.BufferReader(path.read_bytes())).read_all()
    if two_batches:
        midpoint = table.num_rows // 2
        batches = (
            table.slice(0, midpoint).combine_chunks().to_batches()[0],
            table.slice(midpoint).combine_chunks().to_batches()[0],
        )
    else:
        metadata = dict(table.schema.metadata or {})
        metadata[b"tarca.split"] = b"validation"
        table = table.replace_schema_metadata(metadata)
        batches = tuple(table.combine_chunks().to_batches())
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
    return _update_checksum(persisted, filename)


@pytest.mark.parametrize("two_batches", [True, False])
def test_checksummed_arrow_rejects_batch_or_metadata_drift(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    two_batches: bool,
) -> None:
    changed = _rewrite_arrow(persisted_easy, two_batches=two_batches)
    report = validate_synthetic_dataset(easy_dataset, persisted=changed)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.arrow" in _codes(report)
    assert any(issue.location == "file.windows_train.arrow" for issue in report.issues)


def test_checksummed_manifest_semantics_match_memory(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    path = persisted_easy.files["manifest.json"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["synthetic_provenance"]["root_seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed = _update_checksum(persisted_easy, "manifest.json")
    report = validate_synthetic_dataset(easy_dataset, persisted=changed)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.manifest" in _codes(report)


def test_persisted_root_reparse_probe_blocks_reads(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation_module.validation_core,
        "_is_reparse",
        lambda path: Path(path) == persisted_easy.output_root,
    )
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.reparse" in _codes(report)


def test_checksum_index_reparse_probe_is_rejected(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checksum_path = persisted_easy.files["checksums.json"]
    monkeypatch.setattr(
        validation_module.validation_core,
        "_is_reparse",
        lambda path: Path(path) == checksum_path,
    )
    report = validate_synthetic_dataset(easy_dataset, persisted=persisted_easy)
    assert report.status == "VALIDATION_FAIL"
    assert "persistence.reparse" in _codes(report)
    assert any(issue.location == "file.checksums.json" for issue in report.issues)


def test_only_top_level_api_type_errors_raise(easy_dataset: SyntheticDataset) -> None:
    with pytest.raises(TypeError, match="dataset"):
        validate_synthetic_dataset(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="persisted"):
        validate_synthetic_dataset(
            easy_dataset,
            persisted=easy_dataset,  # type: ignore[arg-type]
        )


@pytest.fixture(scope="module")
def e01_report(easy_dataset: SyntheticDataset) -> E01EngineeringSmokeReport:
    return run_e01_engineering_smoke(easy_dataset, pair_count=2)


def test_e01_two_analytic_subcases_meet_frozen_thresholds(
    e01_report: E01EngineeringSmokeReport,
) -> None:
    assert e01_report.status == "ENGINEERING_SMOKE_PASS"
    assert e01_report.validation.status == "VALIDATION_PASS"
    assert e01_report.trend_mean_rmse <= 1e-12
    assert e01_report.scale_std_relative_error <= 0.20
    assert e01_report.conditional_variance_relative_error <= 0.30
    assert e01_report.quantile_normalized_rmse <= 0.35
    assert (e01_report.true_delay, e01_report.estimated_delay) == (2, 2)
    assert e01_report.delay_absolute_error == 0


def test_e01_wrong_and_random_composites_are_strictly_worse(
    e01_report: E01EngineeringSmokeReport,
) -> None:
    correct = e01_report.correct_signature_distance
    assert e01_report.wrong_delay_signature_distance > correct
    assert e01_report.wrong_scale_signature_distance > correct
    assert e01_report.random_concept_signature_distance > correct


def test_e01_nested_mc_has_endpoint_improvement_and_negative_slope(
    e01_report: E01EngineeringSmokeReport,
) -> None:
    assert e01_report.mc_sample_sizes == MC_SIZES
    assert tuple(point.mc_samples for point in e01_report.convergence) == MC_SIZES
    for point in e01_report.convergence:
        expected = math.sqrt(
            (
                point.mean_rmse**2
                + point.std_relative_error**2
                + point.conditional_variance_relative_error**2
                + point.quantile_normalized_rmse**2
            )
            / 4.0
        )
        assert point.total_error == pytest.approx(expected)
        assert point.estimator_variance >= 0.0
    assert e01_report.convergence[-1].total_error < e01_report.convergence[0].total_error
    assert e01_report.convergence_log_error_slope < 0.0


def test_e01_reports_bounded_cpu_only_resources(
    e01_report: E01EngineeringSmokeReport,
) -> None:
    assert e01_report.pair_count == 2
    assert e01_report.runtime_seconds <= 1800.0
    assert e01_report.additional_memory_estimate_bytes <= 4 * GIB
    assert "safety factor" in e01_report.memory_estimation_method
    assert e01_report.output_size_bytes <= 2 * GIB
    assert e01_report.logical_cpu_count >= 1
    assert e01_report.available_memory_bytes >= 0
    assert isinstance(e01_report.gpu_available, bool)
    assert e01_report.gpu_used is False
    assert e01_report.root_seed == 20260725
    assert len(e01_report.git_commit) == 40
    assert e01_report.config_hash == e01_report.validation.config_hash
    assert e01_report.data_hash == e01_report.validation.data_hash


def test_e01_persisted_output_size_is_actual_nine_files(
    easy_dataset: SyntheticDataset,
    persisted_easy: PersistedSyntheticDataset,
) -> None:
    report = run_e01_engineering_smoke(
        easy_dataset,
        persisted=persisted_easy,
        pair_count=1,
    )
    assert report.output_size_bytes == sum(
        path.stat().st_size for path in persisted_easy.files.values()
    )


@pytest.mark.parametrize(
    ("sizes", "pairs", "error_type"),
    [
        ((32, 64, 128), 1, ValueError),
        ((32, 64, 128, 257), 1, ValueError),
        (MC_SIZES, 0, ValueError),
        (MC_SIZES, 17, ValueError),
        (MC_SIZES, 1.5, TypeError),
    ],
)
def test_e01_rejects_scope_expansion(
    easy_dataset: SyntheticDataset,
    sizes: tuple[int, ...],
    pairs: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        run_e01_engineering_smoke(
            easy_dataset,
            mc_sample_sizes=sizes,
            pair_count=pairs,  # type: ignore[arg-type]
        )


def test_e01_non_easy_returns_engineering_fail(
    easy_dataset: SyntheticDataset,
) -> None:
    config = easy_dataset.config.model_copy(update={"name": "synthetic_medium"})
    report = run_e01_engineering_smoke(
        _unsafe_replace(easy_dataset, config=config),
        pair_count=1,
    )
    assert report.status == "ENGINEERING_SMOKE_FAIL"
    assert any(issue.code == "e01.easy_only" for issue in report.issues)


def test_e01_rejects_frozen_easy_config_drift(easy_dataset: SyntheticDataset) -> None:
    changed_config = easy_dataset.config.model_copy(update={"true_delay": 1})
    changed_dataset = build_synthetic_dataset(changed_config)
    report = run_e01_engineering_smoke(changed_dataset, pair_count=1)
    assert report.status == "ENGINEERING_SMOKE_FAIL"
    assert any(issue.code == "e01.easy_contract" for issue in report.issues)
