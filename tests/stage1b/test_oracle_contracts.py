from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from tarca.artifacts.store import LocalArtifactStore
from tarca.contracts import ArtifactRef
from tarca.stage1b.oracle_contracts import SCMTruthManifest, SyntheticConfig
from tarca.stage1b.persistence import GeneratorTruthDataset, build_scm_truth_manifest


def _artifact(artifact_type: str, marker: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"{artifact_type.lower()}-{marker}",
        artifact_type=artifact_type,
        content_hash=marker * 64,
        schema_version="2.0.0",
        relative_path=f"artifacts/{artifact_type.lower()}-{marker}.bin",
    )


def _config() -> SyntheticConfig:
    return SyntheticConfig(
        name="official-l96",
        D=4,
        L=8,
        H=2,
        regimes=2,
        true_delay=(1, 2),
        root_seed=104729,
        burn_in=16,
        total_steps=64,
        generation_settings={"forcing": 10.0, "source": "official"},
        normalization_settings={"fit_partition": "TRAIN"},
    )


def _manifest(**updates: object) -> SCMTruthManifest:
    payload: dict[str, object] = {
        "schema_version": "2.0.0",
        "dataset_hash": "a" * 64,
        "generator_config_hash": "b" * 64,
        "concept_names": ("trend", "scale"),
        "regime_ids": ("seen", "unseen"),
        "true_lags": {"trend": (1,), "scale": (1, 2)},
        "true_graph_ref": _artifact("TRUE_GRAPH", "c"),
        "latent_concepts_ref": _artifact("LATENT_CONCEPTS", "d"),
        "regime_sequence_ref": _artifact("REGIME_SEQUENCE", "e"),
        "exogenous_noise_ref": _artifact("EXOGENOUS_NOISE", "f"),
        "shock_sequence_ref": None,
        "oracle_protocol_hash": "1" * 64,
        "sealed": True,
    }
    payload.update(updates)
    return SCMTruthManifest.model_validate(payload)


def _truth_dataset() -> GeneratorTruthDataset:
    return GeneratorTruthDataset(
        dataset_hash="a" * 64,
        config=_config(),
        concept_names=("trend", "scale"),
        regime_ids=("seen", "unseen"),
        true_lags={"trend": (1,), "scale": (1, 2)},
        true_graph=torch.eye(4, dtype=torch.int8),
        latent_concepts=torch.arange(128, dtype=torch.float64).reshape(64, 2),
        regime_sequence=torch.tensor([0] * 32 + [1] * 32, dtype=torch.int64),
        exogenous_noise=torch.zeros((64, 4), dtype=torch.float64),
        shock_sequence=None,
    )


def test_synthetic_config_is_immutable_and_requires_train_normalization() -> None:
    config = _config()

    assert config.normalization_settings["fit_partition"] == "TRAIN"
    with pytest.raises(ValidationError, match="frozen"):
        config.D = 8  # type: ignore[misc]


def test_truth_manifest_requires_unique_names_and_sealed_state() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _manifest(concept_names=("trend", "trend"))
    with pytest.raises(ValidationError, match="sealed"):
        _manifest(sealed=False)


def test_truth_manifest_rejects_wrong_artifact_roles() -> None:
    with pytest.raises(ValidationError, match="TRUE_GRAPH"):
        _manifest(true_graph_ref=_artifact("EXOGENOUS_NOISE", "9"))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"true_lags": {"trend": (1,)}}, "lag keys"),
        ({"true_lags": {"trend": (0,), "scale": (1,)}}, "positive"),
        ({"regime_ids": ("seen", "seen")}, "unique"),
        (
            {"regime_sequence": torch.full((64,), 2, dtype=torch.int64)},
            "configured regimes",
        ),
        ({"shock_sequence": torch.full((64,), float("nan"))}, "finite"),
    ],
)
def test_truth_dataset_rejects_invalid_generator_truth_before_publish(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_truth_dataset(), **updates)


def test_build_truth_manifest_publishes_separate_verified_artifacts(tmp_path: Path) -> None:
    store = LocalArtifactStore(
        tmp_path,
        producer_stage="stage1b",
        producer_task_id="oracle-contract-test",
        scientific_identity_hash="9" * 64,
    )
    dataset = _truth_dataset()

    manifest = build_scm_truth_manifest(dataset, store)

    assert manifest.sealed is True
    assert manifest.generator_config_hash != manifest.dataset_hash
    refs = (
        manifest.true_graph_ref,
        manifest.latent_concepts_ref,
        manifest.regime_sequence_ref,
        manifest.exogenous_noise_ref,
    )
    assert len({ref.content_hash for ref in refs}) == 4
    assert all(store.verify_artifact(ref) for ref in refs)
    assert all(ref.relative_path is not None for ref in refs)
