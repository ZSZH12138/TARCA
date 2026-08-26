from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from tarca.artifacts.store import ArtifactStore
from tarca.contracts import ArtifactRef, JSONValue, canonical_json_bytes, canonical_json_hash
from tarca.stage1b.oracle_contracts import SCMTruthManifest, SyntheticConfig


@dataclass(frozen=True, slots=True)
class GeneratorTruthDataset:
    dataset_hash: str
    config: SyntheticConfig
    concept_names: tuple[str, ...]
    regime_ids: tuple[str, ...]
    true_lags: Mapping[str, tuple[int, ...]]
    true_graph: Tensor
    latent_concepts: Tensor
    regime_sequence: Tensor
    exogenous_noise: Tensor
    shock_sequence: Tensor | None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.dataset_hash) is None:
            raise ValueError("dataset_hash must be a lowercase SHA-256")
        if (
            not self.concept_names
            or any(not name.strip() for name in self.concept_names)
            or len(self.concept_names) != len(set(self.concept_names))
        ):
            raise ValueError("concept names must be nonempty and unique")
        if (
            len(self.regime_ids) != self.config.regimes
            or any(not regime_id.strip() for regime_id in self.regime_ids)
            or len(self.regime_ids) != len(set(self.regime_ids))
        ):
            raise ValueError("regime IDs must be nonempty, unique, and match configured count")
        if set(self.true_lags) != set(self.concept_names):
            raise ValueError("true lag keys must exactly match concept names")
        copied_lags: dict[str, tuple[int, ...]] = {}
        for concept_name in self.concept_names:
            lags = tuple(self.true_lags[concept_name])
            if (
                not lags
                or any(
                    isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0 for lag in lags
                )
                or len(lags) != len(set(lags))
            ):
                raise ValueError("true lags must be nonempty, unique, positive integers")
            copied_lags[concept_name] = lags
        object.__setattr__(self, "true_lags", MappingProxyType(copied_lags))
        expected = {
            "true_graph": (self.config.D, self.config.D),
            "latent_concepts": (self.config.total_steps, len(self.concept_names)),
            "regime_sequence": (self.config.total_steps,),
            "exogenous_noise": (self.config.total_steps, self.config.D),
        }
        tensors = {
            "true_graph": self.true_graph,
            "latent_concepts": self.latent_concepts,
            "regime_sequence": self.regime_sequence,
            "exogenous_noise": self.exogenous_noise,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected[name]:
                raise ValueError(f"{name} does not match generator truth shape")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, tensor.detach().cpu().contiguous().clone())
        if self.regime_sequence.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("regime sequence must contain integer regime indices")
        if not bool(
            ((self.regime_sequence >= 0) & (self.regime_sequence < self.config.regimes)).all()
        ):
            raise ValueError("regime sequence contains values outside configured regimes")
        if self.shock_sequence is not None:
            if not isinstance(self.shock_sequence, Tensor):
                raise ValueError("shock sequence must be a tensor")
            if self.shock_sequence.shape[0] != self.config.total_steps:
                raise ValueError("shock sequence length must match total_steps")
            if self.shock_sequence.is_floating_point() and not bool(
                torch.isfinite(self.shock_sequence).all()
            ):
                raise ValueError("shock sequence must contain finite values")
            object.__setattr__(
                self,
                "shock_sequence",
                self.shock_sequence.detach().cpu().contiguous().clone(),
            )


def _tensor_bytes(tensor: Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    metadata: dict[str, JSONValue] = {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }
    payload = contiguous.numpy().tobytes()
    if not isinstance(payload, bytes):
        raise TypeError("tensor serialization must produce bytes")
    return canonical_json_bytes(metadata) + b"\n" + payload


def _publish_tensor(
    store: ArtifactStore,
    tensor: Tensor,
    artifact_type: str,
) -> ArtifactRef:
    return store.publish_bytes(
        _tensor_bytes(tensor),
        artifact_type=artifact_type,
        media_type="application/vnd.tarca.tensor-v2",
        schema_version="2.0.0",
    )


def build_scm_truth_manifest(
    dataset: GeneratorTruthDataset,
    store: ArtifactStore,
) -> SCMTruthManifest:
    graph_ref = _publish_tensor(store, dataset.true_graph, "TRUE_GRAPH")
    concepts_ref = _publish_tensor(store, dataset.latent_concepts, "LATENT_CONCEPTS")
    regimes_ref = _publish_tensor(store, dataset.regime_sequence, "REGIME_SEQUENCE")
    noise_ref = _publish_tensor(store, dataset.exogenous_noise, "EXOGENOUS_NOISE")
    shock_ref = (
        None
        if dataset.shock_sequence is None
        else _publish_tensor(store, dataset.shock_sequence, "SHOCK_SEQUENCE")
    )
    generator_hash = canonical_json_hash(dataset.config)
    protocol_hash = canonical_json_hash(
        {
            "protocol": "TARCA-STAGE1B-GENERATOR-OWNED-ORACLE-2.0",
            "dataset_hash": dataset.dataset_hash,
            "generator_config_hash": generator_hash,
            "concept_names": list(dataset.concept_names),
            "regime_ids": list(dataset.regime_ids),
            "true_lags": {name: list(lags) for name, lags in dataset.true_lags.items()},
            "artifact_hashes": [
                graph_ref.content_hash,
                concepts_ref.content_hash,
                regimes_ref.content_hash,
                noise_ref.content_hash,
                None if shock_ref is None else shock_ref.content_hash,
            ],
        }
    )
    return SCMTruthManifest(
        schema_version="2.0.0",
        dataset_hash=dataset.dataset_hash,
        generator_config_hash=generator_hash,
        concept_names=dataset.concept_names,
        regime_ids=dataset.regime_ids,
        true_lags=dict(dataset.true_lags),
        true_graph_ref=graph_ref,
        latent_concepts_ref=concepts_ref,
        regime_sequence_ref=regimes_ref,
        exogenous_noise_ref=noise_ref,
        shock_sequence_ref=shock_ref,
        oracle_protocol_hash=protocol_hash,
        sealed=True,
    )
