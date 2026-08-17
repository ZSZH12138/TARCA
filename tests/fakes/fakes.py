from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
from torch import Tensor

from tarca.contracts.common import ArtifactRef
from tarca.contracts.concepts import ConceptBatch
from tarca.contracts.data import WindowBatch
from tarca.contracts.forecast import ForecastDistribution
from tarca.contracts.future import (
    ConceptIntervention,
    InterventionPairSet,
    LeakageAudit,
)
from tarca.contracts.interventions import InterventionSite, InterventionSpec


class FakePredictor:
    adapter_name = "fake-predictor"
    model_hash = "sha256:" + "1" * 64
    is_frozen = True

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        horizon = len(batch.forecast_time[0])
        target_names = batch.target_names or ("target",)
        shape = (batch.x.shape[0], horizon, len(target_names))
        mean = torch.zeros(shape, dtype=batch.x.dtype, device=batch.x.device)
        return ForecastDistribution(
            mean=mean,
            scale=torch.ones_like(mean),
            quantiles={0.5: mean},
            logits=None,
            samples=None,
            window_id=batch.window_id,
            target_names=target_names,
        )


class FakeMechanisticAdapter(FakePredictor):
    def list_intervention_sites(self) -> tuple[InterventionSite, ...]:
        return (
            InterventionSite(
                site_name="fake.site",
                layer=0,
                tensor_rank=3,
                batch_axis=0,
                variable_axis=1,
                patch_axis=None,
                feature_axis=2,
                shape_template=(None, 1, 1),
            ),
        )

    def capture(
        self, batch: WindowBatch, sites: tuple[InterventionSite, ...]
    ) -> Mapping[str, Tensor]:
        return {site.site_name: batch.x for site in sites}

    def intervene(
        self, base: WindowBatch, source: WindowBatch, spec: InterventionSpec
    ) -> ForecastDistribution:
        return self.predict_distribution(base)


class FakeConceptExtractor:
    def compute(self, batch: WindowBatch) -> ConceptBatch:
        values = torch.zeros((batch.x.shape[0], 1), dtype=batch.x.dtype, device=batch.x.device)
        return ConceptBatch(
            values=values,
            valid_mask=torch.ones_like(values, dtype=torch.bool),
            names=("fake_concept",),
            window_id=batch.window_id,
            computed_from_history_only=True,
            definition_version="fake-v1",
        )

    def leakage_audit(self, batch: WindowBatch) -> LeakageAudit:
        return LeakageAudit(passed=True)


class FakeHighLevelInterventionModel:
    def intervene(
        self,
        base: ForecastDistribution,
        source: ForecastDistribution,
        concept_intervention: ConceptIntervention,
    ) -> ForecastDistribution:
        return source


class FakeOTBackend:
    def solve(self, source: object, target: object) -> InterventionPairSet:
        return InterventionPairSet(pair_ids=("fake-pair",))


class FakeArtifactStore:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def publish_atomic(self, value: object, artifact_type: str) -> ArtifactRef:
        payload = repr(value).encode("utf-8")
        content_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
        artifact_id = f"artifact:{artifact_type}:{content_hash[7:19]}"
        if artifact_id in self._values:
            raise ValueError("completed artifact identity already exists")
        self._values[artifact_id] = value
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash=content_hash,
            schema_version="contract-only-v1",
            relative_path=f"tests/fakes/{artifact_id.replace(':', '_')}.bin",
        )

    def verify_artifact(self, reference: ArtifactRef) -> bool:
        return reference.artifact_id in self._values

    def load_typed(self, reference: ArtifactRef, expected_type: type[object]) -> object:
        value = self._values[reference.artifact_id]
        if not isinstance(value, expected_type):
            raise TypeError("stored value does not match expected type")
        return value

    def resolve_artifact(self, reference: ArtifactRef) -> ArtifactRef:
        return reference
