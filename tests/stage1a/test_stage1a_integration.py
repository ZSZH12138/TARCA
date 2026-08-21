from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import torch

from tarca.artifacts import LocalArtifactStore
from tarca.contracts import (
    AccessScope,
    DatasetRegistryManifest,
    DatasetSpec,
    DatasetWindowPartition,
    ForecastDistribution,
    ForecastPredictor,
    WindowBatch,
    validate_forecast_distribution,
)
from tarca.contracts.arrow_schemas import PREDICTIONS_SCHEMA
from tarca.data import PersistedDatasetRepository


class PersistedFixture(Protocol):
    repo_root: Path
    dataset: DatasetSpec
    registry: DatasetRegistryManifest


PersistedFixtureFactory = Callable[..., PersistedFixture]


class FrozenFixturePredictor:
    adapter_name = "fixture-predictor"
    model_hash = "c" * 64
    is_frozen = True

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        if batch.y is None:
            raise ValueError("fixture predictor requires labels")
        return ForecastDistribution(
            mean=batch.y + 0.25,
            scale=torch.full_like(batch.y, 0.1),
            quantiles={},
            logits=None,
            samples=None,
            window_id=batch.window_id,
            target_names=batch.target_names,
        )


def test_stage1a_minimal_typed_data_to_verified_artifact_loop(
    persisted_fixture_factory: PersistedFixtureFactory,
) -> None:
    fixture = persisted_fixture_factory(partitions=(DatasetWindowPartition.TRAIN,))
    repository = PersistedDatasetRepository(fixture.repo_root, fixture.registry)
    entry = repository.resolve_dataset(fixture.dataset)
    assert repository.hash_dataset(
        fixture.dataset, AccessScope(sealed=False, scope_name="integration")
    ) == entry.expected_dataset_hash
    batch = repository.build_windows(
        fixture.dataset,
        DatasetWindowPartition.TRAIN,
        AccessScope(sealed=False, scope_name="integration"),
    )

    predictor: ForecastPredictor = FrozenFixturePredictor()
    distribution = validate_forecast_distribution(predictor.predict_distribution(batch))
    table = pa.Table.from_pylist(
        [
            {
                "window_id": batch.window_id[0],
                "split": "TRAIN",
                "forecast_time": batch.forecast_time[0][0],
                "horizon": 0,
                "target": batch.target_names[0],
                "y_true": float(batch.y[0, 0, 0]) if batch.y is not None else None,
                "mean": float(distribution.mean[0, 0, 0]),
                "scale": (
                    float(distribution.scale[0, 0, 0])
                    if distribution.scale is not None
                    else None
                ),
            }
        ],
        schema=PREDICTIONS_SCHEMA,
    )
    store = LocalArtifactStore(
        fixture.repo_root,
        producer_stage="STAGE1A",
        producer_task_id="minimal-integration",
        scientific_identity_hash=entry.expected_dataset_hash,
    )
    ref = store.publish_arrow(table, PREDICTIONS_SCHEMA, "PREDICTIONS")

    assert store.load_arrow(ref, PREDICTIONS_SCHEMA).equals(table)
    assert store.verify_artifact(ref)
