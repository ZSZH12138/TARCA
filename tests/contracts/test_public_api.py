from __future__ import annotations

import doctest
import importlib
from pathlib import Path

import tarca.contracts as contracts

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DOCUMENT = REPOSITORY_ROOT / "docs" / "stage1_unified_data_contract.md"

EXPECTED_PUBLIC_EXPORTS = (
    "CONTRACT_SCHEMA_VERSION",
    "ArtifactLayout",
    "ConceptBatch",
    "DataManifest",
    "DataSplitSummary",
    "ForecastDistribution",
    "ForecastModelAdapter",
    "InterventionKind",
    "InterventionPair",
    "InterventionSite",
    "InterventionSpec",
    "JSONMetadata",
    "JSONScalar",
    "JSONValue",
    "MetricRecord",
    "RegimeRelation",
    "RunManifest",
    "RunStatus",
    "SplitPartition",
    "WindowBatch",
    "WindowContractSummary",
    "basis_orthonormality_tolerance",
    "intervention_pairs_schema",
    "metrics_by_regime_schema",
    "predictions_schema",
    "validate_arrow_schema",
    "validate_disjoint_window_partitions",
    "validate_intervention_pair_partitions",
    "validate_spec_against_site",
)

EXPORT_MODULES = {
    "CONTRACT_SCHEMA_VERSION": "tarca.contracts.version",
    "ArtifactLayout": "tarca.contracts.artifacts",
    "ConceptBatch": "tarca.contracts.concepts",
    "DataManifest": "tarca.contracts.manifests",
    "DataSplitSummary": "tarca.contracts.manifests",
    "ForecastDistribution": "tarca.contracts.forecast",
    "ForecastModelAdapter": "tarca.contracts.adapters",
    "InterventionKind": "tarca.contracts.types",
    "InterventionPair": "tarca.contracts.manifests",
    "InterventionSite": "tarca.contracts.interventions",
    "InterventionSpec": "tarca.contracts.interventions",
    "JSONMetadata": "tarca.contracts.types",
    "JSONScalar": "tarca.contracts.types",
    "JSONValue": "tarca.contracts.types",
    "MetricRecord": "tarca.contracts.manifests",
    "RegimeRelation": "tarca.contracts.types",
    "RunManifest": "tarca.contracts.manifests",
    "RunStatus": "tarca.contracts.types",
    "SplitPartition": "tarca.contracts.types",
    "WindowBatch": "tarca.contracts.data",
    "WindowContractSummary": "tarca.contracts.manifests",
    "basis_orthonormality_tolerance": "tarca.contracts.interventions",
    "intervention_pairs_schema": "tarca.contracts.arrow_schemas",
    "metrics_by_regime_schema": "tarca.contracts.arrow_schemas",
    "predictions_schema": "tarca.contracts.arrow_schemas",
    "validate_arrow_schema": "tarca.contracts.arrow_schemas",
    "validate_disjoint_window_partitions": "tarca.contracts.manifests",
    "validate_intervention_pair_partitions": "tarca.contracts.manifests",
    "validate_spec_against_site": "tarca.contracts.interventions",
}


def test_public_api_has_the_exact_canonical_export_tuple() -> None:
    assert isinstance(contracts.__all__, tuple)
    assert contracts.__all__ == EXPECTED_PUBLIC_EXPORTS
    assert tuple(EXPORT_MODULES) == EXPECTED_PUBLIC_EXPORTS


def test_every_root_export_is_the_original_definition() -> None:
    for export_name, module_name in EXPORT_MODULES.items():
        implementation_module = importlib.import_module(module_name)
        assert getattr(contracts, export_name) is getattr(implementation_module, export_name)


def test_representative_canonical_imports_work_without_compatibility_aliases() -> None:
    from tarca.contracts import (
        CONTRACT_SCHEMA_VERSION,
        ArtifactLayout,
        ForecastModelAdapter,
        InterventionSpec,
        WindowBatch,
        predictions_schema,
    )

    assert CONTRACT_SCHEMA_VERSION == "1.0.0"
    assert WindowBatch is importlib.import_module("tarca.contracts.data").WindowBatch
    assert (
        InterventionSpec
        is importlib.import_module("tarca.contracts.interventions").InterventionSpec
    )
    assert (
        ForecastModelAdapter
        is importlib.import_module("tarca.contracts.adapters").ForecastModelAdapter
    )
    assert ArtifactLayout is importlib.import_module("tarca.contracts.artifacts").ArtifactLayout
    assert (
        predictions_schema
        is importlib.import_module("tarca.contracts.arrow_schemas").predictions_schema
    )
    assert "StrictContractModel" not in contracts.__all__
    assert not hasattr(contracts, "StrictContractModel")


def test_minimal_window_batch_documentation_example_is_executable() -> None:
    assert CONTRACT_DOCUMENT.is_file(), f"missing contract document: {CONTRACT_DOCUMENT}"

    result = doctest.testfile(
        str(CONTRACT_DOCUMENT),
        module_relative=False,
        encoding="utf-8",
    )

    assert result.attempted > 0
    assert result.failed == 0
