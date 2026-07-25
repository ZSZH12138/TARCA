"""Versioned data contracts for TARCA workflows."""

from .adapters import ForecastModelAdapter
from .arrow_schemas import (
    intervention_pairs_schema,
    metrics_by_regime_schema,
    predictions_schema,
    validate_arrow_schema,
)
from .artifacts import ArtifactLayout
from .concepts import ConceptBatch
from .data import WindowBatch
from .forecast import ForecastDistribution
from .interventions import (
    InterventionSite,
    InterventionSpec,
    basis_orthonormality_tolerance,
    validate_spec_against_site,
)
from .manifests import (
    DataManifest,
    DataSplitSummary,
    InterventionPair,
    MetricRecord,
    RunManifest,
    WindowContractSummary,
    validate_disjoint_window_partitions,
    validate_intervention_pair_partitions,
)
from .types import (
    InterventionKind,
    JSONMetadata,
    JSONScalar,
    JSONValue,
    RegimeRelation,
    RunStatus,
    SplitPartition,
)
from .version import CONTRACT_SCHEMA_VERSION

__all__ = (
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
