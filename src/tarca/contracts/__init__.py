"""Versioned data contracts for TARCA workflows.

The canonical Phase 8 worker needs only tensor contracts. Persistent artifact
and manifest contracts remain available when optional Pydantic is installed,
but they must not make the server image unable to import training models.
"""

from .adapters import (
    MODEL_CAPABILITY_PROTOCOL_VERSION,
    ForecastModelAdapter,
    ForecastPredictor,
)
from .concepts import ConceptBatch
from .data import WindowBatch
from .forecast import ForecastDistribution
from .interventions import (
    InterventionSite,
    InterventionSpec,
    basis_orthonormality_tolerance,
    validate_spec_against_site,
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

try:
    from .arrow_schemas import (
        intervention_pairs_schema,
        metrics_by_regime_schema,
        predictions_schema,
        validate_arrow_schema,
    )
except ModuleNotFoundError as error:
    if error.name != "pyarrow":
        raise

    def _missing_arrow_dependency(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("pyarrow is required only for Arrow contract operations")

    intervention_pairs_schema = _missing_arrow_dependency
    metrics_by_regime_schema = _missing_arrow_dependency
    predictions_schema = _missing_arrow_dependency
    validate_arrow_schema = _missing_arrow_dependency

_PYDANTIC_EXPORTS = {
    "ArtifactLayout",
    "DataManifest",
    "DataSplitSummary",
    "InterventionPair",
    "MetricRecord",
    "RunManifest",
    "WindowContractSummary",
    "validate_disjoint_window_partitions",
    "validate_intervention_pair_partitions",
}

try:
    from .artifacts import ArtifactLayout
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
except ModuleNotFoundError as error:
    if error.name != "pydantic":
        raise

    def __getattr__(name: str) -> object:
        if name in _PYDANTIC_EXPORTS:
            raise RuntimeError(
                "pydantic is required only for persistent artifact/manifest contracts"
            )
        raise AttributeError(name)


__all__ = (
    "CONTRACT_SCHEMA_VERSION",
    "MODEL_CAPABILITY_PROTOCOL_VERSION",
    "ArtifactLayout",
    "ConceptBatch",
    "DataManifest",
    "DataSplitSummary",
    "ForecastDistribution",
    "ForecastModelAdapter",
    "ForecastPredictor",
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
