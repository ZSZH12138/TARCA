"""Minimal typed contracts for future modules; no scientific implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .adapters import ForecastPredictor
from .common import TaskResult, TaskSpec, _require_text
from .forecast import ForecastDistribution
from .interventions import InterventionSpec


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.version, "version")


@dataclass(frozen=True, slots=True)
class SplitSpec:
    train_fraction: float
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(fraction <= 0.0 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("split fractions must be positive and sum to one")


@dataclass(frozen=True, slots=True)
class AccessScope:
    sealed: bool = False
    scope_name: str = "unsealed"

    def __post_init__(self) -> None:
        _require_text(self.scope_name, "scope_name")


@dataclass(frozen=True, slots=True)
class ConceptIntervention:
    concept_name: str
    delta: float

    def __post_init__(self) -> None:
        _require_text(self.concept_name, "concept_name")


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    passed: bool
    findings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InterventionPairSet:
    pair_ids: tuple[str, ...]
    source: str = "contract-only"

    def __post_init__(self) -> None:
        if not self.pair_ids or len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("pair_ids must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class InterventionResult:
    pair_id: str
    spec: InterventionSpec
    factual: ForecastDistribution
    intervened: ForecastDistribution


@dataclass(frozen=True, slots=True)
class EffectNormalizationSpec:
    train_only: bool = True


class LocalizationStage(StrEnum):
    COARSE_LAYER = "COARSE_LAYER"
    TIME_PATCH = "TIME_PATCH"
    VARIABLE = "VARIABLE"
    SUBSPACE = "SUBSPACE"
    DAS_REFINEMENT = "DAS_REFINEMENT"
    HELDOUT_EVAL = "HELDOUT_EVAL"


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    stage: LocalizationStage
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    environment_id: str
    definition_hash: str

    def __post_init__(self) -> None:
        _require_text(self.environment_id, "environment_id")
        _require_text(self.definition_hash, "definition_hash")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    protocol_id: str
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        _require_text(self.experiment_id, "experiment_id")
        _require_text(self.protocol_id, "protocol_id")
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("tasks must be non-empty and task IDs must be unique")


@dataclass(frozen=True, slots=True)
class TaskManifest:
    manifest_id: str
    tasks: tuple[TaskSpec, ...]
    completed_task_policy: str = "NEVER_RERUN"

    def __post_init__(self) -> None:
        _require_text(self.manifest_id, "manifest_id")
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("tasks must be non-empty and task IDs must be unique")
        if self.completed_task_policy != "NEVER_RERUN":
            raise ValueError("completed_task_policy must be NEVER_RERUN")


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    experiment_id: str
    results: tuple[TaskResult, ...]

    def __post_init__(self) -> None:
        _require_text(self.experiment_id, "experiment_id")


@dataclass(frozen=True, slots=True)
class ModelResolution:
    model_id: str
    predictor: ForecastPredictor

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model_id")


@dataclass(frozen=True, slots=True)
class RobustnessSpec:
    train_environment: EnvironmentSpec
    validation_environment: EnvironmentSpec
    test_environment: EnvironmentSpec
