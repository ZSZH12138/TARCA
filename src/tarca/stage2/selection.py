from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationScore:
    model_id: str
    seed: int | None
    crps: float
    artifact_ref: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not math.isfinite(self.crps) or self.crps < 0:
            raise ValueError("validation score identity and CRPS must be valid")
        if not self.artifact_ref.startswith("VALIDATION/"):
            raise ValueError("model selection may reference only VALIDATION artifacts")


@dataclass(frozen=True, slots=True)
class ModelSelection:
    model_id: str
    seed: int | None
    validation_score: float
    validation_artifact_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not math.isfinite(self.validation_score):
            raise ValueError("model selection identity and score must be valid")
        if not self.validation_artifact_refs or any(
            not reference.startswith("VALIDATION/")
            for reference in self.validation_artifact_refs
        ):
            raise ValueError("model selection must contain only VALIDATION artifact references")


def _linear_scores(
    scores: Mapping[str, float] | tuple[ValidationScore, ...],
) -> tuple[ValidationScore, ...]:
    if isinstance(scores, Mapping):
        return tuple(
            ValidationScore(model_id, None, score, f"VALIDATION/{model_id.lower()}.json")
            for model_id, score in scores.items()
        )
    return scores


def select_strongest_linear(
    validation_scores: Mapping[str, float] | tuple[ValidationScore, ...],
) -> ModelSelection:
    scores = _linear_scores(validation_scores)
    by_id = {score.model_id: score for score in scores}
    if set(by_id) != {"VAR", "DLINEAR"} or len(scores) != 2:
        raise ValueError("linear selection requires exactly VAR and DLINEAR validation scores")
    order = {"DLINEAR": 0, "VAR": 1}
    selected = min(scores, key=lambda score: (score.crps, order[score.model_id]))
    return ModelSelection(
        model_id=selected.model_id,
        seed=None,
        validation_score=selected.crps,
        validation_artifact_refs=tuple(score.artifact_ref for score in scores),
    )


def select_primary_initialization(
    model_id: str,
    validation_scores: Mapping[int, float] | tuple[ValidationScore, ...],
    *,
    seed_order: tuple[int, ...],
) -> ModelSelection:
    if not seed_order or len(seed_order) != len(set(seed_order)):
        raise ValueError("initialization seed order must be nonempty and unique")
    if isinstance(validation_scores, Mapping):
        scores = tuple(
            ValidationScore(
                model_id,
                seed,
                score,
                f"VALIDATION/{model_id.lower()}-{seed}.json",
            )
            for seed, score in validation_scores.items()
        )
    else:
        scores = validation_scores
    if (
        len(scores) != len(seed_order)
        or {score.seed for score in scores} != set(seed_order)
        or any(score.model_id != model_id for score in scores)
    ):
        raise ValueError("initialization selection scores must match the frozen seed order")
    order = {seed: index for index, seed in enumerate(seed_order)}
    selected = min(
        scores,
        key=lambda score: (score.crps, order[score.seed]),  # type: ignore[index]
    )
    return ModelSelection(
        model_id=model_id,
        seed=selected.seed,
        validation_score=selected.crps,
        validation_artifact_refs=tuple(score.artifact_ref for score in scores),
    )

