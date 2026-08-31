from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from tarca.contracts import canonical_json_hash
from tarca.stage2.selection import ModelSelection

_PREDICTOR_IDS = (
    "LAST_VALUE",
    "SEASONAL_NAIVE",
    "VAR",
    "DLINEAR",
    "PATCHTST",
    "ITRANSFORMER",
)
_NEURAL_IDS = ("PATCHTST", "ITRANSFORMER")


def _is_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class Stage2CompilationInputs:
    scientific_config_sha256: str
    stage1b_manifest_sha256: str
    e01_receipt_sha256: str
    source_receipt_sha256: str
    normalizer_sha256: str
    data_manifest_sha256: str
    precision_receipt_sha256: str
    predictor_sha256: tuple[tuple[str, str], ...]
    neural_checkpoint_sha256: tuple[tuple[str, int, str], ...]
    strongest_linear: ModelSelection
    primary_itransformer: ModelSelection
    runtime_failure_ids: tuple[str, ...]
    formal_access_event_count: int
    gpu_order: tuple[int, int]

    def __post_init__(self) -> None:
        hashes = (
            self.scientific_config_sha256,
            self.stage1b_manifest_sha256,
            self.e01_receipt_sha256,
            self.source_receipt_sha256,
            self.normalizer_sha256,
            self.data_manifest_sha256,
            self.precision_receipt_sha256,
            *(digest for _, digest in self.predictor_sha256),
            *(digest for _, _, digest in self.neural_checkpoint_sha256),
        )
        if any(not _is_sha256(digest) for digest in hashes):
            raise ValueError("Stage 2 compilation identities must be lowercase SHA-256 values")
        if tuple(model_id for model_id, _ in self.predictor_sha256) != _PREDICTOR_IDS:
            raise ValueError("Stage 2 manifest requires the exact six predictor identities")
        checkpoint_groups = {
            model_id: tuple(
                seed
                for candidate, seed, _ in self.neural_checkpoint_sha256
                if candidate == model_id
            )
            for model_id in _NEURAL_IDS
        }
        if (
            len(self.neural_checkpoint_sha256) != 6
            or set(model_id for model_id, _, _ in self.neural_checkpoint_sha256)
            != set(_NEURAL_IDS)
            or any(
                len(seeds) != 3 or len(seeds) != len(set(seeds))
                for seeds in checkpoint_groups.values()
            )
        ):
            raise ValueError("Stage 2 manifest requires three checkpoints for each neural model")
        if self.strongest_linear.model_id not in {"VAR", "DLINEAR"}:
            raise ValueError("strongest linear selection must be VAR or DLINEAR")
        if (
            self.primary_itransformer.model_id != "ITRANSFORMER"
            or self.primary_itransformer.seed not in checkpoint_groups["ITRANSFORMER"]
        ):
            raise ValueError("primary iTransformer selection must reference a frozen checkpoint")
        if self.formal_access_event_count != 0:
            raise ValueError("Stage 2 compilation refuses any formal access event")
        if len(self.runtime_failure_ids) != len(set(self.runtime_failure_ids)) or any(
            not item.strip() for item in self.runtime_failure_ids
        ):
            raise ValueError("runtime failure IDs must be unique and nonblank")
        if tuple(sorted(self.gpu_order)) != (0, 1):
            raise ValueError("Stage 2 GPU order must contain devices zero and one")

    def with_gpu_order(self, gpu_order: tuple[int, int]) -> Stage2CompilationInputs:
        return replace(self, gpu_order=gpu_order)

    def with_runtime_failures(
        self, failures: tuple[str, ...]
    ) -> Stage2CompilationInputs:
        return replace(self, runtime_failure_ids=failures)


@dataclass(frozen=True, slots=True)
class Stage2Manifest:
    schema_version: str
    scientific_config_sha256: str
    stage1b_manifest_sha256: str
    e01_receipt_sha256: str
    source_receipt_sha256: str
    normalizer_sha256: str
    data_manifest_sha256: str
    precision_receipt_sha256: str
    predictor_sha256: tuple[tuple[str, str], ...]
    neural_checkpoint_sha256: tuple[tuple[str, int, str], ...]
    strongest_linear: ModelSelection
    primary_itransformer: ModelSelection
    runtime_failure_ids: tuple[str, ...]
    formal_access_event_count: int
    gpu_order: tuple[int, int]
    scientific_sha256: str
    runtime_sha256: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _scientific_payload(inputs: Stage2CompilationInputs) -> dict[str, Any]:
    payload = asdict(inputs)
    payload.pop("gpu_order")
    payload.pop("runtime_failure_ids")
    return payload


def compile_stage2_manifest(inputs: Stage2CompilationInputs) -> Stage2Manifest:
    scientific_sha256 = canonical_json_hash(_scientific_payload(inputs))
    runtime_sha256 = canonical_json_hash(
        {
            "scientific_sha256": scientific_sha256,
            "gpu_order": inputs.gpu_order,
            "runtime_failure_ids": inputs.runtime_failure_ids,
        }
    )
    return Stage2Manifest(
        schema_version="tarca-stage2-manifest-v1",
        scientific_config_sha256=inputs.scientific_config_sha256,
        stage1b_manifest_sha256=inputs.stage1b_manifest_sha256,
        e01_receipt_sha256=inputs.e01_receipt_sha256,
        source_receipt_sha256=inputs.source_receipt_sha256,
        normalizer_sha256=inputs.normalizer_sha256,
        data_manifest_sha256=inputs.data_manifest_sha256,
        precision_receipt_sha256=inputs.precision_receipt_sha256,
        predictor_sha256=inputs.predictor_sha256,
        neural_checkpoint_sha256=inputs.neural_checkpoint_sha256,
        strongest_linear=inputs.strongest_linear,
        primary_itransformer=inputs.primary_itransformer,
        runtime_failure_ids=inputs.runtime_failure_ids,
        formal_access_event_count=inputs.formal_access_event_count,
        gpu_order=inputs.gpu_order,
        scientific_sha256=scientific_sha256,
        runtime_sha256=runtime_sha256,
    )


def stage2_manifest_from_payload(payload: object) -> Stage2Manifest:
    if not isinstance(payload, dict):
        raise ValueError("Stage 2 manifest must be a JSON object")
    value = cast(dict[str, Any], payload)
    strongest = ModelSelection(**value["strongest_linear"])
    primary = ModelSelection(**value["primary_itransformer"])
    inputs = Stage2CompilationInputs(
        scientific_config_sha256=value["scientific_config_sha256"],
        stage1b_manifest_sha256=value["stage1b_manifest_sha256"],
        e01_receipt_sha256=value["e01_receipt_sha256"],
        source_receipt_sha256=value["source_receipt_sha256"],
        normalizer_sha256=value["normalizer_sha256"],
        data_manifest_sha256=value["data_manifest_sha256"],
        precision_receipt_sha256=value["precision_receipt_sha256"],
        predictor_sha256=tuple(tuple(item) for item in value["predictor_sha256"]),
        neural_checkpoint_sha256=tuple(
            tuple(item) for item in value["neural_checkpoint_sha256"]
        ),
        strongest_linear=strongest,
        primary_itransformer=primary,
        runtime_failure_ids=tuple(value["runtime_failure_ids"]),
        formal_access_event_count=value["formal_access_event_count"],
        gpu_order=tuple(value["gpu_order"]),
    )
    compiled = compile_stage2_manifest(inputs)
    if canonical_json_hash(compiled.payload()) != canonical_json_hash(value):
        raise ValueError("Stage 2 manifest hashes or fields do not match")
    return compiled
