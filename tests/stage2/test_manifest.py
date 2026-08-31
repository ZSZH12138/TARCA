from __future__ import annotations

import pytest

from tarca.stage2.manifest import Stage2CompilationInputs, compile_stage2_manifest
from tarca.stage2.selection import ModelSelection


def compilation_inputs() -> Stage2CompilationInputs:
    seeds = (1797287582, 883082243, 1933050005)
    return Stage2CompilationInputs(
        scientific_config_sha256="1" * 64,
        stage1b_manifest_sha256="2" * 64,
        e01_receipt_sha256="3" * 64,
        source_receipt_sha256="4" * 64,
        normalizer_sha256="5" * 64,
        data_manifest_sha256="6" * 64,
        precision_receipt_sha256="7" * 64,
        predictor_sha256=(
            ("LAST_VALUE", "8" * 64),
            ("SEASONAL_NAIVE", "9" * 64),
            ("VAR", "a" * 64),
            ("DLINEAR", "b" * 64),
            ("PATCHTST", "c" * 64),
            ("ITRANSFORMER", "d" * 64),
        ),
        neural_checkpoint_sha256=tuple(
            (model_id, seed, character * 64)
            for model_id, character in (("PATCHTST", "e"), ("ITRANSFORMER", "f"))
            for seed in seeds
        ),
        strongest_linear=ModelSelection(
            model_id="DLINEAR",
            seed=None,
            validation_score=0.29,
            validation_artifact_refs=("VALIDATION/linear.json",),
        ),
        primary_itransformer=ModelSelection(
            model_id="ITRANSFORMER",
            seed=seeds[0],
            validation_score=0.21,
            validation_artifact_refs=("VALIDATION/itransformer.json",),
        ),
        runtime_failure_ids=(),
        formal_access_event_count=0,
        gpu_order=(0, 1),
    )


def test_science_hash_ignores_worker_placement() -> None:
    first = compile_stage2_manifest(compilation_inputs().with_gpu_order((0, 1)))
    second = compile_stage2_manifest(compilation_inputs().with_gpu_order((1, 0)))

    assert first.scientific_sha256 == second.scientific_sha256
    assert first.runtime_sha256 != second.runtime_sha256


def test_manifest_requires_exact_six_predictors_and_six_neural_checkpoints() -> None:
    inputs = compilation_inputs()
    with pytest.raises(ValueError, match="six predictor"):
        Stage2CompilationInputs(
            **{
                **inputs.__dict__,
                "predictor_sha256": inputs.predictor_sha256[:-1],
            }
        )


def test_manifest_rejects_any_formal_access_event() -> None:
    inputs = compilation_inputs()
    with pytest.raises(ValueError, match="formal access"):
        Stage2CompilationInputs(
            **{
                **inputs.__dict__,
                "formal_access_event_count": 1,
            }
        )

