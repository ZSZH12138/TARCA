from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tarca.contracts import ArtifactRef, ForecastDistribution, sha256_file
from tarca.e02.server_probe import (
    _probe_cpu_path,
    _probe_neural_checkpoint_worker,
    _verified_probe_inputs,
    estimate_e02_critical_path_seconds,
    run_e02_server_probe,
)

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[2]


def test_e02_eta_uses_two_gpu_waves_and_cpu_backfill() -> None:
    estimated = estimate_e02_critical_path_seconds(
        formal_trajectory_count=120,
        windows_per_trajectory=425,
        generation_trajectories_per_second=2.0,
        neural_windows_per_second=(1000.0, 800.0, 900.0),
        neural_startup_seconds=(10.0, 12.0, 11.0),
        linear_windows_per_second=500.0,
        scoring_windows_per_second=2000.0,
        bootstrap_seconds=30.0,
        score_parallel_tasks=4,
        fixed_overhead_seconds=60.0,
        safety_multiplier=1.35,
    )

    # generation=60; neural=max(61,75.75)+67.667; linear=102 (backfilled);
    # four scores fit one 25.5-second CPU wave; bootstrap=30.
    assert estimated == pytest.approx(409.5375, abs=0.001)


def test_e02_probe_schedules_two_checkpoints_first_then_third_on_second_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[tuple[int, int]] = []
    executor_sizes: list[int] = []

    class _Future:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class _Pool:
        def __init__(self, max_workers: int, **kwargs: object) -> None:
            del kwargs
            executor_sizes.append(max_workers)

        def __enter__(self) -> _Pool:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def submit(self, function: object, *args: object) -> _Future:
            del function
            checkpoint_index = int(args[-2])
            gpu_id = int(args[-1])
            submitted.append((checkpoint_index, gpu_id))
            return _Future(
                {
                    "checkpoint_index": checkpoint_index,
                    "gpu_id": gpu_id,
                    "sample_window_count": 1024,
                    "windows_per_second": 1000.0 - checkpoint_index * 100.0,
                    "startup_seconds": 10.0 + checkpoint_index,
                    "forecast_finite": True,
                    "positive_scales": True,
                    "checkpoint_hash_unchanged": True,
                }
            )

    monkeypatch.setattr("tarca.e02.server_probe.ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(
        "tarca.e02.server_probe._verified_probe_inputs",
        lambda root: {
            "development_data_ref": {"artifact_id": "development"},
            "itransformer_checkpoints": tuple(
                {"seed": seed, "ref": {"artifact_id": f"checkpoint-{index}"}}
                for index, seed in enumerate((1797287582, 883082243, 1933050005))
            ),
        },
    )
    monkeypatch.setattr(
        "tarca.e02.server_probe._probe_cpu_path",
        lambda *args: {
            "generation_trajectories_per_second": 2.0,
            "linear_windows_per_second": 500.0,
            "scoring_windows_per_second": 2000.0,
            "bootstrap_seconds": 30.0,
        },
    )
    gate: list[tuple[float, float, float]] = []
    monkeypatch.setattr(
        "tarca.e02.server_probe.stage2_reset_time_gate",
        lambda **values: gate.append(
            (
                float(values["estimated_remaining_seconds"]),
                float(values["remaining_rental_hours"]),
                float(values["margin_hours"]),
            )
        ),
    )

    result = run_e02_server_probe(
        ROOT,
        ROOT / "configs/e02/e02_v1.yaml",
        tmp_path / "runtime",
        remaining_rental_hours=24.0,
    )

    assert executor_sizes == [2, 1]
    assert submitted == [(0, 0), (1, 1), (2, 0)]
    assert result["probe_contract"] == "e02-v1-three-frozen-checkpoints-two-gpu-waves"
    assert result["formal_trajectory_count"] == 120
    assert result["formal_window_count"] == 51_000
    assert result["formal_tasks_executed"] == 0
    assert result["eta_gate_passed"] is True
    assert gate[0][1:] == (24.0, 1.0)


def test_e02_cpu_probe_uses_the_frozen_development_artifact_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tarca.stage2.data._read_formal_storage",
        lambda: (_ for _ in ()).throw(AssertionError("formal storage opened")),
    )

    class _World:
        config = SimpleNamespace(
            regimes=(
                SimpleNamespace(
                    split_role=SimpleNamespace(value="SEEN"), regime_id="seen-probe"
                ),
            )
        )

        def simulate(self, request: object) -> SimpleNamespace:
            del request
            return SimpleNamespace(values=torch.ones((512, 8), dtype=torch.float64))

    class _Predictor:
        adapter_name = "frozen-probe-baseline"
        model_hash = "a" * 64
        is_frozen = True

        def predict_distribution(self, batch: SimpleNamespace) -> ForecastDistribution:
            return ForecastDistribution(
                mean=torch.zeros_like(batch.y),
                scale=torch.ones_like(batch.y),
                quantiles={},
                logits=None,
                samples=None,
                window_id=None,
                target_names=tuple(f"x{index}" for index in range(8)),
            )

    batch = SimpleNamespace(
        x=torch.zeros((425, 64, 8), dtype=torch.float32),
        y=torch.zeros((425, 24, 8), dtype=torch.float32),
    )
    manifest = json.loads(
        (ROOT / "artifacts/stage2/frozen/v1/stage2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    strongest = manifest["strongest_linear"]["model_id"]
    ref = ArtifactRef(
        artifact_id="probe",
        artifact_type="STAGE2_DEVELOPMENT_DATA",
        content_hash="a" * 64,
        schema_version="1.0.0",
        relative_path="artifacts/stage2/runtime/store/probe.bin",
    ).model_dump(mode="json")
    monkeypatch.setattr("tarca.e02.server_probe.build_world", lambda config: _World())
    monkeypatch.setattr("tarca.e02.server_probe._load_torch", lambda *args: {})
    monkeypatch.setattr("tarca.e02.server_probe._bundle_from_payload", lambda value: object())
    monkeypatch.setattr("tarca.e02.server_probe._window_batch", lambda bundle: batch)
    monkeypatch.setattr(
        "tarca.e02.server_probe._baseline_from_payload",
        lambda *args: _Predictor(),
    )

    observation = _probe_cpu_path(
        ROOT,
        {
            "development_data_ref": ref,
            "baseline_refs": {strongest: ref},
            "stage2_manifest": manifest,
        },
    )

    assert set(observation) == {
        "generation_trajectories_per_second",
        "linear_windows_per_second",
        "scoring_windows_per_second",
        "bootstrap_seconds",
    }
    assert all(math.isfinite(value) and value > 0 for value in observation.values())


def test_e02_verified_probe_inputs_requires_and_propagates_frozen_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "artifacts/stage2/frozen/v1/stage2_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    frozen_manifest = SimpleNamespace(payload=lambda: {"frozen": True})
    monkeypatch.setattr(
        "tarca.e02.server_probe.stage2_manifest_from_payload",
        lambda payload: frozen_manifest,
    )
    monkeypatch.setattr(
        "tarca.e02.server_probe._stage2_completed", lambda root: {"task": "artifact"}
    )
    monkeypatch.setattr(
        "tarca.e02.server_probe._verify_stage2_artifacts",
        lambda root, manifest, completed: {
            "development_data_ref": {"artifact_id": "development"}
        },
    )

    result = _verified_probe_inputs(tmp_path)

    assert result == {
        "development_data_ref": {"artifact_id": "development"},
        "stage2_manifest": {"frozen": True},
    }


def test_e02_neural_probe_worker_checks_probabilities_and_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "artifacts/stage2/runtime/store/checkpoint.bin"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"frozen-checkpoint")
    data_ref = ArtifactRef(
        artifact_id="development",
        artifact_type="STAGE2_DEVELOPMENT_DATA",
        content_hash="b" * 64,
        schema_version="1.0.0",
        relative_path="artifacts/stage2/runtime/store/development.bin",
    )
    checkpoint_ref = ArtifactRef(
        artifact_id="checkpoint",
        artifact_type="VALIDATED_STAGE2_CHECKPOINT",
        content_hash=sha256_file(checkpoint_path),
        schema_version="1.0.0",
        relative_path="artifacts/stage2/runtime/store/checkpoint.bin",
    )
    validation_x = torch.zeros((512, 64, 8), dtype=torch.float32)
    validation_y = torch.zeros((512, 24, 8), dtype=torch.float32)

    class _Neural:
        model_hash = "c" * 64

        def load_state_dict(self, value: object, strict: bool) -> None:
            assert value == {} and strict is True

        def to(self, device: object) -> _Neural:
            del device
            return self

        def freeze(self) -> None:
            return None

        def forward_distribution(self, values: torch.Tensor) -> ForecastDistribution:
            shape = (values.shape[0], 24, 8)
            return ForecastDistribution(
                mean=torch.zeros(shape, dtype=torch.float32),
                scale=torch.ones(shape, dtype=torch.float32),
                quantiles={},
                logits=None,
                samples=None,
                window_id=None,
                target_names=tuple(f"x{index}" for index in range(8)),
            )

    checkpoint = {
        "state_dict": {},
        "model_sha256": "c" * 64,
        "seed": 1797287582,
        "precision": "FP32",
    }
    monkeypatch.setattr(
        "tarca.e02.server_probe._load_torch",
        lambda root, ref: checkpoint if ref.artifact_id == "checkpoint" else {},
    )
    monkeypatch.setattr("tarca.e02.server_probe._bundle_from_payload", lambda value: object())
    monkeypatch.setattr(
        "tarca.e02.server_probe.stack_partition",
        lambda bundle, partition: (validation_x, validation_y, ("window",) * 512),
    )
    monkeypatch.setattr("tarca.e02.server_probe.load_stage2_config", lambda path: object())
    monkeypatch.setattr("tarca.e02.server_probe._new_neural", lambda *args: _Neural())
    cpu_device = torch.device("cpu")
    monkeypatch.setattr("tarca.e02.server_probe.torch.device", lambda value: cpu_device)
    monkeypatch.setattr("tarca.e02.server_probe.torch.cuda.set_device", lambda value: None)
    monkeypatch.setattr("tarca.e02.server_probe.torch.cuda.synchronize", lambda value: None)

    result = _probe_neural_checkpoint_worker(
        str(tmp_path),
        data_ref.model_dump(mode="json"),
        checkpoint_ref.model_dump(mode="json"),
        0,
        0,
    )

    assert result["sample_window_count"] == 512
    assert result["forecast_finite"] is True
    assert result["positive_scales"] is True
    assert result["checkpoint_hash_unchanged"] is True


@pytest.mark.parametrize(
    "changes",
    (
        {"generation_trajectories_per_second": 0.0},
        {"neural_windows_per_second": (1000.0, float("nan"), 900.0)},
        {"score_parallel_tasks": 0},
        {"safety_multiplier": 0.99},
    ),
)
def test_e02_eta_rejects_nonconservative_or_invalid_observations(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "formal_trajectory_count": 120,
        "windows_per_trajectory": 425,
        "generation_trajectories_per_second": 2.0,
        "neural_windows_per_second": (1000.0, 800.0, 900.0),
        "neural_startup_seconds": (10.0, 12.0, 11.0),
        "linear_windows_per_second": 500.0,
        "scoring_windows_per_second": 2000.0,
        "bootstrap_seconds": 30.0,
        "score_parallel_tasks": 4,
        "fixed_overhead_seconds": 60.0,
        "safety_multiplier": 1.35,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        estimate_e02_critical_path_seconds(**values)  # type: ignore[arg-type]
