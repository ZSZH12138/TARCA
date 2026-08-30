from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tarca.e01.config import E01Condition


def _finite_matrix(value: Tensor, label: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{label} must be a rank-2 tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be a finite floating tensor")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class EffectSamples:
    values: Tensor
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _finite_matrix(self.values, "effect samples")
        if values.shape[0] != len(self.group_ids) or not self.group_ids:
            raise ValueError("effect group IDs must match a nonempty sample axis")
        if len(self.group_ids) != len(set(self.group_ids)):
            raise ValueError("effect sample group IDs must be unique")
        object.__setattr__(self, "values", values)


def paired_difference(factual: Tensor, counterfactual: Tensor) -> Tensor:
    factual_value = _finite_matrix(factual, "factual rollout")
    counterfactual_value = _finite_matrix(counterfactual, "counterfactual rollout")
    if factual_value.shape != counterfactual_value.shape:
        raise ValueError("paired rollouts must share a shape")
    return counterfactual_value - factual_value


def analytic_delayed_effect(
    *,
    horizon: int,
    true_lag: int,
    delta: float,
    decay: float = 0.75,
) -> Tensor:
    if type(horizon) is not int or type(true_lag) is not int or horizon <= 0 or true_lag <= 0:
        raise ValueError("horizon and true lag must be positive integers")
    if true_lag > horizon:
        raise ValueError("true lag cannot exceed the evaluation horizon")
    if not math.isfinite(delta) or delta == 0.0:
        raise ValueError("intervention delta must be finite and nonzero")
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("delayed-control decay must be finite and inside (0, 1)")
    effect = torch.zeros(horizon, dtype=torch.float64)
    steps = torch.arange(horizon - true_lag + 1, dtype=torch.float64)
    effect[true_lag - 1 :] = float(delta) * torch.pow(float(decay), steps)
    return effect


def _condition_curve(
    condition: E01Condition,
    *,
    horizon: int,
    true_lag: int,
    wrong_lag: int,
    delta: float,
) -> Tensor:
    if condition == "IDENTITY":
        return torch.zeros(horizon, dtype=torch.float64)
    if condition == "CORRECT_SCM":
        return analytic_delayed_effect(
            horizon=horizon,
            true_lag=true_lag,
            delta=delta,
        )
    if condition == "WRONG_SCM":
        return analytic_delayed_effect(
            horizon=horizon,
            true_lag=true_lag,
            delta=delta * 0.55,
            decay=0.25,
        )
    if condition == "WRONG_LAG":
        return analytic_delayed_effect(
            horizon=horizon,
            true_lag=wrong_lag,
            delta=delta,
        )
    raise ValueError("random concept does not have a fixed effect curve")


def simulate_delayed_effects(
    *,
    seed: int,
    sample_count: int,
    horizon: int,
    true_lag: int,
    wrong_lag: int,
    delta: float,
    condition: E01Condition,
    device: str,
    batch_size: int,
) -> EffectSamples:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample count must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or an explicit CUDA device")
    if true_lag == wrong_lag:
        raise ValueError("wrong lag must differ from true lag")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gains = 1.0 + 0.35 * torch.randn(sample_count, 1, generator=generator, dtype=torch.float64)
    if condition == "RANDOM_CONCEPT":
        source = torch.randn(
            sample_count,
            horizon,
            generator=generator,
            dtype=torch.float64,
        )
        source = source / source.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        unbatched = source * (abs(float(delta)) * 0.5)
    else:
        unbatched = gains * _condition_curve(
            condition,
            horizon=horizon,
            true_lag=true_lag,
            wrong_lag=wrong_lag,
            delta=delta,
        ).reshape(1, horizon)

    chunks: list[Tensor] = []
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        chunks.append(unbatched[start:stop].to(device=device).mul(1.0).to(device="cpu"))
    values = torch.cat(chunks, dim=0)
    return EffectSamples(
        values=values,
        group_ids=tuple(f"base-{index:06d}" for index in range(sample_count)),
    )
