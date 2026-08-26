from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import torch
from torch import Tensor, nn

from tarca.contracts import (
    ForecastDistribution,
    InterventionKind,
    InterventionSite,
    InterventionSpec,
    WindowBatch,
    validate_forecast_distribution,
    validate_intervention_site,
    validate_intervention_spec,
    validate_window_batch,
)
from tarca.stage1b.modeling.itransformer import OfficialITransformerPredictor
from tarca.stage1b.modeling.patchtst import OfficialPatchTSTPredictor


@dataclass(frozen=True, slots=True)
class NormalizationContext:
    center: Tensor
    scale: Tensor


def _normalize_history(histories: Tensor) -> tuple[Tensor, NormalizationContext]:
    center = histories.mean(dim=1, keepdim=True).detach()
    variance = torch.var(histories, dim=1, keepdim=True, unbiased=False)
    scale = torch.sqrt(variance + 1e-5).detach()
    return (histories - center) / scale, NormalizationContext(center=center, scale=scale)


class OperableNeuralPredictor(nn.Module, ABC):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        input_dimension: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(history_length, horizon, input_dimension, d_model, n_layers, n_heads, d_ff) <= 0:
            raise ValueError("neural dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.history_length = history_length
        self.horizon = horizon
        self.input_dimension = input_dimension
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self._frozen = False

    @property
    @abstractmethod
    def adapter_name(self) -> str: ...

    @abstractmethod
    def _tokenize(self, normalized_histories: Tensor) -> Tensor: ...

    @abstractmethod
    def _apply_encoder_layer(self, tokens: Tensor, layer: nn.Module) -> Tensor: ...

    @abstractmethod
    def _decode(
        self, representation: Tensor, context: NormalizationContext
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def _site(self, site_name: str, layer: int | None) -> InterventionSite: ...

    def _reset_adapter_parameters(self) -> None:
        """Reset non-module adapter parameters after module resets."""

    def list_intervention_sites(self) -> tuple[InterventionSite, ...]:
        sites = (
            self._site("encoder.input", None),
            *(self._site(f"encoder.layer.{index}", index) for index in range(self.n_layers)),
            self._site("encoder.representation", None),
        )
        return tuple(validate_intervention_site(site) for site in sites)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def model_hash(self) -> str:
        digest = hashlib.sha256(self.adapter_name.encode())
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def reset_for_training(self, seed: int) -> None:
        if self._frozen:
            raise RuntimeError("frozen neural predictor cannot be reset")
        torch.manual_seed(seed)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()  # type: ignore[no-untyped-call]
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        self._reset_adapter_parameters()

    def freeze(self) -> Self:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._frozen = True
        return self

    def _forward_with_activations(
        self,
        histories: Tensor,
        override_site: str | None = None,
        override_value: Tensor | None = None,
    ) -> tuple[ForecastDistribution, dict[str, Tensor]]:
        normalized, context = _normalize_history(histories)
        tokens = self._tokenize(normalized)
        activations: dict[str, Tensor] = {"encoder.input": tokens}
        if override_site == "encoder.input":
            if override_value is None:
                raise ValueError("intervention override value is missing")
            tokens = override_value
            activations["encoder.input"] = tokens
        for index, layer in enumerate(self.layers):
            tokens = self._apply_encoder_layer(tokens, layer)
            site_name = f"encoder.layer.{index}"
            if override_site == site_name:
                if override_value is None:
                    raise ValueError("intervention override value is missing")
                tokens = override_value
            activations[site_name] = tokens
        representation = self.final_norm(tokens)
        if override_site == "encoder.representation":
            if override_value is None:
                raise ValueError("intervention override value is missing")
            representation = override_value
        activations["encoder.representation"] = representation
        mean, scale = self._decode(representation, context)
        distribution = ForecastDistribution(
            mean=mean,
            scale=scale.clamp_min(1e-4),
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=None,
            target_names=tuple(f"x{index}" for index in range(self.input_dimension)),
        )
        return distribution, activations

    def forward_distribution(self, histories: Tensor) -> ForecastDistribution:
        if histories.ndim != 3 or histories.shape[1:] != (
            self.history_length,
            self.input_dimension,
        ):
            raise ValueError("neural histories do not match the configured shape")
        distribution, _ = self._forward_with_activations(histories)
        return distribution

    def predict_distribution(self, batch: WindowBatch) -> ForecastDistribution:
        validate_window_batch(batch)
        distribution = self.forward_distribution(batch.x)
        return validate_forecast_distribution(
            ForecastDistribution(
                mean=distribution.mean,
                scale=distribution.scale,
                quantiles=distribution.quantiles,
                logits=None,
                samples=None,
                window_id=batch.window_id,
                target_names=batch.target_names,
            )
        )

    def capture(
        self,
        batch: WindowBatch,
        sites: tuple[InterventionSite, ...],
    ) -> MappingProxyType[str, Tensor]:
        validate_window_batch(batch)
        approved = {site.site_name: site for site in self.list_intervention_sites()}
        if any(
            site.site_name not in approved or approved[site.site_name] != site for site in sites
        ):
            raise ValueError("capture requested an undeclared intervention site")
        with torch.no_grad():
            _, activations = self._forward_with_activations(batch.x)
        return MappingProxyType(
            {site.site_name: activations[site.site_name].detach().clone() for site in sites}
        )

    def intervene(
        self,
        base: WindowBatch,
        source: WindowBatch,
        spec: InterventionSpec,
    ) -> ForecastDistribution:
        if not self._frozen:
            raise RuntimeError("intervention requires a frozen neural predictor")
        validate_window_batch(base)
        validate_window_batch(source)
        validate_intervention_spec(spec, orthogonality_tolerance=1e-5)
        sites = {site.site_name: site for site in self.list_intervention_sites()}
        try:
            site = sites[spec.site_name]
        except KeyError as exc:
            raise ValueError("intervention requested an undeclared site") from exc
        if site.layer != spec.layer:
            raise ValueError("intervention layer does not match the declared site")
        if base.x.shape != source.x.shape:
            raise ValueError("base and source batches must have equal tensor shapes")
        with torch.no_grad():
            _, base_activations = self._forward_with_activations(base.x)
            _, source_activations = self._forward_with_activations(source.x)
            replacement = self._replacement(
                base_activations[site.site_name],
                source_activations[site.site_name],
                site,
                spec,
            )
            distribution, _ = self._forward_with_activations(
                base.x, override_site=site.site_name, override_value=replacement
            )
        return validate_forecast_distribution(
            ForecastDistribution(
                mean=distribution.mean,
                scale=distribution.scale,
                quantiles=distribution.quantiles,
                logits=None,
                samples=None,
                window_id=base.window_id,
                target_names=base.target_names,
            )
        )

    @staticmethod
    def _replacement(
        base: Tensor,
        source: Tensor,
        site: InterventionSite,
        spec: InterventionSpec,
    ) -> Tensor:
        replacement = base.clone()
        selection: list[slice | int] = [slice(None)] * base.ndim
        if spec.variable_index is not None:
            if site.variable_axis is None:
                raise ValueError("site has no variable axis")
            selection[site.variable_axis] = spec.variable_index
        if spec.patch_index is not None:
            if site.patch_axis is None:
                raise ValueError("site has no patch axis")
            selection[site.patch_axis] = spec.patch_index
        selected = tuple(selection)
        if spec.intervention_kind is InterventionKind.FULL_SWAP:
            replacement[selected] = source[selected]
            return replacement
        basis = spec.subspace_basis
        if basis is None or basis.shape[0] != base.shape[site.feature_axis]:
            raise ValueError("subspace basis does not match the site feature width")
        delta = source[selected] - base[selected]
        replacement[selected] = base[selected] + delta @ basis @ basis.transpose(0, 1)
        return replacement


PatchTSTReference = OfficialPatchTSTPredictor
ITransformerReference = OfficialITransformerPredictor
