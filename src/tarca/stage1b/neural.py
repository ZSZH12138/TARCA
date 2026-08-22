from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from tarca.contracts import (
    ForecastDistribution,
    InterventionKind,
    InterventionSite,
    InterventionSpec,
    WindowBatch,
    validate_forecast_distribution,
    validate_intervention_spec,
    validate_window_batch,
)


class OperableNeuralPredictor(nn.Module, ABC):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        input_dimension: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
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
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.log_scale = nn.Parameter(torch.full((horizon, input_dimension), -1.0))
        self._frozen = False

    @property
    @abstractmethod
    def adapter_name(self) -> str: ...

    @abstractmethod
    def _tokenize(self, histories: Tensor) -> Tensor: ...

    @abstractmethod
    def _decode(self, representation: Tensor) -> Tensor: ...

    @abstractmethod
    def _site(self, site_name: str, layer: int | None) -> InterventionSite: ...

    def list_intervention_sites(self) -> tuple[InterventionSite, ...]:
        return (
            self._site("encoder.input", None),
            *(self._site(f"encoder.layer.{index}", index) for index in range(self.n_layers)),
            self._site("encoder.representation", None),
        )

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
                module._reset_parameters()
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        nn.init.constant_(self.log_scale, -1.0)

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
        tokens = self._tokenize(histories)
        activations: dict[str, Tensor] = {"encoder.input": tokens}
        if override_site == "encoder.input":
            if override_value is None:
                raise ValueError("intervention override value is missing")
            tokens = override_value
            activations["encoder.input"] = tokens
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens)
            site_name = f"encoder.layer.{index}"
            activations[site_name] = tokens
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
        mean = self._decode(representation)
        scale = functional.softplus(self.log_scale).clamp_min(1e-4)
        scale = scale.unsqueeze(0).expand(histories.shape[0], -1, -1)
        distribution = ForecastDistribution(
            mean=mean,
            scale=scale,
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
            site.site_name not in approved or approved[site.site_name] != site
            for site in sites
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
                base.x,
                override_site=site.site_name,
                override_value=replacement,
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
        projected = delta @ basis @ basis.transpose(0, 1)
        replacement[selected] = base[selected] + projected
        return replacement


class SmallPatchTST(OperableNeuralPredictor):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        input_dimension: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        patch_size: int = 4,
    ) -> None:
        super().__init__(
            history_length,
            horizon,
            input_dimension,
            d_model,
            n_layers,
            n_heads,
            dropout,
        )
        if self.history_length % patch_size:
            raise ValueError("PatchTST history length must be divisible by patch_size")
        self.patch_size = patch_size
        self.patch_count = self.history_length // patch_size
        self.input_projection = nn.Linear(patch_size * self.input_dimension, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, self.d_model))
        self.output_head = nn.Linear(self.d_model, self.horizon * self.input_dimension)

    @property
    def adapter_name(self) -> str:
        return "SmallPatchTST"

    def _tokenize(self, histories: Tensor) -> Tensor:
        patches = histories.reshape(
            histories.shape[0], self.patch_count, self.patch_size * self.input_dimension
        )
        return cast(Tensor, self.input_projection(patches) + self.position)

    def _decode(self, representation: Tensor) -> Tensor:
        pooled = representation.mean(dim=1)
        decoded = self.output_head(pooled).reshape(
            representation.shape[0], self.horizon, self.input_dimension
        )
        return cast(Tensor, decoded)

    def _site(self, site_name: str, layer: int | None) -> InterventionSite:
        return InterventionSite(
            site_name=site_name,
            layer=layer,
            tensor_rank=3,
            batch_axis=0,
            variable_axis=None,
            patch_axis=1,
            feature_axis=2,
            shape_template=(None, self.patch_count, self.d_model),
        )


class SmallITransformer(OperableNeuralPredictor):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        input_dimension: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(
            history_length,
            horizon,
            input_dimension,
            d_model,
            n_layers,
            n_heads,
            dropout,
        )
        self.input_projection = nn.Linear(self.history_length, self.d_model)
        self.variable_embedding = nn.Parameter(
            torch.zeros(1, self.input_dimension, self.d_model)
        )
        self.output_head = nn.Linear(self.d_model, self.horizon)

    @property
    def adapter_name(self) -> str:
        return "SmallITransformer"

    def _tokenize(self, histories: Tensor) -> Tensor:
        tokens = self.input_projection(histories.transpose(1, 2)) + self.variable_embedding
        return cast(Tensor, tokens)

    def _decode(self, representation: Tensor) -> Tensor:
        return cast(Tensor, self.output_head(representation).transpose(1, 2))

    def _site(self, site_name: str, layer: int | None) -> InterventionSite:
        return InterventionSite(
            site_name=site_name,
            layer=layer,
            tensor_rank=3,
            batch_axis=0,
            variable_axis=1,
            patch_axis=None,
            feature_axis=2,
            shape_template=(None, self.input_dimension, self.d_model),
        )
