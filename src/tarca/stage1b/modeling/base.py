from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

import torch
from torch import Tensor, nn

from tarca.contracts import (
    ForecastDistribution,
    InterventionKind,
    InterventionSite,
    InterventionSpec,
    JSONValue,
    WindowBatch,
    canonical_json_bytes,
    validate_forecast_distribution,
    validate_intervention_site,
    validate_intervention_spec,
    validate_window_batch,
)
from tarca.stage1b.modeling.hooks import (
    RegisteredSite,
    installed_site_capture,
    installed_site_swap,
)


@dataclass(frozen=True, slots=True)
class ModelSourceContext:
    source_id: str
    commit: str
    source_root: Path
    receipt_sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.source_id) is None:
            raise ValueError("model source ID must be lowercase and safe")
        if re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise ValueError("model source commit must be a lowercase Git SHA-1")
        if re.fullmatch(r"[0-9a-f]{64}", self.receipt_sha256) is None:
            raise ValueError("model source receipt must be a lowercase SHA-256")
        resolved = self.source_root.resolve()
        if not resolved.is_dir():
            raise ValueError("model source root must be a materialized directory")
        object.__setattr__(self, "source_root", resolved)


@dataclass(frozen=True, slots=True)
class WindowShape:
    history: int
    horizon: int
    variables: int

    def __post_init__(self) -> None:
        if min(self.history, self.horizon, self.variables) <= 0:
            raise ValueError("window shape dimensions must be positive")


def _state_hash(
    source: ModelSourceContext,
    adapter_name: str,
    adapter_config: Mapping[str, JSONValue],
    state: Mapping[str, Tensor],
) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "source_id": source.source_id,
                "source_commit": source.commit,
                "source_receipt_sha256": source.receipt_sha256,
                "adapter_name": adapter_name,
                "adapter_config": dict(adapter_config),
            }
        )
    )
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


class OfficialOperablePredictor(nn.Module, ABC):
    def __init__(self, source: ModelSourceContext, shape: WindowShape) -> None:
        super().__init__()
        self.source = source
        self.shape = shape
        self._registered_sites: tuple[RegisteredSite, ...] = ()
        self._frozen = False

    @property
    @abstractmethod
    def adapter_name(self) -> str: ...

    @property
    @abstractmethod
    def adapter_config(self) -> Mapping[str, JSONValue]: ...

    @abstractmethod
    def _forward_distribution(self, histories: Tensor) -> ForecastDistribution: ...

    def _reset_adapter_parameters(self) -> None:
        """Reset adapter-only parameters after official module resets."""

    @property
    def model_hash(self) -> str:
        return _state_hash(
            self.source,
            self.adapter_name,
            self.adapter_config,
            self.state_dict(),
        )

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def reset_for_training(self, seed: int) -> None:
        if self._frozen:
            raise RuntimeError("frozen neural predictor cannot be reset")
        torch.manual_seed(seed)
        for module in self.modules():
            if module is self:
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

    def forward_distribution(self, histories: Tensor) -> ForecastDistribution:
        if histories.ndim != 3 or tuple(histories.shape[1:]) != (
            self.shape.history,
            self.shape.variables,
        ):
            raise ValueError("neural histories do not match the configured window shape")
        return self._forward_distribution(histories)

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

    def list_intervention_sites(self) -> tuple[InterventionSite, ...]:
        return tuple(
            validate_intervention_site(registered.contract) for registered in self._registered_sites
        )

    def _registered_site(self, requested: InterventionSite) -> RegisteredSite:
        matches = tuple(
            site
            for site in self._registered_sites
            if site.contract.site_name == requested.site_name and site.contract == requested
        )
        if len(matches) != 1:
            raise ValueError("operation requested an undeclared intervention site")
        return matches[0]

    def capture(
        self,
        batch: WindowBatch,
        sites: tuple[InterventionSite, ...],
    ) -> MappingProxyType[str, Tensor]:
        validate_window_batch(batch)
        registered = tuple(self._registered_site(site) for site in sites)
        captured: dict[str, Tensor] = {}
        with torch.no_grad(), installed_site_capture(self, registered, captured):
            self.forward_distribution(batch.x)
        if set(captured) != {site.contract.site_name for site in registered}:
            raise RuntimeError("registered capture site did not execute exactly once")
        return MappingProxyType({name: value.detach().clone() for name, value in captured.items()})

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
        if base.x.shape != source.x.shape:
            raise ValueError("base and source batches must have equal tensor shapes")
        sites = {site.site_name: site for site in self.list_intervention_sites()}
        try:
            contract = sites[spec.site_name]
        except KeyError as error:
            raise ValueError("intervention requested an undeclared site") from error
        if contract.layer != spec.layer:
            raise ValueError("intervention layer does not match the declared site")
        registered = self._registered_site(contract)
        base_value = self.capture(base, (contract,))[contract.site_name]
        source_value = self.capture(source, (contract,))[contract.site_name]
        replacement = _replacement(base_value, source_value, contract, spec)
        before_hash = self.model_hash
        with torch.no_grad(), installed_site_swap(self, registered, replacement):
            distribution = self.forward_distribution(base.x)
        if self.model_hash != before_hash:
            raise RuntimeError("activation intervention mutated model weights")
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


def _selection_mask(
    tensor: Tensor,
    site: InterventionSite,
    spec: InterventionSpec,
) -> Tensor:
    mask = torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)
    for axis, index in (
        (site.variable_axis, spec.variable_index),
        (site.patch_axis, spec.patch_index),
    ):
        if index is None:
            continue
        if axis is None:
            raise ValueError("intervention index targets an unavailable site axis")
        if type(index) is not int or index >= tensor.shape[axis]:
            raise ValueError("intervention index is outside the registered site axis")
        coordinate = torch.arange(tensor.shape[axis], device=tensor.device)
        shape = [1] * tensor.ndim
        shape[axis] = tensor.shape[axis]
        mask = mask & (coordinate.reshape(shape) == index)
    return mask


def _replacement(
    base: Tensor,
    source: Tensor,
    site: InterventionSite,
    spec: InterventionSpec,
) -> Tensor:
    if base.shape != source.shape:
        raise ValueError("captured base and source activations must share shape")
    mask = _selection_mask(base, site, spec)
    if spec.intervention_kind is InterventionKind.FULL_SWAP:
        candidate = source
    else:
        basis = spec.subspace_basis
        if basis is None or basis.shape[0] != base.shape[site.feature_axis]:
            raise ValueError("subspace basis does not match the site feature width")
        candidate = base + (source - base) @ basis @ basis.transpose(0, 1)
    return torch.where(mask, candidate, base)
