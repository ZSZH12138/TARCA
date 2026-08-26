from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from tarca.contracts import ForecastDistribution, InterventionSite, JSONValue, canonical_json_hash
from tarca.stage1b.config import load_world_suite
from tarca.stage1b.modeling.base import (
    ModelSourceContext,
    OfficialOperablePredictor,
    WindowShape,
)
from tarca.stage1b.modeling.hooks import RegisteredSite
from tarca.stage1b.reproduction import _isolated_upstream_imports
from tarca.stage1b.sources import (
    SourceMaterializationReceipt,
    SubprocessGitRunner,
    materialize_source,
)

_PATCHTST_SOURCE_ID = "patchtst"
_PATCHTST_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
_PATCHTST_ASSET = Path("PatchTST_supervised/layers/PatchTST_backbone.py")
_PATCHTST_ASSET_SHA256 = "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _receipt_hash(receipt: SourceMaterializationReceipt) -> str:
    return canonical_json_hash(
        {
            "source_id": receipt.source_id,
            "repository_url": receipt.repository_url,
            "commit": receipt.commit,
            "tree_sha256": receipt.tree_sha256,
            "asset_sha256": [list(item) for item in receipt.asset_sha256],
            "authorization_id": receipt.authorization_id,
        }
    )


def default_patchtst_source_context() -> ModelSourceContext:
    repository_root = _repository_root()
    source_cache = repository_root / "third_party/stage1b"
    checkout = source_cache / _PATCHTST_SOURCE_ID / _PATCHTST_COMMIT
    if not checkout.is_dir():
        raise RuntimeError(
            "pinned official PatchTST source is not materialized; run the source materializer"
        )
    suite = load_world_suite(repository_root / "configs/stage1b/worlds_v2.yaml")
    receipt = materialize_source(
        suite.source(_PATCHTST_SOURCE_ID),
        source_cache,
        SubprocessGitRunner.discover(),
    )
    return ModelSourceContext(
        source_id=receipt.source_id,
        commit=receipt.commit,
        source_root=receipt.checkout_root,
        receipt_sha256=_receipt_hash(receipt),
    )


def _verify_source_context(source: ModelSourceContext) -> None:
    repository_root = _repository_root()
    source_cache = repository_root / "third_party/stage1b"
    expected_root = (source_cache / _PATCHTST_SOURCE_ID / _PATCHTST_COMMIT).resolve()
    if source.source_root != expected_root:
        raise ValueError("PatchTST source root is outside the registered source cache")
    suite = load_world_suite(repository_root / "configs/stage1b/worlds_v2.yaml")
    receipt = materialize_source(
        suite.source(_PATCHTST_SOURCE_ID),
        source_cache,
        SubprocessGitRunner.discover(),
    )
    if (
        receipt.checkout_root.resolve() != source.source_root
        or receipt.commit != source.commit
        or _receipt_hash(receipt) != source.receipt_sha256
    ):
        raise ValueError("PatchTST model source context does not match its verified receipt")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_patchtst(
    source: ModelSourceContext,
    shape: WindowShape,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    patch_length: int,
    patch_stride: int,
) -> nn.Module:
    if source.source_id != _PATCHTST_SOURCE_ID or source.commit != _PATCHTST_COMMIT:
        raise ValueError("PatchTST adapter requires the registered official source commit")
    _verify_source_context(source)
    asset = source.source_root / _PATCHTST_ASSET
    if not asset.is_file() or _sha256_file(asset) != _PATCHTST_ASSET_SHA256:
        raise ValueError("PatchTST official backbone asset hash does not match")
    import_root = source.source_root / "PatchTST_supervised"
    with _isolated_upstream_imports(import_root, ("layers",)):
        module = importlib.import_module("layers.PatchTST_backbone")
        model_type = cast(Any, module.PatchTST_backbone)
        model = model_type(
            c_in=shape.variables,
            context_window=shape.history,
            target_window=shape.horizon,
            patch_len=patch_length,
            stride=patch_stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=0.0,
            head_dropout=0.0,
            revin=True,
            res_attention=True,
        )
    if not isinstance(model, nn.Module):
        raise TypeError("official PatchTST constructor did not return a torch module")
    return model


class ConditionalScaleHead(nn.Module):
    def __init__(self, d_model: int, patch_count: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model * patch_count, horizon)
        nn.init.constant_(self.linear.bias, -1.0)

    def forward(self, representation: Tensor) -> Tensor:
        flattened = representation.flatten(start_dim=2)
        return functional.softplus(self.linear(flattened)) + 1e-6


def _site_registry(
    variables: int,
    patch_count: int,
    d_model: int,
    n_layers: int,
) -> tuple[RegisteredSite, ...]:
    def contract(name: str, layer: int | None) -> InterventionSite:
        return InterventionSite(
            site_name=name,
            layer=layer,
            tensor_rank=4,
            batch_axis=0,
            variable_axis=1,
            patch_axis=2,
            feature_axis=3,
            shape_template=(None, variables, patch_count, d_model),
        )

    return (
        RegisteredSite(
            contract("encoder.input", None),
            "_mean_backbone.backbone.dropout",
            "FLAT_PATCH_FEATURE",
        ),
        *(
            RegisteredSite(
                contract(f"encoder.layer.{index}", index),
                f"_mean_backbone.backbone.encoder.layers.{index}",
                "FLAT_PATCH_FEATURE",
                tuple_output=True,
            )
            for index in range(n_layers)
        ),
        RegisteredSite(
            contract("encoder.representation", None),
            "_mean_backbone.backbone",
            "FEATURE_PATCH",
        ),
    )


class OfficialPatchTSTPredictor(OfficialOperablePredictor):
    supports_cross_variable_claim = False

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
        patch_length: int,
        patch_stride: int,
        source: ModelSourceContext | None = None,
    ) -> None:
        shape = WindowShape(history_length, horizon, input_dimension)
        resolved_source = default_patchtst_source_context() if source is None else source
        super().__init__(resolved_source, shape)
        if min(d_model, n_layers, n_heads, d_ff, patch_length, patch_stride) <= 0:
            raise ValueError("PatchTST adapter dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("PatchTST d_model must be divisible by n_heads")
        if patch_length > history_length or (history_length - patch_length) % patch_stride:
            raise ValueError("PatchTST patches must exactly cover the history")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("PatchTST dropout must be in [0, 1)")
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.patch_count = 1 + (history_length - patch_length) // patch_stride
        self._mean_backbone = _load_pinned_patchtst(
            resolved_source,
            shape,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )
        self._scale_head = ConditionalScaleHead(d_model, self.patch_count, horizon)
        self._registered_sites = _site_registry(
            input_dimension,
            self.patch_count,
            d_model,
            n_layers,
        )

    @property
    def history_length(self) -> int:
        return self.shape.history

    @property
    def horizon(self) -> int:
        return self.shape.horizon

    @property
    def input_dimension(self) -> int:
        return self.shape.variables

    @property
    def mean_backbone(self) -> nn.Module:
        return self._mean_backbone

    @property
    def adapter_name(self) -> str:
        return "PatchTSTReference"

    @property
    def adapter_config(self) -> Mapping[str, JSONValue]:
        return MappingProxyType(
            {
                "history_length": self.history_length,
                "horizon": self.horizon,
                "input_dimension": self.input_dimension,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout,
                "patch_length": self.patch_length,
                "patch_stride": self.patch_stride,
                "revin": True,
            }
        )

    def _reset_adapter_parameters(self) -> None:
        official = cast(Any, self._mean_backbone)
        nn.init.uniform_(official.backbone.W_pos, -0.02, 0.02)
        nn.init.ones_(official.revin_layer.affine_weight)
        nn.init.zeros_(official.revin_layer.affine_bias)
        nn.init.constant_(self._scale_head.linear.bias, -1.0)

    def _forward_distribution(self, histories: Tensor) -> ForecastDistribution:
        representation: list[Tensor] = []

        def capture_representation(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not isinstance(output, Tensor) or output.ndim != 4:
                raise ValueError("official PatchTST representation has an invalid shape")
            representation.append(output.permute(0, 1, 3, 2))

        backbone = self._mean_backbone.get_submodule("backbone")
        handle = backbone.register_forward_hook(capture_representation)
        try:
            official_mean = self._mean_backbone(histories.permute(0, 2, 1))
        finally:
            handle.remove()
        if not isinstance(official_mean, Tensor) or official_mean.shape != (
            histories.shape[0],
            self.input_dimension,
            self.horizon,
        ):
            raise ValueError("official PatchTST mean output has an invalid shape")
        if len(representation) != 1:
            raise RuntimeError("official PatchTST representation was not captured exactly once")
        normalized_scale = self._scale_head(representation[0]).transpose(1, 2)
        input_scale = torch.sqrt(
            torch.var(histories, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return ForecastDistribution(
            mean=official_mean.permute(0, 2, 1),
            scale=normalized_scale * input_scale,
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=None,
            target_names=tuple(f"x{index}" for index in range(self.input_dimension)),
        )
