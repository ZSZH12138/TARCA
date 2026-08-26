from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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
from tarca.stage1b.reproduction import _isolated_upstream_imports, _load_file_module
from tarca.stage1b.sources import (
    SourceMaterializationReceipt,
    SubprocessGitRunner,
    materialize_source,
)

_ITRANSFORMER_SOURCE_ID = "itransformer"
_ITRANSFORMER_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"
_ITRANSFORMER_ASSET = Path("models/iTransformer.py")
_ITRANSFORMER_ASSET_SHA256 = "7fdc721d041b0f8f63be8fa794ecd68422fd958c7c8d449026320fd9f368788e"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_itransformer_source_context() -> ModelSourceContext:
    repository_root = _repository_root()
    source_cache = repository_root / "third_party/stage1b"
    checkout = source_cache / _ITRANSFORMER_SOURCE_ID / _ITRANSFORMER_COMMIT
    if not checkout.is_dir():
        raise RuntimeError(
            "pinned official iTransformer source is not materialized; run the source materializer"
        )
    suite = load_world_suite(repository_root / "configs/stage1b/worlds_v2.yaml")
    receipt = materialize_source(
        suite.source(_ITRANSFORMER_SOURCE_ID),
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
    expected_root = (source_cache / _ITRANSFORMER_SOURCE_ID / _ITRANSFORMER_COMMIT).resolve()
    if source.source_root != expected_root:
        raise ValueError("iTransformer source root is outside the registered source cache")
    suite = load_world_suite(repository_root / "configs/stage1b/worlds_v2.yaml")
    receipt = materialize_source(
        suite.source(_ITRANSFORMER_SOURCE_ID),
        source_cache,
        SubprocessGitRunner.discover(),
    )
    if (
        receipt.checkout_root.resolve() != source.source_root
        or receipt.commit != source.commit
        or _receipt_hash(receipt) != source.receipt_sha256
    ):
        raise ValueError("iTransformer model source context does not match its verified receipt")


def _load_pinned_itransformer(
    source: ModelSourceContext,
    shape: WindowShape,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
) -> nn.Module:
    if source.source_id != _ITRANSFORMER_SOURCE_ID or source.commit != _ITRANSFORMER_COMMIT:
        raise ValueError("iTransformer adapter requires the registered official source commit")
    _verify_source_context(source)
    asset = source.source_root / _ITRANSFORMER_ASSET
    if not asset.is_file() or _sha256_file(asset) != _ITRANSFORMER_ASSET_SHA256:
        raise ValueError("iTransformer official model asset hash does not match")
    configuration = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=shape.history,
        pred_len=shape.horizon,
        d_model=d_model,
        embed="timeF",
        freq="h",
        dropout=dropout,
        factor=1,
        n_heads=n_heads,
        e_layers=n_layers,
        d_ff=d_ff,
        activation="gelu",
        enc_in=shape.variables,
        num_class=1,
    )
    with _isolated_upstream_imports(
        source.source_root,
        ("layers", "utils", "reformer_pytorch"),
        stub_reformer=True,
    ):
        module = _load_file_module(asset, "itransformer_model")
        model_type = cast(Any, module.Model)
        model = model_type(configuration)
    if not isinstance(model, nn.Module):
        raise TypeError("official iTransformer constructor did not return a torch module")
    return model


class ITransformerConditionalScaleHead(nn.Module):
    def __init__(self, d_model: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, horizon)
        nn.init.constant_(self.linear.bias, -1.0)

    def forward(self, representation: Tensor) -> Tensor:
        return functional.softplus(self.linear(representation)) + 1e-6


def _site_registry(
    variables: int,
    d_model: int,
    n_layers: int,
) -> tuple[RegisteredSite, ...]:
    def contract(name: str, layer: int | None) -> InterventionSite:
        return InterventionSite(
            site_name=name,
            layer=layer,
            tensor_rank=3,
            batch_axis=0,
            variable_axis=1,
            patch_axis=None,
            feature_axis=2,
            shape_template=(None, variables, d_model),
        )

    return (
        RegisteredSite(contract("encoder.input", None), "_mean_backbone.enc_embedding", "IDENTITY"),
        *(
            RegisteredSite(
                contract(f"encoder.layer.{index}", index),
                f"_mean_backbone.encoder.attn_layers.{index}",
                "IDENTITY",
                tuple_output=True,
            )
            for index in range(n_layers)
        ),
        RegisteredSite(
            contract("encoder.representation", None),
            "_mean_backbone.encoder.norm",
            "IDENTITY",
        ),
    )


class OfficialITransformerPredictor(OfficialOperablePredictor):
    supports_cross_variable_claim = True

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
        source: ModelSourceContext | None = None,
    ) -> None:
        shape = WindowShape(history_length, horizon, input_dimension)
        resolved_source = default_itransformer_source_context() if source is None else source
        super().__init__(resolved_source, shape)
        if min(d_model, n_layers, n_heads, d_ff) <= 0:
            raise ValueError("iTransformer adapter dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("iTransformer d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("iTransformer dropout must be in [0, 1)")
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self._mean_backbone = _load_pinned_itransformer(
            resolved_source,
            shape,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self._scale_head = ITransformerConditionalScaleHead(d_model, horizon)
        self._registered_sites = _site_registry(input_dimension, d_model, n_layers)

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
        return "ITransformerReference"

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
                "normalization": "OFFICIAL_NON_STATIONARY_TRANSFORMER",
            }
        )

    def _reset_adapter_parameters(self) -> None:
        nn.init.constant_(self._scale_head.linear.bias, -1.0)

    def _forward_distribution(self, histories: Tensor) -> ForecastDistribution:
        representations: list[Tensor] = []

        def capture_representation(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not isinstance(output, Tensor) or output.ndim != 3:
                raise ValueError("official iTransformer representation has an invalid shape")
            representations.append(output)

        representation_module = self._mean_backbone.get_submodule("encoder.norm")
        handle = representation_module.register_forward_hook(capture_representation)
        try:
            official_mean = self._mean_backbone(histories, None, None, None)
        finally:
            handle.remove()
        expected_mean_shape = (histories.shape[0], self.horizon, self.input_dimension)
        if not isinstance(official_mean, Tensor) or official_mean.shape != expected_mean_shape:
            raise ValueError("official iTransformer mean output has an invalid shape")
        if len(representations) != 1:
            raise RuntimeError("official iTransformer representation was not captured exactly once")
        normalized_scale = self._scale_head(representations[0]).transpose(1, 2)
        input_scale = torch.sqrt(
            torch.var(histories, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return ForecastDistribution(
            mean=official_mean,
            scale=normalized_scale * input_scale,
            quantiles=MappingProxyType({}),
            logits=None,
            samples=None,
            window_id=None,
            target_names=tuple(f"x{index}" for index in range(self.input_dimension)),
        )
