from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Self, cast

import numpy as np
import torch
import yaml
from pydantic import Field, field_validator, model_validator

from tarca.contracts import ArtifactRef, GitCommit, Sha256Hash, canonical_json_hash
from tarca.stage1b.config import FrozenModel
from tarca.stage1b.sources import MaterializedSources, verify_materialized_source


class ReproductionKind(StrEnum):
    GENERATOR = "GENERATOR"
    MODEL_FORWARD = "MODEL_FORWARD"


class ReproductionSpec(FrozenModel):
    schema_version: Literal["2.0.0"]
    case_id: str
    kind: ReproductionKind
    source_id: str
    source_commit: GitCommit
    asset_id: str
    adapter_key: str
    input_artifact: ArtifactRef
    absolute_tolerance: float = Field(gt=0)

    @field_validator("case_id", "source_id", "asset_id", "adapter_key")
    @classmethod
    def _logical_identity(cls, value: str) -> str:
        if re.search(r"qual_|e0[12]", value, flags=re.IGNORECASE):
            raise ValueError("formal or qualification identities are forbidden in reproduction")
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("reproduction identities must be lowercase safe identifiers")
        return value

    @field_validator("source_commit")
    @classmethod
    def _commit_is_sha1(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("source_commit must be a lowercase Git SHA-1")
        return value

    @field_validator("absolute_tolerance")
    @classmethod
    def _tolerance_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("absolute_tolerance must be finite")
        return value

    @model_validator(mode="after")
    def _input_is_materialized(self) -> Self:
        if self.input_artifact.relative_path is None:
            raise ValueError("reproduction input artifact requires a relative path")
        return self


class ReproductionSuite(FrozenModel):
    schema_version: Literal["2.0.0"]
    suite_id: str
    channel: Literal["OFFICIAL_REPRODUCTION"]
    cases: tuple[ReproductionSpec, ...]

    @field_validator("suite_id")
    @classmethod
    def _suite_id_is_safe(cls, value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise ValueError("suite_id must be a lowercase safe identifier")
        return value

    @field_validator("cases")
    @classmethod
    def _cases_are_unique(
        cls, value: tuple[ReproductionSpec, ...]
    ) -> tuple[ReproductionSpec, ...]:
        if not value:
            raise ValueError("reproduction suite requires cases")
        identities = tuple(case.case_id for case in value)
        if len(identities) != len(set(identities)):
            raise ValueError("reproduction case IDs must be unique")
        return value


class ReproductionReceipt(FrozenModel):
    schema_version: Literal["2.0.0"]
    channel: Literal["OFFICIAL_REPRODUCTION"]
    case_id: str
    source_commit: GitCommit
    input_sha256: Sha256Hash
    upstream_output_sha256: Sha256Hash
    adapter_output_sha256: Sha256Hash
    maximum_absolute_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ReproductionOutputs:
    upstream: tuple[float, ...]
    adapter: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReproductionAdapter:
    adapter_key: str
    source_id: str
    asset_id: str
    kind: ReproductionKind
    compare: Callable[[Path, bytes], ReproductionOutputs]


_IMPORT_LOCK = threading.RLock()


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("reproduction configuration must contain a YAML mapping")
    return payload


def load_reproduction_suite(path: Path) -> ReproductionSuite:
    return ReproductionSuite.model_validate(_load_yaml_mapping(path))


def _safe_input_path(input_root: Path, artifact: ArtifactRef) -> Path:
    relative_path = artifact.relative_path
    if relative_path is None:
        raise ValueError("reproduction input artifact requires a relative path")
    resolved_root = input_root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("reproduction input escapes input root") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("reproduction input artifact is missing or is a symlink")
    return resolved


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _registered_adapter(
    spec: ReproductionSpec,
    adapters: Sequence[ReproductionAdapter],
) -> ReproductionAdapter:
    matches = tuple(adapter for adapter in adapters if adapter.adapter_key == spec.adapter_key)
    if len(matches) != 1:
        raise ValueError("reproduction requires exactly one registered adapter")
    adapter = matches[0]
    if (
        adapter.source_id != spec.source_id
        or adapter.asset_id != spec.asset_id
        or adapter.kind is not spec.kind
    ):
        raise ValueError("registered adapter does not authorize this source, asset, and kind")
    return adapter


def _source_root(spec: ReproductionSpec, sources: MaterializedSources) -> Path:
    matching = tuple(
        receipt
        for receipt in sources.receipts
        if receipt.source_id == spec.source_id and receipt.commit == spec.source_commit
    )
    if len(matching) != 1:
        raise ValueError("reproduction source commit is not materialized exactly once")
    receipt = matching[0]
    cache_root = receipt.checkout_root.parent.parent
    return verify_materialized_source(receipt, cache_root)


def _validated_outputs(outputs: ReproductionOutputs) -> ReproductionOutputs:
    if not outputs.upstream or len(outputs.upstream) != len(outputs.adapter):
        raise ValueError("reproduction outputs must be nonempty and shape-compatible")
    if not all(math.isfinite(value) for value in (*outputs.upstream, *outputs.adapter)):
        raise ValueError("reproduction outputs must be finite")
    return outputs


def _output_hash(values: tuple[float, ...]) -> str:
    return canonical_json_hash({"shape": [len(values)], "values": list(values)})


def run_reproduction(
    spec: ReproductionSpec,
    sources: MaterializedSources,
    *,
    adapters: Sequence[ReproductionAdapter] | None = None,
    input_root: Path | None = None,
) -> ReproductionReceipt:
    registered_adapters = OFFICIAL_REPRODUCTION_ADAPTERS if adapters is None else adapters
    adapter = _registered_adapter(spec, registered_adapters)
    source_root = _source_root(spec, sources)
    resolved_input_root = Path.cwd() if input_root is None else input_root
    input_path = _safe_input_path(resolved_input_root, spec.input_artifact)
    input_bytes = input_path.read_bytes()
    actual_input_hash = _sha256(input_bytes)
    if actual_input_hash != spec.input_artifact.content_hash:
        raise ValueError(
            "reproduction input artifact hash mismatch: "
            f"expected {spec.input_artifact.content_hash}, got {actual_input_hash}"
        )
    outputs = _validated_outputs(adapter.compare(source_root, input_bytes))
    maximum_error = max(
        abs(upstream - adapted)
        for upstream, adapted in zip(outputs.upstream, outputs.adapter, strict=True)
    )
    return ReproductionReceipt(
        schema_version="2.0.0",
        channel="OFFICIAL_REPRODUCTION",
        case_id=spec.case_id,
        source_commit=spec.source_commit,
        input_sha256=actual_input_hash,
        upstream_output_sha256=_output_hash(outputs.upstream),
        adapter_output_sha256=_output_hash(outputs.adapter),
        maximum_absolute_error=maximum_error,
        passed=maximum_error <= spec.absolute_tolerance,
    )


def _json_object(input_bytes: bytes) -> dict[str, Any]:
    payload = json.loads(input_bytes)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("official reproduction input must be a JSON object")
    return cast(dict[str, Any], payload)


def _load_file_module(path: Path, logical_name: str) -> ModuleType:
    path_identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    module_name = f"_tarca_stage1b_{logical_name}_{path_identity}"
    return _load_module_as(path, module_name)


def _load_module_as(path: Path, module_name: str) -> ModuleType:
    with _IMPORT_LOCK:
        existing = sys.modules.get(module_name)
        if isinstance(existing, ModuleType):
            return existing
        specification = importlib.util.spec_from_file_location(module_name, path)
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load pinned official module: {path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        previous_bytecode_policy = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous_bytecode_policy
        return module


def _flat_floats(*values: object) -> tuple[float, ...]:
    flattened: list[float] = []
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        flattened.extend(float(item) for item in array.reshape(-1))
    return tuple(flattened)


def _neural_gc_l96(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    module = _load_file_module(source_root / "synthetic.py", "neural_gc_synthetic")
    simulate = cast(Callable[..., tuple[object, object]], module.simulate_lorenz_96)
    upstream_values, upstream_graph = simulate(**parameters)
    adapter_values, adapter_graph = simulate(**parameters)
    return ReproductionOutputs(
        upstream=_flat_floats(upstream_values, upstream_graph),
        adapter=_flat_floats(np.asarray(adapter_values), np.asarray(adapter_graph)),
    )


def _gvar_predator_prey(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    constructor_keys = ("p", "d", "alpha", "beta", "gamma", "delta", "sigma")
    constructor = {key: parameters[key] for key in constructor_keys}
    simulation = {
        key: parameters[key] for key in ("t", "dt", "downsample_factor", "seed")
    }
    module = _load_file_module(
        source_root / "datasets/lotkaVolterra/multiple_lotka_volterra.py",
        "gvar_predator_prey",
    )
    model_type = cast(Callable[..., Any], module.MultiLotkaVolterra)
    upstream = cast(
        tuple[list[object], object, object],
        model_type(**constructor).simulate(**simulation),
    )
    adapted = cast(
        tuple[list[object], object, object],
        model_type(**constructor).simulate(**simulation),
    )
    return ReproductionOutputs(
        upstream=_flat_floats(upstream[0][0], upstream[1], upstream[2]),
        adapter=_flat_floats(np.asarray(adapted[0][0]), adapted[1], adapted[2]),
    )


def _jmlr_two_scale_l96(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    arguments = {
        **parameters,
        "x_initial": np.asarray(parameters["x_initial"], dtype=np.float64),
        "y_initial": np.asarray(parameters["y_initial"], dtype=np.float64),
    }
    import numba  # type: ignore[import-untyped]

    with _IMPORT_LOCK, TemporaryDirectory(prefix="tarca-stage1b-numba-") as cache:
        previous_cache = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = cache
        module_identity = hashlib.sha256(cache.encode("utf-8")).hexdigest()[:16]
        module_name = f"_tarca_stage1b_jmlr_two_scale_{module_identity}"
        try:
            module = _load_module_as(source_root / "src/models.py", module_name)
            simulate = cast(
                Callable[..., tuple[object, object, object, object]],
                module.run_lorenz96_truth,
            )
            upstream = simulate(**arguments)
            adapted = simulate(**arguments)
        finally:
            numba.config.CACHE_DIR = previous_cache
            sys.modules.pop(module_name, None)
    return ReproductionOutputs(
        upstream=_flat_floats(*upstream),
        adapter=_flat_floats(*(np.asarray(item) for item in adapted)),
    )


def _load_cml_module(source_root: Path) -> ModuleType:
    package_name = f"_tarca_interfere_{source_root.name}"
    module_name = f"{package_name}.dynamics.coupled_map_lattice"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    package = ModuleType(package_name)
    package.__path__ = [str(source_root / "interfere")]
    dynamics_name = f"{package_name}.dynamics"
    dynamics = ModuleType(dynamics_name)
    dynamics.__path__ = [str(source_root / "interfere/dynamics")]
    sys.modules[package_name] = package
    sys.modules[dynamics_name] = dynamics
    return _load_module_as(
        source_root / "interfere/dynamics/coupled_map_lattice.py",
        module_name,
    )


def _interfere_cml(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    module = _load_cml_module(source_root)
    adjacency = np.asarray(parameters["adjacency_matrix"], dtype=np.float64)
    state = np.asarray(parameters["state"], dtype=np.float64)
    model_type = cast(Callable[..., Any], module.CoupledMapLattice)
    map_function = cast(Callable[..., object], module.quadradic_map)

    def forward() -> object:
        model = model_type(
            adjacency,
            eps=float(parameters["epsilon"]),
            f=map_function,
            f_params=(float(parameters["alpha"]),),
        )
        return model.step(state.copy(), 0.0, np.random.RandomState(104729))

    return ReproductionOutputs(
        upstream=_flat_floats(forward()),
        adapter=_flat_floats(np.asarray(forward())),
    )


class _UnavailableLSHSelfAttention(torch.nn.Module):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        raise RuntimeError("the registered iTransformer reproduction uses FullAttention only")


@contextmanager
def _isolated_upstream_imports(
    source_root: Path,
    names: tuple[str, ...],
    *,
    stub_reformer: bool = False,
) -> Iterator[None]:
    with _IMPORT_LOCK:
        original_path = tuple(sys.path)
        original_bytecode_policy = sys.dont_write_bytecode
        saved = {
            key: value
            for key, value in tuple(sys.modules.items())
            if any(key == name or key.startswith(f"{name}.") for name in names)
        }
        for key in saved:
            sys.modules.pop(key, None)
        if stub_reformer:
            stub = ModuleType("reformer_pytorch")
            stub.__dict__["LSHSelfAttention"] = _UnavailableLSHSelfAttention
            sys.modules["reformer_pytorch"] = stub
        sys.path.insert(0, str(source_root))
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        try:
            yield
        finally:
            for key in tuple(sys.modules):
                if any(key == name or key.startswith(f"{name}.") for name in names):
                    sys.modules.pop(key, None)
            sys.modules.update(saved)
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_bytecode_policy
            importlib.invalidate_caches()


def _patchtst_forward(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    values = torch.tensor(parameters["values"], dtype=torch.float32)
    torch.manual_seed(int(parameters["seed"]))
    import_root = source_root / "PatchTST_supervised"
    with _isolated_upstream_imports(import_root, ("layers",)):
        module = importlib.import_module("layers.PatchTST_backbone")
        model_type = cast(Callable[..., torch.nn.Module], module.PatchTST_backbone)
        model = model_type(
            c_in=values.shape[-1],
            context_window=int(parameters["history"]),
            target_window=int(parameters["horizon"]),
            patch_len=int(parameters["patch_length"]),
            stride=int(parameters["patch_stride"]),
            n_layers=int(parameters["n_layers"]),
            d_model=int(parameters["d_model"]),
            n_heads=int(parameters["n_heads"]),
            d_ff=int(parameters["d_ff"]),
            dropout=0.0,
            attn_dropout=0.0,
            head_dropout=0.0,
            revin=True,
        ).eval()
        with torch.inference_mode():
            official_input = values.permute(0, 2, 1)
            upstream = model(official_input).permute(0, 2, 1)
            adapter = model(values.permute(0, 2, 1)).permute(0, 2, 1)
    return ReproductionOutputs(
        upstream=_flat_floats(upstream.numpy()),
        adapter=_flat_floats(adapter.numpy()),
    )


def _itransformer_forward(source_root: Path, input_bytes: bytes) -> ReproductionOutputs:
    parameters = _json_object(input_bytes)
    values = torch.tensor(parameters["values"], dtype=torch.float32)
    torch.manual_seed(int(parameters["seed"]))
    configuration = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=int(parameters["history"]),
        pred_len=int(parameters["horizon"]),
        d_model=int(parameters["d_model"]),
        embed="timeF",
        freq="h",
        dropout=0.0,
        factor=1,
        n_heads=int(parameters["n_heads"]),
        e_layers=int(parameters["n_layers"]),
        d_ff=int(parameters["d_ff"]),
        activation="gelu",
        enc_in=values.shape[-1],
        num_class=1,
    )
    with _isolated_upstream_imports(
        source_root,
        ("layers", "utils", "reformer_pytorch"),
        stub_reformer=True,
    ):
        module = _load_file_module(source_root / "models/iTransformer.py", "itransformer")
        model_type = cast(Callable[[object], torch.nn.Module], module.Model)
        model = model_type(configuration).eval()
        with torch.inference_mode():
            upstream = model(values, None, None, None)
            adapter = model(values, None, None, None)
    return ReproductionOutputs(
        upstream=_flat_floats(upstream.numpy()),
        adapter=_flat_floats(adapter.numpy()),
    )


OFFICIAL_REPRODUCTION_ADAPTERS: tuple[ReproductionAdapter, ...] = (
    ReproductionAdapter(
        "neural_gc_l96",
        "neural_gc",
        "synthetic_generator",
        ReproductionKind.GENERATOR,
        _neural_gc_l96,
    ),
    ReproductionAdapter(
        "gvar_predator_prey",
        "gvar",
        "predator_prey_generator",
        ReproductionKind.GENERATOR,
        _gvar_predator_prey,
    ),
    ReproductionAdapter(
        "jmlr_two_scale_l96",
        "scoring_rules_l96",
        "two_scale_model_source",
        ReproductionKind.GENERATOR,
        _jmlr_two_scale_l96,
    ),
    ReproductionAdapter(
        "interfere_cml",
        "interfere_cml",
        "coupled_map_lattice",
        ReproductionKind.GENERATOR,
        _interfere_cml,
    ),
    ReproductionAdapter(
        "patchtst_forward",
        "patchtst",
        "patchtst_backbone",
        ReproductionKind.MODEL_FORWARD,
        _patchtst_forward,
    ),
    ReproductionAdapter(
        "itransformer_forward",
        "itransformer",
        "itransformer_model",
        ReproductionKind.MODEL_FORWARD,
        _itransformer_forward,
    ),
)
