from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from tarca.contracts import InterventionSite


@dataclass(frozen=True, slots=True)
class RegisteredSite:
    contract: InterventionSite
    module_path: str
    layout: Literal["IDENTITY", "FLAT_PATCH_FEATURE", "FEATURE_PATCH"]
    tuple_output: bool = False


def resolve_registered_module(model: nn.Module, site: RegisteredSite) -> nn.Module:
    if not site.module_path or site.module_path.startswith(".") or ".." in site.module_path:
        raise ValueError("registered module path is invalid")
    try:
        return model.get_submodule(site.module_path)
    except AttributeError as error:
        raise ValueError("registered module path does not resolve") from error


def _primary_output(output: object, site: RegisteredSite) -> Tensor:
    primary = output[0] if site.tuple_output and isinstance(output, tuple) else output
    if not isinstance(primary, Tensor):
        raise TypeError("registered site output is not a tensor")
    return primary


def _canonical_output(output: object, site: RegisteredSite) -> Tensor:
    tensor = _primary_output(output, site)
    if site.layout == "IDENTITY":
        canonical = tensor
        expected = site.contract.shape_template
        if canonical.ndim != site.contract.tensor_rank or any(
            size is not None and canonical.shape[index] != size
            for index, size in enumerate(expected)
        ):
            raise ValueError("registered site output violates its declared axes")
        return canonical
    variables = site.contract.shape_template[1]
    if not isinstance(variables, int) or variables <= 0:
        raise ValueError("registered PatchTST site requires a fixed variable axis")
    if site.layout == "FLAT_PATCH_FEATURE":
        if tensor.ndim != 3 or tensor.shape[0] % variables:
            raise ValueError("flattened PatchTST site output has an invalid shape")
        canonical = tensor.reshape(-1, variables, tensor.shape[1], tensor.shape[2])
    else:
        if tensor.ndim != 4 or tensor.shape[1] != variables:
            raise ValueError("PatchTST backbone output has an invalid shape")
        canonical = tensor.permute(0, 1, 3, 2)
    expected = site.contract.shape_template
    if canonical.ndim != site.contract.tensor_rank or any(
        size is not None and canonical.shape[index] != size for index, size in enumerate(expected)
    ):
        raise ValueError("registered site output violates its declared axes")
    return canonical


def _restore_output(original: object, canonical: Tensor, site: RegisteredSite) -> object:
    if site.layout == "IDENTITY":
        restored = canonical
    elif site.layout == "FLAT_PATCH_FEATURE":
        restored = canonical.reshape(
            canonical.shape[0] * canonical.shape[1],
            canonical.shape[2],
            canonical.shape[3],
        )
    else:
        restored = canonical.permute(0, 1, 3, 2)
    if site.tuple_output:
        if not isinstance(original, tuple):
            raise TypeError("registered tuple site returned a non-tuple output")
        return (restored, *original[1:])
    return restored


@contextmanager
def installed_site_capture(
    model: nn.Module,
    sites: tuple[RegisteredSite, ...],
    captured: MutableMapping[str, Tensor],
) -> Iterator[None]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for site in sites:
            module = resolve_registered_module(model, site)

            def capture_hook(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                output: object,
                *,
                registered: RegisteredSite = site,
            ) -> None:
                name = registered.contract.site_name
                if name in captured:
                    raise RuntimeError("registered capture site executed more than once")
                captured[name] = _canonical_output(output, registered).detach().clone()

            handles.append(module.register_forward_hook(capture_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def installed_site_swap(
    model: nn.Module,
    site: RegisteredSite,
    replacement: Tensor,
) -> Iterator[None]:
    module = resolve_registered_module(model, site)
    immutable_replacement = replacement.detach().clone()

    def swap_hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        current = _canonical_output(output, site)
        if current.shape != immutable_replacement.shape:
            raise ValueError("site replacement shape does not match runtime activation")
        replacement_on_device = immutable_replacement.to(
            device=current.device,
            dtype=current.dtype,
        )
        return _restore_output(output, replacement_on_device, site)

    handle = module.register_forward_hook(swap_hook)
    try:
        yield
    finally:
        handle.remove()
