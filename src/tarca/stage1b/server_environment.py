from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from io import BytesIO

import psutil
import torch


@dataclass(frozen=True, slots=True)
class ServerEnvironmentExpectation:
    python_minor: tuple[int, int]
    torch_version: str
    cuda_version: str
    gpu_count: int
    gpu_name_substring: str
    minimum_vram_bytes: int
    minimum_cpu_count: int
    minimum_ram_bytes: int

    def __post_init__(self) -> None:
        if len(self.python_minor) != 2 or any(item < 0 for item in self.python_minor):
            raise ValueError("python_minor must contain two nonnegative components")
        if not self.torch_version or not self.cuda_version or not self.gpu_name_substring:
            raise ValueError("runtime version and GPU expectations must be nonblank")
        if (
            min(
                self.gpu_count,
                self.minimum_vram_bytes,
                self.minimum_cpu_count,
                self.minimum_ram_bytes,
            )
            <= 0
        ):
            raise ValueError("runtime resource expectations must be positive")


@dataclass(frozen=True, slots=True)
class ServerEnvironmentFacts:
    python_version: str
    python_minor: tuple[int, int]
    torch_version: str
    cuda_version: str
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]
    cpu_count: int
    ram_bytes: int


@dataclass(frozen=True, slots=True)
class ServerEnvironmentReceipt:
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_names: tuple[str, ...]
    gpu_vram_bytes: tuple[int, ...]
    cpu_count: int
    ram_bytes: int
    cuda_probe_passed: bool
    receipt_sha256: str


def _release_version(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def _collect_facts() -> ServerEnvironmentFacts:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the server runtime requires CUDA")
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("CUDA version is unavailable from the installed PyTorch build")
    gpu_count = torch.cuda.device_count()
    gpu_names = tuple(torch.cuda.get_device_name(index) for index in range(gpu_count))
    gpu_vram_bytes = tuple(
        int(torch.cuda.get_device_properties(index).total_memory) for index in range(gpu_count)
    )
    physical_cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 0
    return ServerEnvironmentFacts(
        python_version=platform.python_version(),
        python_minor=(sys.version_info.major, sys.version_info.minor),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        gpu_names=gpu_names,
        gpu_vram_bytes=gpu_vram_bytes,
        cpu_count=int(physical_cpu_count),
        ram_bytes=int(psutil.virtual_memory().total),
    )


def _probe_cuda_device(device_index: int) -> bool:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(104729 + device_index)
    model = torch.nn.Linear(8, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    features = torch.randn((16, 8), device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(features).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    serialized = BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    restored = torch.nn.Linear(8, 4).to(device)
    restored.load_state_dict(torch.load(serialized, map_location=device, weights_only=True))
    return all(
        torch.equal(source, target)
        for source, target in zip(model.parameters(), restored.parameters(), strict=True)
    )


def _validate_facts(
    expectation: ServerEnvironmentExpectation,
    facts: ServerEnvironmentFacts,
) -> None:
    if facts.python_minor != expectation.python_minor:
        raise RuntimeError(
            f"Python minor mismatch: expected {expectation.python_minor}, got {facts.python_minor}"
        )
    if _release_version(facts.torch_version) != expectation.torch_version:
        raise RuntimeError(
            f"PyTorch version mismatch: expected {expectation.torch_version}, "
            f"got {facts.torch_version}"
        )
    if facts.cuda_version != expectation.cuda_version:
        raise RuntimeError(
            f"CUDA version mismatch: expected {expectation.cuda_version}, got {facts.cuda_version}"
        )
    if len(facts.gpu_names) != expectation.gpu_count:
        raise RuntimeError(
            f"GPU count mismatch: expected {expectation.gpu_count}, got {len(facts.gpu_names)}"
        )
    if any(expectation.gpu_name_substring not in name for name in facts.gpu_names):
        raise RuntimeError(
            f"GPU model mismatch: expected all devices to contain "
            f"{expectation.gpu_name_substring!r}"
        )
    if any(value < expectation.minimum_vram_bytes for value in facts.gpu_vram_bytes):
        raise RuntimeError("GPU VRAM is below the required per-device minimum")
    if facts.cpu_count < expectation.minimum_cpu_count:
        raise RuntimeError("physical CPU count is below the required minimum")
    if facts.ram_bytes < expectation.minimum_ram_bytes:
        raise RuntimeError("host RAM is below the required minimum")


def _receipt_hash(facts: ServerEnvironmentFacts, cuda_probe_passed: bool) -> str:
    payload = {**asdict(facts), "cuda_probe_passed": cuda_probe_passed}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_server_environment(
    expectation: ServerEnvironmentExpectation,
) -> ServerEnvironmentReceipt:
    facts = _collect_facts()
    _validate_facts(expectation, facts)
    cuda_probe_passed = all(
        _probe_cuda_device(device_index) for device_index in range(expectation.gpu_count)
    )
    if not cuda_probe_passed:
        raise RuntimeError("one or more CUDA device probes failed")
    return ServerEnvironmentReceipt(
        python_version=facts.python_version,
        torch_version=facts.torch_version,
        cuda_version=facts.cuda_version,
        gpu_names=facts.gpu_names,
        gpu_vram_bytes=facts.gpu_vram_bytes,
        cpu_count=facts.cpu_count,
        ram_bytes=facts.ram_bytes,
        cuda_probe_passed=cuda_probe_passed,
        receipt_sha256=_receipt_hash(facts, cuda_probe_passed),
    )
