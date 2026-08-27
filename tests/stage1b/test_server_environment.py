from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tarca.stage1b.server_environment as server_environment
from tarca.stage1b.server_environment import (
    ServerEnvironmentExpectation,
    ServerEnvironmentFacts,
    validate_server_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_ROOT = REPO_ROOT / "deploy" / "stage1b" / "py310"


def test_py310_bootstrap_supplies_frozen_stdlib_names() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(BOOTSTRAP_ROOT), str(REPO_ROOT / "src"))),
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "from datetime import UTC; from enum import StrEnum; "
            "from typing import Self; import tomllib; "
            "assert str(StrEnum('Probe', {'OK': 'OK'}).OK) == 'OK'; "
            "assert UTC is not None and Self is not None and tomllib is not None",
        ),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _expectation() -> ServerEnvironmentExpectation:
    return ServerEnvironmentExpectation(
        python_minor=(3, 10),
        torch_version="2.2.2",
        cuda_version="12.1",
        gpu_count=2,
        gpu_name_substring="RTX 4090",
        minimum_vram_bytes=24 * 1024**3,
        minimum_cpu_count=28,
        minimum_ram_bytes=224 * 1024**3,
    )


def test_server_environment_rejects_cpu_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        validate_server_environment(_expectation())


def test_server_environment_returns_immutable_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = ServerEnvironmentFacts(
        python_version="3.10.20",
        python_minor=(3, 10),
        torch_version="2.2.2+cu121",
        cuda_version="12.1",
        gpu_names=("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"),
        gpu_vram_bytes=(24 * 1024**3, 24 * 1024**3),
        cpu_count=28,
        ram_bytes=224 * 1024**3,
    )
    monkeypatch.setattr(server_environment, "_collect_facts", lambda: facts)
    monkeypatch.setattr(server_environment, "_probe_cuda_device", lambda _device: True)

    receipt = validate_server_environment(_expectation())

    assert receipt.cuda_probe_passed is True
    assert receipt.gpu_names == facts.gpu_names
    assert len(receipt.receipt_sha256) == 64
    with pytest.raises((AttributeError, TypeError)):
        receipt.cpu_count = 4  # type: ignore[misc]


def test_server_environment_rejects_wrong_gpu_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = ServerEnvironmentFacts(
        python_version="3.10.20",
        python_minor=(3, 10),
        torch_version="2.2.2+cu121",
        cuda_version="12.1",
        gpu_names=("NVIDIA A100", "NVIDIA A100"),
        gpu_vram_bytes=(80 * 1024**3, 80 * 1024**3),
        cpu_count=28,
        ram_bytes=224 * 1024**3,
    )
    monkeypatch.setattr(server_environment, "_collect_facts", lambda: facts)

    with pytest.raises(RuntimeError, match="GPU model"):
        validate_server_environment(_expectation())


def test_runtime_expectation_accepts_driver_reported_rtx4090_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation = ServerEnvironmentExpectation(
        python_minor=(3, 10),
        torch_version="2.2.2",
        cuda_version="12.1",
        gpu_count=2,
        gpu_name_substring="RTX 4090",
        minimum_vram_bytes=server_environment.NOMINAL_24_GB_BYTES,
        minimum_cpu_count=28,
        minimum_ram_bytes=224 * 1024**3,
    )
    facts = ServerEnvironmentFacts(
        python_version="3.10.20",
        python_minor=(3, 10),
        torch_version="2.2.2+cu121",
        cuda_version="12.1",
        gpu_names=("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"),
        gpu_vram_bytes=(24080 * 1024**2, 24080 * 1024**2),
        cpu_count=28,
        ram_bytes=224 * 1024**3,
    )
    monkeypatch.setattr(server_environment, "_collect_facts", lambda: facts)
    monkeypatch.setattr(server_environment, "_probe_cuda_device", lambda _device: True)

    receipt = validate_server_environment(expectation)

    assert server_environment.NOMINAL_24_GB_BYTES == 24_000_000_000
    assert receipt.gpu_vram_bytes == facts.gpu_vram_bytes
