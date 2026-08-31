import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/stage2"


def test_dockerfile_preserves_base_cuda_torch() -> None:
    text = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
    assert text.startswith("FROM node:20-bookworm-slim AS ui-build")
    assert "FROM pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04" in text
    assert "pip install torch" not in text
    assert "USER tarca" in text
    assert "--require-hashes" in text


def test_compose_is_read_only_loopback_and_requests_all_gpus() -> None:
    text = (DEPLOY / "compose.yaml").read_text(encoding="utf-8")
    assert "read_only: true" in text
    assert "gpus: all" in text
    assert "127.0.0.1:8765:8765" in text
    assert "shm_size: 16gb" in text
    assert "/tmp:rw,noexec,nosuid" in text


def test_server_lock_contains_no_torch_reinstallation() -> None:
    lock = (DEPLOY / "requirements-server.lock").read_text(encoding="utf-8")
    assert "torch==" not in lock
    assert all("--hash=sha256:" in line for line in lock.splitlines() if line.strip())


def test_server_lock_excludes_audited_vulnerable_runtime_versions() -> None:
    lock = (DEPLOY / "requirements-server.lock").read_text(encoding="utf-8")
    assert "fastapi==0.141.1" in lock
    assert "starlette==1.6.0" in lock
    assert "pyarrow==25.0.1" in lock
    assert "annotated-doc==0.0.5" in lock
    assert "starlette==0.47.3" not in lock
    assert "pyarrow==20.0.0" not in lock


def test_official_itransformer_dependency_is_locked_for_offline_install() -> None:
    requested = (DEPLOY / "requirements-server.in").read_text(encoding="utf-8")
    lock = (DEPLOY / "requirements-server.lock").read_text(encoding="utf-8")
    wheel = DEPLOY / "wheelhouse/einops-0.8.2-py3-none-any.whl"

    assert "einops==0.8.2" in requested
    assert (
        "einops==0.8.2 "
        "--hash=sha256:54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193"
    ) in lock
    assert wheel.is_file()
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == (
        "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193"
    )
