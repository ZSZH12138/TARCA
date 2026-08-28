from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "deploy/stage1b/Dockerfile"
ENTRYPOINT = REPOSITORY_ROOT / "deploy/stage1b/entrypoint.sh"
COMPOSE_PATH = REPOSITORY_ROOT / "deploy/stage1b/compose.stage1b-v2.yaml"


def test_monitor_port_is_host_loopback_only() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))

    assert compose["services"]["stage1b"]["ports"] == ["127.0.0.1:8765:8765"]


def test_container_uses_exact_authorized_base_and_all_visible_gpus() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose["services"]["stage1b"]

    assert "FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime" in dockerfile
    assert "pip install --require-hashes" in dockerfile
    assert "COPY --from=ui-build" in dockerfile
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
    assert service["environment"]["TARCA_STAGE1B_QUALIFICATION_CONFIG"].endswith(
        "qualification_v2_confirmation_r2.yaml"
    )
    assert service["shm_size"] == "16gb"


def test_runtime_has_no_docker_socket_or_public_port_and_sources_are_read_only() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose["services"]["stage1b"]
    serialized = repr(service).lower()

    assert "docker.sock" not in serialized
    assert "0.0.0.0:8765" not in serialized
    source_mount = next(
        item for item in service["volumes"] if item["target"] == "/opt/tarca/official_sources"
    )
    assert source_mount["read_only"] is True


def test_entrypoint_forwards_signals_and_monitor_is_read_only() -> None:
    script = ENTRYPOINT.read_text(encoding="utf-8")

    assert "trap terminate TERM INT" in script
    assert "tarca.monitoring.server:create_app_from_environment" in script
    assert "exec python scripts/run_stage1b_runtime.py" in script


def test_fresh_server_imports_a_local_capsule_into_a_separate_writable_source_volume() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert "stage1b-source-init" not in compose["services"]
    importer = compose["services"]["stage1b-source-import"]

    assert importer["entrypoint"] == [
        "python",
        "scripts/import_stage1b_source_capsule.py",
        "--cache-root",
        "/opt/tarca/official_sources",
    ]
    source_mount = next(
        item for item in importer["volumes"] if item["target"] == "/opt/tarca/official_sources"
    )
    assert source_mount["read_only"] is False
    transfer_mount = next(
        item for item in importer["volumes"] if item["target"] == "/opt/tarca/source-transfer"
    )
    assert transfer_mount["read_only"] is True
    assert compose["services"]["stage1b"]["environment"]["TARCA_STAGE1B_SOURCE_MODE"] == (
        "offline-capsule"
    )
    assert "deploy" not in importer
    assert "ports" not in importer
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "apt-get install" in dockerfile
    assert "git" in dockerfile and "ca-certificates" in dockerfile
    assert "TARCA_STAGE1B_SOURCE_MODE=offline-capsule" in dockerfile
