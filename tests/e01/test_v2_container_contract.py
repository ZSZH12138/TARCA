from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "deploy/e01/Dockerfile.v2"
COMPOSE = ROOT / "deploy/e01/compose.e01-v2.yaml"
ENTRYPOINT = ROOT / "deploy/e01/entrypoint-v2.sh"
BOOTSTRAP = ROOT / "deploy/e01/server_bootstrap_v2.sh"
SUPERVISOR = ROOT / "deploy/e01/server_supervisor_v2.sh"
FRONTEND_DIST = ROOT / "frontend/stage1b-monitor/dist"


def test_v2_container_uses_exact_image_single_gpu_and_host_bound_artifacts() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["e01-v2"]

    assert "FROM pytorch:2.2.2-cuda12.1-cudnn8-py310-ubuntu22.04" in dockerfile
    assert service["build"]["dockerfile"] == "deploy/e01/Dockerfile.v2"
    assert service["shm_size"] == "16gb"
    assert service["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}
    ]
    artifact = next(
        item for item in service["volumes"] if item["target"] == "/opt/tarca/artifacts/e01-v2"
    )
    assert artifact["type"] == "bind"
    assert "TARCA_E01_V2_ARTIFACT_DIR" in artifact["source"]
    assert artifact.get("read_only", False) is False
    assert "docker.sock" not in repr(service).lower()


def test_v2_monitor_is_loopback_only_and_runtime_is_independently_supervised() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["e01-v2"]
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")

    assert service["ports"] == ["127.0.0.1:8765:8765"]
    assert service["environment"]["TARCA_EXECUTION_KIND"] == "e01-v2"
    assert service["entrypoint"] == ["/opt/tarca/deploy/e01/entrypoint-v2.sh"]
    assert "trap terminate TERM INT" in entrypoint
    assert "tarca.monitoring.server:create_app_from_environment" in entrypoint
    assert "scripts/run_e01_v2.py" in entrypoint
    assert 'case "${1:-}" in' in supervisor
    assert "launch|resume)" in supervisor
    assert "nohup" in supervisor
    assert "http://127.0.0.1:8765/api/v1/run" in supervisor
    assert "urlopen" in supervisor
    assert 'wait "${runtime_pid}"' in supervisor
    assert "codex" not in supervisor.lower()


def test_v2_bootstrap_runs_only_prepare_dry_run_and_bounded_preflight() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")

    assert "scripts/run_e01_v2.py" in script
    assert "prepare" in script and "dry-run" in script and "preflight" in script
    assert "--remaining-rental-hours" in script
    assert "import_stage1b_source_capsule.py" not in script
    assert "I_ACKNOWLEDGE_E01_V2_FORMAL_RUN" not in script
    assert "TARCA_E01_V2_CONFIG" in script


def test_v2_scripts_use_python310_compatible_fallback() -> None:
    for path in (ENTRYPOINT, BOOTSTRAP, SUPERVISOR):
        script = path.read_text(encoding="utf-8")
        assert '[[ -x "/opt/conda/bin/python" ]]' in script
        assert "command -v python" in script
        assert "command -v python3" in script
        assert 'tarca_python="$(command -v python3)"' in script
        assert '"${tarca_python}"' in script


def test_prebuilt_direct_mode_frontend_loads_runtime_identity_from_api() -> None:
    javascript = tuple((FRONTEND_DIST / "assets").glob("*.js"))

    assert javascript
    assert any(b"/api/v1/runtime" in path.read_bytes() for path in javascript)
