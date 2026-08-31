import json
from pathlib import Path

from fastapi.testclient import TestClient

from tarca.monitoring.api import create_monitoring_app
from tests.monitoring.conftest import monitoring_database as monitoring_database

FORBIDDEN_KEYS = {"crps", "nll", "mae", "coverage", "ranking", "best_seed", "skill"}


def test_running_e02_api_never_exposes_partial_science(
    monitoring_database: Path, tmp_path: Path
) -> None:
    client = TestClient(
        create_monitoring_app(monitoring_database, tmp_path / "static", "e02-v1")
    )
    assert client.get("/api/v1/runtime").json()["execution_kind"] == "e02-v1"
    for endpoint in ("run", "jobs", "resources", "alerts"):
        serialized = json.dumps(client.get(f"/api/v1/{endpoint}").json(), sort_keys=True).lower()
        assert not any(key in serialized for key in FORBIDDEN_KEYS)
