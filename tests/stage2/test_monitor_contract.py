import json
from pathlib import Path

from fastapi.testclient import TestClient

from tarca.monitoring.api import create_monitoring_app
from tests.monitoring.conftest import monitoring_database as monitoring_database

FORBIDDEN_KEYS = {"crps", "nll", "mae", "coverage", "ranking", "best_seed", "skill"}


def test_running_stage2_api_is_read_only_and_science_blind(
    monitoring_database: Path, tmp_path: Path
) -> None:
    client = TestClient(
        create_monitoring_app(monitoring_database, tmp_path / "static", "stage2-v1")
    )
    assert client.get("/api/v1/runtime").json()["display_label"] == "Stage 2 v1"
    payload = json.dumps(client.get("/api/v1/run").json(), sort_keys=True).lower()
    assert not any(key in payload for key in FORBIDDEN_KEYS)
    assert client.post("/api/v1/jobs/x/restart").status_code == 405
