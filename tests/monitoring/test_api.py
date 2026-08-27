from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tarca.monitoring.api import create_monitoring_app


def test_api_rejects_mutation(monitoring_database: Path, tmp_path: Path) -> None:
    client = TestClient(create_monitoring_app(monitoring_database, tmp_path / "static"))

    assert client.post("/api/v1/jobs/task-1/restart").status_code == 405
    assert client.put("/api/v1/jobs/task-1").status_code == 405
    assert client.patch("/api/v1/jobs/task-1").status_code == 405
    assert client.delete("/api/v1/jobs/task-1").status_code == 405


def test_api_excludes_scientific_fields(monitoring_database: Path, tmp_path: Path) -> None:
    client = TestClient(create_monitoring_app(monitoring_database, tmp_path / "static"))

    for endpoint in ("run", "jobs", "resources", "alerts"):
        response = client.get(f"/api/v1/{endpoint}")
        assert response.status_code == 200
        payload = json.dumps(response.json()).lower()
        for forbidden in ("crps", "nll", "mae", "truth", "ranking", "best_seed"):
            assert forbidden not in payload


def test_websocket_streams_the_same_safe_snapshot(
    monitoring_database: Path,
    tmp_path: Path,
) -> None:
    client = TestClient(create_monitoring_app(monitoring_database, tmp_path / "static"))

    with client.websocket_connect("/api/v1/stream") as websocket:
        payload = websocket.receive_json()

    assert payload["run"]["run_id"] == "run-a"
    assert payload["run"]["total_tasks"] == 2
    assert "last_sampled_at_utc" in payload["run"]
    assert payload["jobs"][0]["task_id"] == "task-1"
    assert "telemetry_status" in payload["resources"][0]
    assert "truth" not in json.dumps(payload).lower()


def test_static_serving_cannot_escape_the_approved_root(
    monitoring_database: Path,
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("dashboard", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    client = TestClient(create_monitoring_app(monitoring_database, static))

    assert client.get("/").text == "dashboard"
    escaped = client.get("/../secret.txt")
    assert escaped.status_code == 404
    assert "secret" not in escaped.text
