from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from tarca.monitoring.api import create_monitoring_app


def create_app_from_environment() -> FastAPI:
    database = Path(
        os.environ.get("TARCA_RUNTIME_DATABASE")
        or os.environ.get(
            "TARCA_STAGE1B_DATABASE", "/opt/tarca/artifacts/stage1b/runtime/execution.sqlite3"
        )
    )
    static_root = Path(
        os.environ.get("TARCA_RUNTIME_STATIC_ROOT")
        or os.environ.get("TARCA_STAGE1B_STATIC_ROOT", "/opt/tarca/frontend/stage1b-monitor/dist")
    )
    return create_monitoring_app(database, static_root)
