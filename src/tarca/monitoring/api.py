from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tarca.monitoring.repository import MonitoringRepository
from tarca.monitoring.schemas import (
    AlertView,
    JobStatusView,
    ResourceView,
    RunSummaryView,
)


def create_monitoring_app(database_path: Path, static_root: Path) -> FastAPI:
    repository = MonitoringRepository(database_path)
    app = FastAPI(
        title="TARCA Stage1B Runtime Monitor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/api/v1/run", response_model=RunSummaryView)
    def get_run() -> RunSummaryView:
        return repository.snapshot().run

    @app.get("/api/v1/jobs", response_model=tuple[JobStatusView, ...])
    def get_jobs() -> tuple[JobStatusView, ...]:
        return repository.snapshot().jobs

    @app.get("/api/v1/resources", response_model=tuple[ResourceView, ...])
    def get_resources() -> tuple[ResourceView, ...]:
        return repository.snapshot().resources

    @app.get("/api/v1/alerts", response_model=tuple[AlertView, ...])
    def get_alerts() -> tuple[AlertView, ...]:
        return repository.snapshot().alerts

    @app.websocket("/api/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(repository.snapshot().model_dump_json())
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            return

    @app.api_route(
        "/api/{path:path}",
        methods=["POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def reject_mutation(path: str) -> JSONResponse:
        del path
        raise HTTPException(status_code=405, detail="monitoring API is read-only")

    resolved_static = static_root.resolve()
    if resolved_static.is_dir():
        app.mount("/", StaticFiles(directory=resolved_static, html=True), name="dashboard")
    else:
        @app.get("/", include_in_schema=False)
        def no_dashboard() -> JSONResponse:
            return JSONResponse({"status": "monitoring-api-only"})

    return app
