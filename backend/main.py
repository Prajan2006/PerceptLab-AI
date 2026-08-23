"""FastAPI application factory — transport/orchestration ONLY.

Camera capture logic lives exclusively in the ``camera`` package; this
layer composes it, exposes the versioned WebSocket protocol, and manages
connection lifecycle.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from .api import devices_router, health_router, ws_router
from .core.container import Container
from .core.settings import Settings
from .models.messages import SERVER_VERSION
from .services.camera_gateway import CameraGateway


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    container = container or Container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gateway = CameraGateway(container.camera_service, settings.jpeg_quality)
        app.state.container = container
        app.state.camera_gateway = gateway
        try:
            yield
        finally:
            # Clean shutdown: stop streaming, release any held camera.
            await gateway.shutdown()
            await run_in_threadpool(container.camera_service.disconnect)

    app = FastAPI(
        title="PerceptLab AI Backend",
        version=SERVER_VERSION,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(devices_router)
    app.include_router(ws_router)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run(app, host=settings.host, port=settings.port)
