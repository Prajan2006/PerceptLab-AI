"""HTTP/REST endpoints (health, discovery convenience)."""

from .routes_devices import router as devices_router
from .routes_health import router as health_router
from .ws_camera import router as ws_router

__all__ = ["devices_router", "health_router", "ws_router"]
