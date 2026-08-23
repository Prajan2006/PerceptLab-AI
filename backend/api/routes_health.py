import time

from fastapi import APIRouter

from ..models.messages import PROTOCOL_VERSION, SERVER_VERSION

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "server": "perceptlab-backend",
        "protocolVersion": PROTOCOL_VERSION,
        "serverVersion": SERVER_VERSION,
        "timeNs": time.time_ns(),
    }
