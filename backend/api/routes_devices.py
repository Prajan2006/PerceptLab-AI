from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from ..models.messages import device_to_dto

router = APIRouter(tags=["devices"])


@router.get("/api/devices")
async def list_devices(request: Request) -> dict:
    """REST convenience mirror of the WS ``devices.list`` message."""
    service = request.app.state.container.camera_service
    devices = await run_in_threadpool(service.list_devices)
    return {"type": "devices.list.result", "devices": [device_to_dto(d).model_dump() for d in devices]}
