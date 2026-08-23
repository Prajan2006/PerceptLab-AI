"""WebSocket endpoint for camera control + frame streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.camera_gateway import CameraGateway, ClientSession

router = APIRouter(tags=["camera"])

_MAX_TEXT_MESSAGE = 64 * 1024


@router.websocket("/api/camera/ws")
async def camera_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    gateway: CameraGateway = websocket.app.state.camera_gateway
    loop = asyncio.get_running_loop()
    session = ClientSession(websocket, loop)
    gateway.attach(session)

    sender_task = asyncio.create_task(session.run_sender())
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > _MAX_TEXT_MESSAGE:
                session.publish_text(
                    json.dumps(
                        {"type": "error", "code": "protocol", "message": "Message too large."}
                    )
                )
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                session.publish_text(
                    json.dumps({"type": "error", "code": "protocol", "message": "Invalid JSON."})
                )
                continue
            await gateway.handle_message(session, data)
    except WebSocketDisconnect:
        pass
    finally:
        # Releases the camera when this client owned it (requirement: client
        # disconnect must free capture resources).
        await gateway.detach(session)
        sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
