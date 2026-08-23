"""Camera gateway — the ONLY place where camera events meet transport.

Responsibilities (transport/orchestration only):
- encode frames to versioned binary messages (metadata header + JPEG)
- fan out state transitions to attached clients
- serialize camera link ownership across concurrent clients
- bridge camera reader-thread callbacks into the asyncio world

It never touches capture hardware and never re-implements camera logic.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from typing import Dict, Optional, Set

import cv2
from starlette.concurrency import run_in_threadpool

from camera.interfaces.contracts import CameraService, Unsubscribe
from camera.interfaces.types import (
    CameraError,
    ConnectionState,
    DeviceNotFoundError,
    Frame,
)

from ..models.messages import (
    PROTOCOL_VERSION,
    SERVER_VERSION,
    ProtocolError,
    device_to_dto,
    parse_client_message,
)

_BINARY_HEADER_STRUCT = struct.Struct("<I")


def _offer(queue: "asyncio.Queue", item) -> None:
    """Drop-oldest enqueue. Must run on the event-loop thread."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:  # pragma: no cover - maxsize>=1 makes this moot
        pass


def _json(payload: Dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def encode_frame_message(frame: Frame, jpeg_quality: int = 80) -> bytes:
    """[uint32 LE header length][JSON header][JPEG payload]"""
    if frame.image is None:
        header = {
            "seq": frame.stamp.sequence,
            "monoNs": frame.stamp.monotonic_ns,
            "wallNs": frame.stamp.wallclock_ns,
            "fps": frame.stamp.fps,
            "w": frame.width,
            "h": frame.height,
            "enc": "none",
        }
        payload = b""
    else:
        ok, buffer = cv2.imencode(
            ".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        payload = buffer.tobytes()
        header = {
            "seq": frame.stamp.sequence,
            "monoNs": frame.stamp.monotonic_ns,
            "wallNs": frame.stamp.wallclock_ns,
            "fps": frame.stamp.fps,
            "w": frame.width,
            "h": frame.height,
            "enc": "jpeg",
        }
    raw_header = _json(header).encode("utf-8")
    return (
        _BINARY_HEADER_STRUCT.pack(len(raw_header)) + raw_header + payload
    )


class ClientSession:
    """Per-socket outbound mailbox + ownership flag."""

    def __init__(self, websocket, loop: asyncio.AbstractEventLoop) -> None:
        self.websocket = websocket
        self.loop = loop
        self.outbox: "asyncio.Queue" = asyncio.Queue(maxsize=64)
        self.owns_link = False
        self.closed = False

    def publish_text(self, text: str) -> None:
        if self.closed:
            return
        self.loop.call_soon_threadsafe(_offer, self.outbox, ("text", text))

    def publish_bytes(self, data: bytes) -> None:
        if self.closed:
            return
        self.loop.call_soon_threadsafe(_offer, self.outbox, ("bytes", data))

    def close_mailbox(self) -> None:
        self.closed = True
        try:
            self.loop.call_soon_threadsafe(_offer, self.outbox, ("close", None))
        except RuntimeError:  # loop already shut down
            pass

    async def run_sender(self) -> None:
        while True:
            kind, item = await self.outbox.get()
            if kind == "close":
                return
            if kind == "text":
                await self.websocket.send_text(item)
            else:
                await self.websocket.send_bytes(item)


class CameraGateway:
    def __init__(self, service: CameraService, jpeg_quality: int = 80) -> None:
        self._service = service
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._sessions: Set[ClientSession] = set()
        self._owner: Optional[ClientSession] = None
        self._unsubscribers: list[Unsubscribe] = []
        self._attached = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, session: ClientSession) -> None:
        self._ensure_attached()
        with self._lock:
            self._sessions.add(session)

    def _ensure_attached(self) -> None:
        """Subscribe once to the shared camera service's event streams."""
        with self._lock:
            if self._attached:
                return
            self._attached = True
        self._unsubscribers.append(self._service.on_frame(self._on_frame))
        self._unsubscribers.append(
            self._service.on_state_change(self._on_state)
        )

    async def detach(self, session: ClientSession) -> None:
        """Socket gone: stop delivery and release the camera if owned."""
        with self._lock:
            was_owner = self._owner is session
            if was_owner:
                self._owner = None
            self._sessions.discard(session)
        session.close_mailbox()
        if was_owner and self._service.get_state() != ConnectionState.DISCONNECTED:
            await run_in_threadpool(self._service.disconnect)

    async def shutdown(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        with self._lock:
            self._owner = None
            sessions = tuple(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close_mailbox()

    # ------------------------------------------------------------------
    # Inbound dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, session: ClientSession, data: Dict) -> None:
        try:
            message = parse_client_message(data)
        except ProtocolError as exc:
            session.publish_text(_json({"type": "error", "code": "protocol", "message": str(exc)}))
            return

        kind = message.type  # type: ignore[union-attr]

        if kind == "hello":
            session.publish_text(
                _json(
                    {
                        "type": "hello.ack",
                        "protocolVersion": PROTOCOL_VERSION,
                        "serverVersion": SERVER_VERSION,
                    }
                )
            )
            session.publish_text(self.state_snapshot())
            return

        if kind == "devices.list":
            devices = await run_in_threadpool(self._service.list_devices)
            session.publish_text(
                _json(
                    {
                        "type": "devices.list.result",
                        "devices": [device_to_dto(d).model_dump() for d in devices],
                    }
                )
            )
            return

        if kind == "camera.connect":
            await self._handle_connect(session, message.deviceId)  # type: ignore[union-attr]
            return

        if kind == "camera.disconnect":
            await self._release(session)

    # ------------------------------------------------------------------
    # Connect / disconnect flows
    # ------------------------------------------------------------------

    async def _handle_connect(self, session: ClientSession, device_id: Optional[str]) -> None:
        with self._lock:
            owner = self._owner
        if owner is not None and owner is not session:
            session.publish_text(
                _json(
                    {
                        "type": "error",
                        "code": "camera_busy",
                        "message": "Another client currently owns the camera link.",
                    }
                )
            )
            return

        # Mock-parity: an existing own link is transparently recycled.
        if owner is session and self._service.get_state() != ConnectionState.DISCONNECTED:
            await run_in_threadpool(self._service.disconnect)

        with self._lock:
            self._owner = session
        session.owns_link = True
        try:
            await run_in_threadpool(self._service.connect, device_id)
        except DeviceNotFoundError as exc:
            self._clear_owner(session)
            session.publish_text(_json({"type": "error", "code": "device_not_found", "message": str(exc)}))
        except CameraError as exc:
            self._clear_owner(session)
            session.publish_text(_json({"type": "error", "code": "internal", "message": str(exc)}))
        # Subsequent CONNECTING/CONNECTED/ERROR states are broadcast from
        # the camera service's own listener callbacks.

    async def _release(self, session: ClientSession) -> None:
        with self._lock:
            if self._owner is not session:
                return  # not the owner → ignore silently
            self._owner = None
        session.owns_link = False
        await run_in_threadpool(self._service.disconnect)

    def _clear_owner(self, session: ClientSession) -> None:
        with self._lock:
            if self._owner is session:
                self._owner = None
        session.owns_link = False

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    def state_snapshot(self) -> str:
        service = self._service
        return _json(
            {
                "type": "camera.state",
                "state": service.get_state().value,
                "activeDeviceId": service.get_active_device_id(),
                "error": None,
            }
        )

    # ------------------------------------------------------------------
    # Camera-service callbacks (reader thread)
    # ------------------------------------------------------------------

    def _on_state(self, state: ConnectionState) -> None:
        payload = _json(
            {
                "type": "camera.state",
                "state": state.value,
                "activeDeviceId": self._service.get_active_device_id(),
                "error": None,
            }
        )
        with self._lock:
            sessions = tuple(self._sessions)
        for session in sessions:
            session.publish_text(payload)

    def _on_frame(self, frame: Frame) -> None:
        with self._lock:
            owner = self._owner
        if owner is None or not owner.owns_link or owner.closed:
            return  # nobody subscribed to frames right now
        try:
            payload = encode_frame_message(frame, self._jpeg_quality)
        except Exception:
            return
        owner.publish_bytes(payload)
