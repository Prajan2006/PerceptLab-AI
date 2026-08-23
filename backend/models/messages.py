"""Versioned WebSocket message protocol (v1).

Wire format
-----------
Control messages: JSON text frames, discriminated by ``type``.
Frame messages:   single binary WebSocket message per frame:

    [uint32 LE header-length][UTF-8 JSON header][JPEG payload]

Header keys preserve the camera module's frame metadata verbatim:
``seq``, ``mono_ns``, ``wall_ns``, ``fps``, ``w``, ``h``, ``enc``.
Field naming here is camelCase so the JSON maps one-to-one onto the
frontend's TypeScript contracts.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

PROTOCOL_VERSION = 1
SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Shared DTOs
# ---------------------------------------------------------------------------


class ResolutionDto(BaseModel):
    width: int
    height: int


class DeviceDto(BaseModel):
    id: str
    label: str
    kind: str = "camera"
    resolution: Optional[ResolutionDto] = None
    maxFps: Optional[float] = None
    backend: Optional[str] = None


class FrameHeader(BaseModel):
    seq: int
    monoNs: int
    wallNs: int
    fps: float
    w: int
    h: int
    enc: str = "jpeg"


# ---------------------------------------------------------------------------
# Client → Server
# ---------------------------------------------------------------------------


class HelloMessage(BaseModel):
    type: Literal["hello"]
    client: Optional[str] = None


class DeviceListRequest(BaseModel):
    type: Literal["devices.list"]


class CameraConnectMessage(BaseModel):
    type: Literal["camera.connect"]
    deviceId: Optional[str] = None


class CameraDisconnectMessage(BaseModel):
    type: Literal["camera.disconnect"]


ClientMessage = Union[
    HelloMessage,
    DeviceListRequest,
    CameraConnectMessage,
    CameraDisconnectMessage,
]


class ProtocolError(ValueError):
    """Raised for malformed/unrecognized inbound messages."""


def parse_client_message(payload: dict) -> ClientMessage:
    if not isinstance(payload, dict):
        raise ProtocolError("Message must be a JSON object.")
    msg_type = payload.get("type")
    parsers = {
        "hello": HelloMessage,
        "devices.list": DeviceListRequest,
        "camera.connect": CameraConnectMessage,
        "camera.disconnect": CameraDisconnectMessage,
    }
    parser = parsers.get(msg_type)
    if parser is None:
        raise ProtocolError(f"Unknown message type: {msg_type!r}")
    try:
        return parser.model_validate(payload)
    except Exception as exc:
        raise ProtocolError(f"Invalid {msg_type!r} message: {exc}") from exc


# ---------------------------------------------------------------------------
# Server → Client
# ---------------------------------------------------------------------------


class HelloAckMessage(BaseModel):
    type: Literal["hello.ack"]
    protocolVersion: int
    serverVersion: str


class CameraStateMessage(BaseModel):
    type: Literal["camera.state"]
    state: str  # 'disconnected' | 'connecting' | 'connected' | 'error'
    activeDeviceId: Optional[str] = None
    error: Optional[str] = None


class DeviceListResultMessage(BaseModel):
    type: Literal["devices.list.result"]
    devices: List[DeviceDto]


class ErrorMessage(BaseModel):
    type: Literal["error"]
    code: str  # 'protocol' | 'camera_busy' | 'device_not_found' | 'internal'
    message: str


# ---------------------------------------------------------------------------
# Camera-module DTO mapping
# ---------------------------------------------------------------------------


def device_to_dto(info) -> DeviceDto:
    """Map ``camera.interfaces.types.CaptureDeviceInfo`` to the wire DTO."""
    return DeviceDto(
        id=info.id,
        label=info.label,
        kind=info.kind.value,
        resolution=(
            ResolutionDto(width=info.resolution.width, height=info.resolution.height)
            if info.resolution is not None
            else None
        ),
        maxFps=info.max_fps,
        backend=info.backend or None,
    )
