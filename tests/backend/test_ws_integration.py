"""End-to-end WebSocket transport tests against the real camera service.

The camera service is backed by a synthetic AVI clip (no physical camera
required), registered through a stub discovery provider — exercising the
full path: WS protocol → gateway → CameraService facade → OpenCV session.
"""

import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.container import Container
from backend.core.settings import Settings
from backend.main import create_app
from camera.capture.opencv_service import OpenCVCameraService
from camera.interfaces.contracts import DeviceDiscovery
from camera.interfaces.types import (
    CaptureDeviceInfo,
    ConnectionState,
    Resolution,
    SourceKind,
)

WIDTH, HEIGHT, FPS, FRAME_COUNT = 320, 240, 30.0, 45


def make_test_video(directory: Path) -> Path:
    path = directory / "transport_clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        level = int((index / FRAME_COUNT) * 255)
        writer.write(np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8))
    writer.release()
    return path


class StubDiscovery(DeviceDiscovery):
    def __init__(self, device_id: str) -> None:
        self._device_id = device_id

    def list_devices(self):
        return [
            CaptureDeviceInfo(
                id=self._device_id,
                label="Synthetic clip",
                kind=SourceKind.VIDEO_FILE,
                resolution=Resolution(WIDTH, HEIGHT),
                max_fps=FPS,
            )
        ]


@pytest.fixture()
def video_path(tmp_path: Path) -> Path:
    return make_test_video(tmp_path)


@pytest.fixture()
def app(video_path):
    settings = Settings()
    container = Container(settings)
    container.override_camera_service(
        OpenCVCameraService(discovery=StubDiscovery(str(video_path)))
    )
    return create_app(settings=settings, container=container)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hello(ws) -> dict:
    ws.send_text(json.dumps({"type": "hello"}))
    ack = ws.receive_json()
    assert ack["type"] == "hello.ack"
    snapshot = _next_state_json(ws)
    assert snapshot["type"] == "camera.state"
    return snapshot


def _next_state_json(ws) -> dict:
    """Next control message, skipping any interleaved binary frame data."""
    while True:
        message = ws.receive()
        if "text" in message:
            return json.loads(message["text"])


def wait_state(ws, wanted: str, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        data = _next_state_json(ws)
        if data.get("type") == "camera.state":
            last = data
            if data["state"] == wanted:
                return data
    raise AssertionError(f"Never reached state {wanted!r}; last={last}")


def parse_frame(blob: bytes):
    (header_length,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + header_length].decode("utf-8"))
    payload = blob[4 + header_length :]
    return header, payload


def collect_frames(ws, count: int, timeout_s: float = 15.0):
    frames = []
    deadline = time.monotonic() + timeout_s
    while len(frames) < count:
        if time.monotonic() > deadline:
            raise AssertionError(f"Only {len(frames)}/{count} frames arrived.")
        message = ws.receive()
        data = message.get("bytes")
        assert data is not None, f"Expected binary frame, got: {message}"
        frames.append(parse_frame(data))
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHttpEndpoints:
    def test_health_reports_protocol_version(self, app):
        with TestClient(app) as client:
            body = client.get("/api/health").json()
            assert body["status"] == "ok"
            assert body["protocolVersion"] == 1

    def test_rest_device_discovery(self, app, video_path):
        with TestClient(app) as client:
            body = client.get("/api/devices").json()
            ids = [device["id"] for device in body["devices"]]
            assert str(video_path) in ids


class TestWsLifecycle:
    def test_handshake_delivers_authoritative_snapshot(self, app):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                snapshot = hello(ws)
                assert snapshot["state"] == "disconnected"
                assert snapshot["activeDeviceId"] is None

    def test_discovery_over_websocket(self, app, video_path):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(json.dumps({"type": "devices.list"}))
                result = ws.receive_json()
                assert result["type"] == "devices.list.result"
                device = result["devices"][0]
                assert device["id"] == str(video_path)
                assert device["kind"] == "video_file"
                assert device["resolution"] == {"width": WIDTH, "height": HEIGHT}
                assert device["maxFps"] == FPS

    def test_frames_arrive_ordered_with_full_metadata(self, app, video_path):
        started_ns = time.time_ns()

        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                connecting = wait_state(ws, "connecting")
                assert connecting["activeDeviceId"] == str(video_path)
                connected = wait_state(ws, "connected")

                frames = collect_frames(ws, 12)

                sequences = [header["seq"] for header, _ in frames]
                assert sequences == list(range(1, 13)), "frame ordering must be preserved"

                mono_values = [header["monoNs"] for header, _ in frames]
                assert all(b > a for a, b in zip(mono_values, mono_values[1:])), (
                    "monotonic timestamps must strictly increase"
                )

                wall_values = [header["wallNs"] for header, _ in frames]
                assert wall_values == sorted(wall_values)
                assert all(value >= started_ns for value in wall_values)

                fps_values = [header["fps"] for header, _ in frames]
                positive = [value for value in fps_values if value > 0]
                assert positive and max(fps_values) < 1000.0

                for header, payload in frames:
                    assert header["enc"] == "jpeg"
                    assert (header["w"], header["h"]) == (WIDTH, HEIGHT)
                    assert payload[:2] == b"\xff\xd8"
                    # Wire payload must be genuinely decodable image data.
                    decoded = cv2.imdecode(
                        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    assert decoded is not None and decoded.shape == (
                        HEIGHT,
                        WIDTH,
                        3,
                    ), "frame payload must decode back to the source geometry"

                ws.send_text(json.dumps({"type": "camera.disconnect"}))
                final = wait_state(ws, "disconnected")
                assert final["activeDeviceId"] is None
                assert connected["activeDeviceId"] == str(video_path)

    def test_error_state_reaches_client(self, app):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(
                    json.dumps(
                        {"type": "camera.connect", "deviceId": "Z:/missing/nope.avi"}
                    )
                )
                error_message = wait_state(ws, "error")
                assert error_message["activeDeviceId"] == "Z:/missing/nope.avi"

                ws.send_text(json.dumps({"type": "camera.disconnect"}))
                final = wait_state(ws, "disconnected")
                assert final["activeDeviceId"] is None

    def test_second_client_is_busy_until_first_releases(self, app, video_path):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as first:
                hello(first)
                first.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                wait_state(first, "connected")

                with client.websocket_connect("/api/camera/ws") as second:
                    hello(second)
                    second.send_text(json.dumps({"type": "camera.connect"}))

                    deadline = time.monotonic() + 10.0
                    busy_seen = False
                    while time.monotonic() < deadline and not busy_seen:
                        message = second.receive()
                        if "text" in message:
                            data = json.loads(message["text"])
                            if (
                                data.get("type") == "error"
                                and data.get("code") == "camera_busy"
                            ):
                                busy_seen = True
                        # Binary frames must never reach a non-owner; if they
                        # do, the assertion below catches it.
                        else:
                            assert message.get("bytes") is None, (
                                "non-owner received binary frame data"
                            )
                    assert busy_seen, "second client must be rejected with camera_busy"

                    first.send_text(json.dumps({"type": "camera.disconnect"}))
                    wait_state(first, "disconnected")

                    # Ownership released → second client can now take over.
                    second.send_text(
                        json.dumps(
                            {"type": "camera.connect", "deviceId": str(video_path)}
                        )
                    )
                    wait_state(second, "connected")

    def test_abrupt_client_disconnect_releases_camera(self, app, video_path):
        with TestClient(app) as client:
            service = app.state.container.camera_service
            with client.websocket_connect("/api/camera/ws") as socket:
                hello(socket)
                socket.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                wait_state(socket, "connected")
                collect_frames(socket, 3)
                socket.close()  # abrupt — no camera.disconnect command

            deadline = time.monotonic() + 5.0
            while (
                time.monotonic() < deadline
                and service.get_state() is not ConnectionState.DISCONNECTED
            ):
                time.sleep(0.05)

        assert service.get_state() is ConnectionState.DISCONNECTED
        assert service.get_active_device_id() is None

    def test_reconnect_after_release_streams_again(self, app, video_path):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                wait_state(ws, "connected")
                collect_frames(ws, 4)
                ws.send_text(json.dumps({"type": "camera.disconnect"}))
                wait_state(ws, "disconnected")

            # Fresh socket → fresh lifecycle.
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                wait_state(ws, "connected")
                frames = collect_frames(ws, 4)
                assert [header["seq"] for header, _ in frames] == [1, 2, 3, 4]

    def test_malformed_input_gets_protocol_error(self, app):
        with TestClient(app) as client:
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text("{not-json")
                message = ws.receive_json()
                assert message["type"] == "error" and message["code"] == "protocol"

                ws.send_text(json.dumps({"type": "unknown.thing"}))
                message = ws.receive_json()
                assert message["type"] == "error" and message["code"] == "protocol"

    def test_clean_server_shutdown_disconnects_camera(self, app, video_path):
        service_holder = {}

        with TestClient(app) as client:
            service_holder["service"] = app.state.container.camera_service
            with client.websocket_connect("/api/camera/ws") as ws:
                hello(ws)
                ws.send_text(
                    json.dumps({"type": "camera.connect", "deviceId": str(video_path)})
                )
                wait_state(ws, "connected")

        assert service_holder["service"].get_state() is ConnectionState.DISCONNECTED
