"""Integration tests: OpenCVCameraService facade over a real OpenCV source.

Uses a synthetic AVI clip registered through a stub discovery provider —
validates discovery→connect→frame-delivery→disconnect wiring end to end
without a physical camera.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from camera.capture.opencv_service import OpenCVCameraService
from camera.interfaces.contracts import DeviceDiscovery
from camera.interfaces.types import (
    CameraError,
    CaptureDeviceInfo,
    ConnectionState,
    DeviceNotFoundError,
    Resolution,
    SourceKind,
)

WIDTH, HEIGHT, FPS, FRAME_COUNT = 320, 240, 30.0, 45
POLL_TIMEOUT_S = 10.0


def make_test_video(directory: Path) -> Path:
    path = directory / "service_clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        level = int((index / FRAME_COUNT) * 255)
        writer.write(np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8))
    writer.release()
    return path


def wait_until(predicate, timeout_s: float = POLL_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class StubDiscovery(DeviceDiscovery):
    """Discovery stub exposing the synthetic clip as the only device."""

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


class TestServiceLifecycle:
    def test_discovery_then_connect_streams_frames(self, video_path):
        frames, states = [], []

        with OpenCVCameraService(discovery=StubDiscovery(str(video_path))) as service:
            devices = service.list_devices()
            assert len(devices) == 1
            assert devices[0].id == str(video_path)

            unsub_frame = service.on_frame(frames.append)
            unsub_state = service.on_state_change(states.append)
            try:
                service.connect()  # default → first discovered device
                assert service.get_active_device_id() == str(video_path)

                assert wait_until(
                    lambda: service.get_state() is ConnectionState.CONNECTED
                ), f"states so far: {states}"
                assert wait_until(lambda: len(frames) >= 5)

                frame = frames[0]
                assert (frame.width, frame.height) == (WIDTH, HEIGHT)
                assert frame.stamp.monotonic_ns > 0
                assert frame.stamp.wallclock_ns > 0
                assert ConnectionState.CONNECTING in states
            finally:
                unsub_frame()
                unsub_state()

        # Context manager disconnected and released everything.
        assert service.get_state() is ConnectionState.DISCONNECTED
        assert service.get_active_device_id() is None

    def test_disconnect_is_idempotent_and_resets_state(self, video_path):
        with OpenCVCameraService(discovery=StubDiscovery(str(video_path))) as service:
            service.disconnect()  # no-op when idle
            assert service.get_state() is ConnectionState.DISCONNECTED

            service.connect(str(video_path))
            assert wait_until(
                lambda: service.get_state() is ConnectionState.CONNECTED
            )
            service.disconnect()
            assert service.get_state() is ConnectionState.DISCONNECTED
            assert service.get_active_device_id() is None

            service.disconnect()  # second call still safe

    def test_reconnect_creates_fresh_session(self, video_path):
        service = OpenCVCameraService(discovery=StubDiscovery(str(video_path)))
        try:
            service.connect(str(video_path))
            assert wait_until(
                lambda: service.get_state() is ConnectionState.CONNECTED
            )
            service.disconnect()

            service.connect(str(video_path))  # must not raise "already connected"
            assert wait_until(
                lambda: service.get_state() is ConnectionState.CONNECTED
            )
        finally:
            service.disconnect()

    def test_unopenable_source_surfaces_error_state(self, video_path):
        service = OpenCVCameraService(discovery=StubDiscovery(str(video_path)))
        try:
            service.connect("Z:/missing/nope.avi")
            assert wait_until(
                lambda: service.get_state() is ConnectionState.ERROR
            ), "expected async ERROR state"

            service.disconnect()
            assert service.get_state() is ConnectionState.DISCONNECTED
        finally:
            service.disconnect()

    def test_invalid_device_ids_raise_immediately(self, video_path):
        service = OpenCVCameraService(discovery=StubDiscovery(str(video_path)))
        try:
            with pytest.raises(DeviceNotFoundError):
                service.connect("")
            with pytest.raises((DeviceNotFoundError, CameraError, ValueError)):
                service.connect(12345)  # type: ignore[arg-type]
        finally:
            service.disconnect()


class TestListenerFanOut:
    def test_service_level_unsubscribe_stops_delivery(self, video_path):
        first_seen, second_seen = [], []
        service = OpenCVCameraService(discovery=StubDiscovery(str(video_path)))
        try:
            unsubscribe_first = service.on_frame(first_seen.append)
            service.on_frame(second_seen.append)
            service.connect(str(video_path))

            assert wait_until(lambda: len(second_seen) >= 8)
            count_at_unsub = len(first_seen)
            unsubscribe_first()

            def second_advanced():
                return len(second_seen) >= count_at_unsub + 4

            assert wait_until(second_advanced)
            assert len(first_seen) == count_at_unsub
        finally:
            service.disconnect()
