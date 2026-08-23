"""Unit tests for the threaded OpenCV capture session.

Uses a synthetic AVI clip as the frame source, so the full
open → read → stamp → deliver → release pipeline is exercised without a
physical camera.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from camera.capture.opencv_capture import OpenCVCaptureSession, parse_source_descriptor
from camera.interfaces.types import ConnectionState, SourceKind

WIDTH, HEIGHT, FPS, FRAME_COUNT = 320, 240, 30.0, 45

POLL_TIMEOUT_S = 10.0


def make_test_video(directory: Path) -> Path:
    """Write a short synthetic clip; brightness ramps so frames differ."""
    path = directory / "synthetic_clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "VideoWriter failed to open (codec issue)"
    for index in range(FRAME_COUNT):
        level = int((index / FRAME_COUNT) * 255)
        frame = np.full((HEIGHT, WIDTH, 3), level, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def wait_until(predicate, timeout_s: float = POLL_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class TestSourceDescriptorParsing:
    def test_opencv_index_id_resolves_to_int(self):
        assert parse_source_descriptor("opencv:2") == 2

    def test_other_ids_treated_as_file_paths(self):
        assert parse_source_descriptor(r"C:\clips\a.avi") == r"C:\clips\a.avi"

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError):
            parse_source_descriptor("")

    def test_malformed_index_rejected(self):
        with pytest.raises(ValueError):
            parse_source_descriptor("opencv:not-a-number")


class TestOpenCVCaptureSession:
    @pytest.fixture()
    def video_path(self, tmp_path: Path) -> Path:
        return make_test_video(tmp_path)

    def test_full_lifecycle_with_stamped_frames(self, video_path):
        states = []
        frames = []

        session = OpenCVCaptureSession(str(video_path))
        unsub_state = session.on_state_change(states.append)
        unsub_frame = session.on_frame(frames.append)

        session.start()
        try:
            assert wait_until(lambda: len(frames) >= 10), (
                f"only {len(frames)} frames arrived"
            )

            # --- state transitions -------------------------------
            assert states[0] is ConnectionState.CONNECTING
            assert ConnectionState.CONNECTED in states

            # --- sequence numbers --------------------------------
            sequences = [frame.stamp.sequence for frame in frames[:10]]
            assert sequences == list(range(1, 11))

            # --- timestamps --------------------------------------
            monotonic = [frame.stamp.monotonic_ns for frame in frames[:10]]
            assert all(b > a for a, b in zip(monotonic, monotonic[1:])), (
                "monotonic stamps must strictly increase"
            )
            wallclock = [frame.stamp.wallclock_ns for frame in frames[:10]]
            assert wallclock == sorted(wallclock)

            # --- fps + resolution metadata -----------------------
            recent_fps = [frame.stamp.fps for frame in frames if frame.stamp.fps > 0]
            assert recent_fps, "fps never became positive"
            assert max(recent_fps) < 1000.0
            sample = frames[5]
            assert (sample.width, sample.height) == (WIDTH, HEIGHT)
            assert sample.image is not None
            assert sample.image.shape == (HEIGHT, WIDTH, 3)
            assert session.get_last_error() is None
        finally:
            unsub_state()
            unsub_frame()
            session.stop()

        # --- clean shutdown --------------------------------------
        assert not session.is_running()
        assert session.get_state() is ConnectionState.DISCONNECTED

    def test_error_state_for_unopenable_source(self):
        session = OpenCVCaptureSession("Z:/definitely/missing/clip.avi")
        errors = []
        session.on_state_change(
            lambda state: errors.append(state) if state is ConnectionState.ERROR else None
        )

        session.start()  # open happens on reader thread → async ERROR
        try:
            assert wait_until(lambda: session.get_state() is ConnectionState.ERROR)
            assert session.get_last_error() is not None
        finally:
            session.stop()

        assert session.get_state() is ConnectionState.ERROR  # preserved for diagnosis

    def test_double_start_rejected(self, video_path):
        session = OpenCVCaptureSession(str(video_path))
        session.start()
        try:
            assert wait_until(session.is_running)
            from camera.interfaces.types import CameraError

            with pytest.raises(CameraError):
                session.start()
        finally:
            session.stop()

    def test_unsubscribe_stops_delivery(self, video_path):
        first_seen, second_seen = [], []
        session = OpenCVCaptureSession(str(video_path))
        unsubscribe_first = session.on_frame(first_seen.append)
        session.on_frame(second_seen.append)

        session.start()
        try:
            assert wait_until(lambda: len(second_seen) >= 8)
            count_at_unsub = len(first_seen)
            unsubscribe_first()

            def second_advanced():
                return len(second_seen) >= count_at_unsub + 4

            assert wait_until(second_advanced)
            assert len(first_seen) == count_at_unsub
        finally:
            session.stop()

    def test_context_manager_releases(self, video_path):
        with OpenCVCaptureSession(str(video_path)) as session:
            assert wait_until(lambda: session.get_state() is ConnectionState.CONNECTED)
        assert not session.is_running()
