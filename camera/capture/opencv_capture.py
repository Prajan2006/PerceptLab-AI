"""Threaded OpenCV capture session with research-grade frame timing.

Design notes
------------
- A dedicated reader thread drains the backend buffer continuously, which
  keeps latency low on Media Foundation / DirectShow cameras.
- Every frame is stamped at grab time:
    * monotonic_ns  — ``time.perf_counter_ns()`` right after the read
      returns; basis for interval/FFT work in future rPPG analysis.
    * wallclock_ns  — ``time.time_ns()`` at the same instant.
    * fps           — EMA of inter-frame intervals on the same clock.
- File sources are paced to their nominal FPS so playback behaves like a
  live stream and tests remain deterministic.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Union

import cv2
import numpy as np

from ..interfaces.contracts import (
    CaptureSession,
    FrameListener,
    ListenerSet,
    StateListener,
    Unsubscribe,
    notify_listeners,
)
from ..interfaces.types import (
    CameraError,
    CaptureConfig,
    CaptureStartError,
    ConnectionState,
    Frame,
    FrameStamp,
)

_FPS_SMOOTHING = 0.9
_MAX_CONSECUTIVE_FAILURES = 10

SourceDescriptor = Union[int, str]


def parse_source_descriptor(device_id: str) -> SourceDescriptor:
    """Map an opaque device id to an OpenCV source.

    ``"opencv:N"`` → camera index N. Any other string is treated as a
    video-file path (or URL), which OpenCV natively supports — this is the
    extension point for dataset/file sources without new code paths here.
    """
    prefix = "opencv:"
    if device_id.startswith(prefix):
        try:
            return int(device_id[len(prefix):])
        except ValueError as exc:
            raise ValueError(f"Invalid camera index in id: {device_id!r}") from exc
    if device_id == "":
        raise ValueError("Device id must not be empty.")
    return device_id


class OpenCVCaptureSession(CaptureSession):
    """Frame pump for exactly one source. Not reusable after stop —
    create a fresh session per connection (cheap by design)."""

    def __init__(
        self,
        source: SourceDescriptor,
        config: Optional[CaptureConfig] = None,
    ) -> None:
        self._source = source
        self._config = config or CaptureConfig()

        self._lock = threading.RLock()
        self._state = ConnectionState.DISCONNECTED
        self._last_error: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nominal_interval_s = 0.0

        self._state_listeners: ListenerSet[StateListener] = ListenerSet()
        self._frame_listeners: ListenerSet[FrameListener] = ListenerSet()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                raise CameraError("Capture session is already running.")
            self._last_error = None
            self._stop_event.clear()
            self._set_state(ConnectionState.CONNECTING)
            thread = threading.Thread(
                target=self._run,
                name="perceptlab-camera-capture",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
        # Preserve ERROR for diagnostics; otherwise reflect the idle link.
        if self.get_state() != ConnectionState.ERROR:
            self._set_state(ConnectionState.DISCONNECTED)

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_state(self) -> ConnectionState:
        with self._lock:
            return self._state

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def on_state_change(self, listener: StateListener) -> Unsubscribe:
        return self._state_listeners.add(listener)

    def on_frame(self, listener: FrameListener) -> Unsubscribe:
        return self._frame_listeners.add(listener)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_state(self, state: ConnectionState) -> None:
        with self._lock:
            if self._state == state:
                return
            self._state = state
            listeners = self._state_listeners.snapshot()
        notify_listeners(listeners, state)

    def _fail(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        self._set_state(ConnectionState.ERROR)
        self._stop_event.set()

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            cap.release()
            raise CaptureStartError(
                f"Could not open capture source: {self._source!r}"
            )
        if self._config.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        if self._config.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self._config.fps))
        nominal_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._nominal_interval_s = (
            1.0 / nominal_fps if nominal_fps and nominal_fps > 0 else 0.0
        )
        return cap

    def _run(self) -> None:
        sequence = 0
        failures = 0
        last_monotonic: Optional[int] = None
        smoothed_fps = 0.0
        delivered_at = time.perf_counter()

        try:
            cap = self._open_capture()
        except Exception as exc:  # open failure → ERROR state
            self._fail(str(exc))
            return

        try:
            while not self._stop_event.is_set():
                ok, image = cap.read()
                now_mono = time.perf_counter_ns()
                now_wall = time.time_ns()

                if not ok or image is None:
                    failures += 1
                    if failures >= _MAX_CONSECUTIVE_FAILURES:
                        self._fail(
                            "Capture stream ended or frames became unavailable."
                        )
                        return
                    if self._stop_event.wait(0.05):
                        return
                    continue

                failures = 0

                if last_monotonic is not None:
                    delta_ns = now_mono - last_monotonic
                    if delta_ns > 0:
                        instant_fps = 1e9 / delta_ns
                        smoothed_fps = (
                            instant_fps
                            if smoothed_fps == 0.0
                            else smoothed_fps * _FPS_SMOOTHING
                            + instant_fps * (1.0 - _FPS_SMOOTHING)
                        )
                last_monotonic = now_mono

                sequence += 1
                stamp = FrameStamp(
                    sequence=sequence,
                    monotonic_ns=now_mono,
                    wallclock_ns=now_wall,
                    fps=round(smoothed_fps, 1),
                )
                height, width = image.shape[:2]
                frame = Frame(stamp=stamp, width=width, height=height, image=image)

                self._set_state(ConnectionState.CONNECTED)

                notify_listeners(self._frame_listeners.snapshot(), frame)

                # Pace file-style sources to their nominal rate so the loop
                # neither spins hot nor races ahead of real time.
                if self._nominal_interval_s > 0.0:
                    elapsed = time.perf_counter() - delivered_at
                    remaining = self._nominal_interval_s - elapsed
                    if remaining > 0 and self._stop_event.wait(remaining):
                        return
                    delivered_at = time.perf_counter()
        finally:
            cap.release()
