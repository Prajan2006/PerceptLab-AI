"""Abstract contracts for device discovery and capture.

Concrete implementations live in ``camera.devices`` and
``camera.capture``; application code depends only on these abstractions,
so capture backends can be swapped without touching higher layers.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, TypeVar

from .types import CaptureDeviceInfo, ConnectionState, Frame

StateListener = Callable[[ConnectionState], None]
FrameListener = Callable[[Frame], None]
Unsubscribe = Callable[[], None]

T = TypeVar("T")


class ListenerSet(Generic[T]):
    """Thread-safe listener registry returning unsubscribe callables.

    Notification order is registration order. Exceptions raised by a
    listener never propagate to the notifier's control flow — each
    listener is isolated.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: List[T] = []

    def add(self, listener: T) -> Unsubscribe:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass  # already removed — unsubscribe is idempotent

        return unsubscribe

    def snapshot(self) -> tuple:
        with self._lock:
            return tuple(self._listeners)


def notify_listeners(listeners: tuple, event) -> None:
    for listener in listeners:
        try:
            listener(event)
        except Exception:  # noqa: BLE001 — one bad listener must not kill capture
            pass


class DeviceDiscovery(ABC):
    """Enumerates available capture sources."""

    @abstractmethod
    def list_devices(self) -> List[CaptureDeviceInfo]:
        """Return currently available devices (may be empty)."""


class FrameSource(ABC):
    """One underlying media stream: webcam index, video file, RTSP URL, ...

    Blocking reads by design; threading/lifecycle belongs to the session.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the stream. Raises ``CaptureStartError`` on failure."""

    @abstractmethod
    def read(self):
        """Return ``(ok, image)`` mirroring cv2 semantics."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource. Must be idempotent."""


class CaptureSession(ABC):
    """Frame pump bound to exactly one source."""

    @abstractmethod
    def start(self) -> None:
        """Begin acquisition. Idempotency: raises if already running."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition and release resources. Safe to call twice."""

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def get_state(self) -> ConnectionState: ...

    @abstractmethod
    def get_last_error(self) -> Optional[str]: ...

    @abstractmethod
    def on_state_change(self, listener: StateListener) -> Unsubscribe: ...

    @abstractmethod
    def on_frame(self, listener: FrameListener) -> Unsubscribe: ...

    def __enter__(self) -> "CaptureSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False


class CameraService(ABC):
    """Application-facing facade — mirrors
    ``frontend/services/interfaces/CameraService.ts`` one-to-one:

    listDevices / connect / disconnect / getState / getActiveDeviceId /
    onStateChange / onFrame.
    """

    @abstractmethod
    def list_devices(self) -> List[CaptureDeviceInfo]: ...

    @abstractmethod
    def connect(self, device_id: Optional[str] = None) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_state(self) -> ConnectionState: ...

    @abstractmethod
    def get_active_device_id(self) -> Optional[str]: ...

    @abstractmethod
    def on_state_change(self, listener: StateListener) -> Unsubscribe: ...

    @abstractmethod
    def on_frame(self, listener: FrameListener) -> Unsubscribe: ...
