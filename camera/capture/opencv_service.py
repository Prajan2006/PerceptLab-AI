"""Application-facing camera service facade (OpenCV-backed).

Implements the ``CameraService`` contract one-to-one as the frontend's
``CameraService`` TypeScript interface, so a future HTTP/WebSocket bridge
can translate between them without reshaping either side.

The facade owns lifecycle state only; frame pumping lives in
``OpenCVCaptureSession`` and device enumeration behind ``DeviceDiscovery``.
Both collaborators are injectable for testing and future backends.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from ..interfaces.contracts import (
    CameraService,
    CaptureSession,
    DeviceDiscovery,
    FrameListener,
    ListenerSet,
    StateListener,
    Unsubscribe,
    notify_listeners,
)
from ..interfaces.types import (
    CameraError,
    CaptureConfig,
    CaptureDeviceInfo,
    ConnectionState,
    DeviceNotFoundError,
)
from .opencv_capture import (
    OpenCVCaptureSession,
    SourceDescriptor,
    parse_source_descriptor,
)

SessionFactory = Callable[[SourceDescriptor], CaptureSession]


class OpenCVCameraService(CameraService):
    """Single-active-connection camera service.

    ``connect()`` resolves the id through the injected resolver (default:
    ``parse_source_descriptor``), builds a session via the factory, and
    starts it. Stream-open failures surface asynchronously as an ERROR
    state (the reader thread performs the actual open); callers observe
    them via ``on_state_change`` / ``get_state``, mirroring how the
    frontend consumes state transitions.
    """

    def __init__(
        self,
        discovery: Optional[DeviceDiscovery] = None,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        if discovery is None:
            # Lazy import keeps capture decoupled from concrete devices.
            from ..devices.opencv_discovery import OpenCVDeviceDiscovery

            discovery = OpenCVDeviceDiscovery()
        self._discovery = discovery

        self._session_factory: SessionFactory = session_factory or (
            lambda source: OpenCVCaptureSession(source)
        )

        self._lock = threading.RLock()
        self._session: Optional[CaptureSession] = None
        self._active_device_id: Optional[str] = None
        self._forwarder_unsubscribers: List[Unsubscribe] = []

        self._state_observers = ListenerSet[StateListener]()
        self._frame_observers = ListenerSet[FrameListener]()

    # ------------------------------------------------------------------
    # CameraService contract
    # ------------------------------------------------------------------

    def list_devices(self) -> List[CaptureDeviceInfo]:
        return self._discovery.list_devices()

    def connect(self, device_id: Optional[str] = None) -> None:
        resolved_id = self._resolve_device_id(device_id)

        with self._lock:
            current = self._session
            if current is not None and current.is_running():
                raise CameraError("A camera is already connected.")
            # Stale finished session (e.g., after stream error): discard.
            if current is not None:
                self._teardown_session_locked()

            session = self._session_factory(parse_source_descriptor(resolved_id))
            self._active_device_id = resolved_id

            self._forwarder_unsubscribers.append(
                session.on_state_change(self._on_session_state)
            )
            self._forwarder_unsubscribers.append(
                session.on_frame(self._on_session_frame)
            )
            self._session = session

            try:
                session.start()  # spawns reader thread; errors → ERROR state
            except Exception:
                self._teardown_session_locked()
                raise

    def disconnect(self) -> None:
        with self._lock:
            had_link = self._session is not None
            self._teardown_session_locked()
        if had_link:
            # Always end the public state timeline at DISCONNECTED, even if
            # the session ended in ERROR.
            self._publish_state(ConnectionState.DISCONNECTED)

    def get_state(self) -> ConnectionState:
        with self._lock:
            session = self._session
        return session.get_state() if session is not None else ConnectionState.DISCONNECTED

    def get_active_device_id(self) -> Optional[str]:
        with self._lock:
            return self._active_device_id

    def on_state_change(self, listener: StateListener) -> Unsubscribe:
        return self._state_observers.add(listener)

    def on_frame(self, listener: FrameListener) -> Unsubscribe:
        return self._frame_observers.add(listener)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device_id(self, device_id: Optional[str]) -> str:
        if device_id is not None:
            if not isinstance(device_id, str) or device_id == "":
                raise DeviceNotFoundError(f"Invalid device id: {device_id!r}")
            return device_id
        devices = self.list_devices()
        if not devices:
            raise DeviceNotFoundError(
                "No capture devices found and no device id provided."
            )
        return devices[0].id

    def _teardown_session_locked(self) -> None:
        session = self._session
        for unsubscribe in self._forwarder_unsubscribers:
            unsubscribe()
        self._forwarder_unsubscribers.clear()
        if session is not None:
            session.stop()
        self._session = None
        self._active_device_id = None

    def _on_session_state(self, state: ConnectionState) -> None:
        self._publish_state(state)

    def _on_session_frame(self, frame) -> None:
        notify_listeners(self._frame_observers.snapshot(), frame)

    def _publish_state(self, state: ConnectionState) -> None:
        notify_listeners(self._state_observers.snapshot(), state)

    # Context-manager convenience for scripts/tests.
    def __enter__(self) -> "OpenCVCameraService":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.disconnect()
        return False
