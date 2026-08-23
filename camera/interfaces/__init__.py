"""Abstract contracts and shared types for PerceptLab AI camera foundations."""

from .contracts import (
    CameraService,
    CaptureSession,
    DeviceDiscovery,
    FrameListener,
    ListenerSet,
    StateListener,
    Unsubscribe,
    notify_listeners,
)
from .types import (
    CameraError,
    CaptureConfig,
    CaptureDeviceInfo,
    CaptureStartError,
    ConnectionState,
    DeviceNotFoundError,
    Frame,
    FrameStamp,
    Resolution,
    SourceKind,
)

__all__ = [
    "CameraError",
    "CameraService",
    "CaptureConfig",
    "CaptureDeviceInfo",
    "CaptureSession",
    "CaptureStartError",
    "ConnectionState",
    "DeviceDiscovery",
    "DeviceNotFoundError",
    "Frame",
    "FrameListener",
    "FrameStamp",
    "ListenerSet",
    "Resolution",
    "SourceKind",
    "StateListener",
    "Unsubscribe",
    "notify_listeners",
]
