"""PerceptLab AI camera foundations."""

from .devices.opencv_discovery import OpenCVDeviceDiscovery
from .capture.opencv_capture import OpenCVCaptureSession, parse_source_descriptor
from .capture.opencv_service import OpenCVCameraService
from .interfaces.types import (
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

__version__ = "0.1.0"

__all__ = [
    "CameraError",
    "CaptureConfig",
    "CaptureDeviceInfo",
    "CaptureStartError",
    "ConnectionState",
    "DeviceNotFoundError",
    "Frame",
    "FrameStamp",
    "OpenCVCameraService",
    "OpenCVCaptureSession",
    "OpenCVDeviceDiscovery",
    "Resolution",
    "SourceKind",
    "parse_source_descriptor",
]
