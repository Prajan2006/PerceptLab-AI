"""Concrete capture implementations (OpenCV-backed)."""

from .opencv_capture import OpenCVCaptureSession, parse_source_descriptor
from .opencv_service import OpenCVCameraService

__all__ = ["OpenCVCameraService", "OpenCVCaptureSession", "parse_source_descriptor"]
