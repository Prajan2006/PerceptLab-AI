"""Composition root for the transport layer.

Holds the single application-scoped ``CameraService`` (the existing
camera-module facade). The transport never re-implements camera logic;
it only orchestrates access to this instance.
"""

from __future__ import annotations

import threading

from camera.capture.opencv_service import OpenCVCameraService
from camera.devices.opencv_discovery import OpenCVDeviceDiscovery
from camera.interfaces.contracts import CameraService

from .settings import Settings


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._camera_service: CameraService | None = None

    @property
    def camera_service(self) -> CameraService:
        with self._lock:
            if self._camera_service is None:
                self._camera_service = self.build_camera_service()
            return self._camera_service

    def build_camera_service(self) -> CameraService:
        return OpenCVCameraService(
            discovery=OpenCVDeviceDiscovery(max_probes=self.settings.discovery_max_probes),
        )

    def override_camera_service(self, service: CameraService) -> None:
        """Test seam: inject a deterministic service implementation."""
        with self._lock:
            self._camera_service = service
