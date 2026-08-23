"""OpenCV-backed device discovery (Windows: Media Foundation / DirectShow).

Probes camera indices through OpenCV's native backends and reports the
negotiated default resolution and nominal FPS. Vendor-neutral by design:
any UVC laptop/USB camera that Windows exposes is discoverable.
"""

from __future__ import annotations

import cv2

from ..interfaces.contracts import DeviceDiscovery
from ..interfaces.types import CaptureDeviceInfo, Resolution, SourceKind

_BACKEND_LABELS = {
    cv2.CAP_MSMF: "Media Foundation",
    cv2.CAP_DSHOW: "DirectShow",
}


class OpenCVDeviceDiscovery(DeviceDiscovery):
    """Probe-based discovery. ``max_probes`` bounds the index scan;
    ``backends`` orders preference (first backend that opens wins)."""

    def __init__(
        self,
        max_probes: int = 5,
        backends: tuple = (cv2.CAP_MSMF, cv2.CAP_DSHOW),
    ) -> None:
        self._max_probes = max(1, int(max_probes))
        self._backends = tuple(backends)

    def list_devices(self) -> list[CaptureDeviceInfo]:
        devices: list[CaptureDeviceInfo] = []
        for index in range(self._max_probes):
            info = self._probe_index(index)
            if info is not None:
                devices.append(info)
        return devices

    def _probe_index(self, index: int) -> CaptureDeviceInfo | None:
        for backend in self._backends:
            cap = cv2.VideoCapture(index, backend)
            try:
                if not cap.isOpened():
                    continue
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                label = _BACKEND_LABELS.get(backend, f"backend-{backend}")
                return CaptureDeviceInfo(
                    id=f"opencv:{index}",
                    label=f"Camera {index} ({label})",
                    kind=SourceKind.CAMERA,
                    resolution=(
                        Resolution(width, height)
                        if width > 0 and height > 0
                        else None
                    ),
                    max_fps=fps if fps and fps > 0 else None,
                    backend=label,
                )
            finally:
                cap.release()  # discovery must never leave devices held open
        return None
