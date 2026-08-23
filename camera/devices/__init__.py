"""Device discovery implementations.

Discovery is intentionally separated from capture: enumerating sources
must never require opening them for streaming, and future backends
(vendor SDKs, network scanners) slot in behind ``DeviceDiscovery``.
"""

from .opencv_discovery import OpenCVDeviceDiscovery

__all__ = ["OpenCVDeviceDiscovery"]
