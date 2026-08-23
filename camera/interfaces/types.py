"""Shared domain types and error hierarchy for the camera module.

These types mirror the application-facing TypeScript contract in
``frontend/services/core/types.ts`` so a transport bridge can translate
one-to-one later. No hardware or framework code belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ConnectionState(str, Enum):
    """Capture-link lifecycle. Values intentionally match the frontend
    contract string-for-string."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class SourceKind(str, Enum):
    """Kind of capture source. Extensible for future sensors without
    breaking consumers (mirrors TS ``CaptureSourceKind``)."""

    CAMERA = "camera"
    VIDEO_FILE = "video_file"


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int


@dataclass(frozen=True)
class CaptureDeviceInfo:
    """Discovery metadata for one capture source. ``id`` is opaque to
    consumers; concrete resolvers map it to a media stream."""

    id: str
    label: str
    kind: SourceKind = SourceKind.CAMERA
    resolution: Optional[Resolution] = None
    max_fps: Optional[float] = None
    backend: str = ""


@dataclass(frozen=True)
class CaptureConfig:
    """Requested capture parameters. ``None`` lets the backend negotiate."""

    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None


@dataclass(frozen=True)
class FrameStamp:
    """Capture timing metadata for one frame.

    - ``monotonic_ns``  — high-resolution monotonic clock taken immediately
      after the frame is grabbed; the basis for inter-frame interval math
      (rPPG time-series). Immune to wall-clock adjustments.
    - ``wallclock_ns``  — epoch nanoseconds at the same instant, for
      correlating frames with logs, experiments, and future sensors.
    - ``sequence``      — per-session counter starting at 1; gaps indicate
      dropped frames.
    - ``fps``           — EMA-smoothed instantaneous frame rate.
    """

    sequence: int
    monotonic_ns: int
    wallclock_ns: int
    fps: float

    @property
    def epoch_ms(self) -> int:
        return self.wallclock_ns // 1_000_000


@dataclass(frozen=True)
class Frame:
    """One delivered frame. ``image`` is BGR uint8 (H, W, 3) shared by
    reference — listeners must treat it as read-only."""

    stamp: FrameStamp
    width: int
    height: int
    image: Optional[np.ndarray]


class CameraError(Exception):
    """Base class for all camera-module errors."""


class DeviceNotFoundError(CameraError):
    """A requested device id could not be resolved."""


class CaptureStartError(CameraError):
    """The underlying media stream failed to open."""
