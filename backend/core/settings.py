"""Transport-layer configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    jpeg_quality: int = 80
    discovery_max_probes: int = 5
    fps_cap: float | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except ValueError:
                return default

        def _float_opt(name: str) -> float | None:
            raw = os.environ.get(name)
            if not raw:
                return None
            try:
                value = float(raw)
                return value if value > 0 else None
            except ValueError:
                return None

        return cls(
            host=os.environ.get("PL_HOST", "127.0.0.1"),
            port=_int("PL_PORT", 8000),
            jpeg_quality=max(1, min(100, _int("PL_JPEG_QUALITY", 80))),
            discovery_max_probes=max(0, _int("PL_DISCOVERY_MAX_PROBES", 5)),
            fps_cap=_float_opt("PL_FPS_CAP"),
        )
