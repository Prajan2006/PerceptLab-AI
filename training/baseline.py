"""Frozen Protocol B — Engineering Baseline configuration loader.

Loads ``config/training/engineering_baseline.json`` and enforces that every
training setting carries an explicit provenance label. This makes it
impossible to run the Project Engineering Baseline with undocumented values:
a missing or unrecognized status label, a missing required setting, or a
config claiming to be an official reproduction is rejected at load time.

The loader validates STRUCTURE and PROVENANCE only; pinning exact Protocol B
numbers is done by ``tests/unit/test_engineering_baseline.py``. Nothing here
changes trainer, model, data, or split behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data.config import PROJECT_ROOT
from .trainer import TrainingConfig

BASELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "training" / "engineering_baseline.json"

REQUIRED_SETTING_KEYS = (
    "loss",
    "optimizer",
    "scheduler",
    "epochs",
    "batch_size",
    "seed",
    "pretrained_initialization",
    "checkpoint",
    "augmentation",
    "shuffling",
)


class BaselineConfigError(ValueError):
    """The frozen baseline configuration is missing, malformed, or undocumented."""


@dataclass(frozen=True)
class EngineeringBaseline:
    name: str
    document: dict

    @property
    def training_config(self) -> TrainingConfig:
        optimizer = self.document["settings"]["optimizer"]
        return TrainingConfig(
            epochs=int(self.document["settings"]["epochs"]["value"]),
            learning_rate=float(optimizer["lr"]),
            weight_decay=float(optimizer["weight_decay"]),
        )

    @property
    def batch_size(self) -> int:
        return int(self.document["settings"]["batch_size"]["value"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineConfigError(message)


def load_engineering_baseline(path: str | Path | None = None) -> EngineeringBaseline:
    path = Path(path) if path is not None else BASELINE_CONFIG_PATH
    if not path.exists():
        raise BaselineConfigError(f"baseline configuration missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    identity = payload.get("identity", {})
    _require(
        identity.get("protocol_name") == "Protocol B - Engineering Baseline",
        "configuration must identify itself as 'Protocol B - Engineering Baseline'",
    )
    _require(
        identity.get("is_official_reproduction") is False,
        "the engineering baseline must never claim to be an official reproduction",
    )

    settings = payload.get("settings", {})
    for key in REQUIRED_SETTING_KEYS:
        entry = settings.get(key)
        _require(isinstance(entry, dict), f"missing required training setting: {key}")
        status = entry.get("status")
        _require(
            isinstance(status, str) and len(status) > 0,
            f"setting {key!r} has no provenance status label",
        )
        if status == "SPLIT":
            _require(
                bool(entry.get("verified_part")) and bool(entry.get("project_chosen_part")),
                f"setting {key!r} uses SPLIT status but lacks verified/project-chosen parts",
            )
        else:
            allowed = set(payload.get("allowed_status_labels", []))
            _require(
                status in allowed,
                f"setting {key!r} status {status!r} is not one of the documented labels "
                f"{sorted(allowed)}",
            )
    undocumented = sorted(set(settings) - set(REQUIRED_SETTING_KEYS))
    _require(
        not undocumented,
        f"undocumented training setting(s) {undocumented}; the frozen baseline only "
        f"defines {list(REQUIRED_SETTING_KEYS)}",
    )

    optimizer = settings["optimizer"]
    _require(
        all(k in optimizer for k in ("type", "lr", "weight_decay", "betas", "eps", "amsgrad")),
        "optimizer must enumerate type/lr/weight_decay/betas/eps/amsgrad explicitly "
        "(no hidden library defaults)",
    )
    _require(
        isinstance(settings["scheduler"], dict) and settings["scheduler"].get("type") == "none",
        "frozen baseline requires an explicit scheduler entry of type 'none'",
    )

    return EngineeringBaseline(name=str(payload.get("name", "")), document=payload)
