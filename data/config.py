"""Configuration-driven dataset path resolution.

Dataset locations are never hard-coded inside adapters: they resolve from
(1) an environment override, (2) ``config/datasets.json``, then (3) the
declared default. The full dataset is never downloaded automatically — a
missing root raises an actionable error instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REGISTRY_PATH = PROJECT_ROOT / "config" / "datasets.json"


@dataclass(frozen=True)
class DatasetLocation:
    name: str
    root: Path
    source: str  # 'env' | 'config' | 'default'


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a dataset root is not configured or does not exist."""


def load_dataset_registry(path: Path = DATASET_REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_dataset_root(
    dataset_name: str,
    registry_path: Path = DATASET_REGISTRY_PATH,
    must_exist: bool = True,
) -> DatasetLocation:
    registry = load_dataset_registry(registry_path)
    entries = registry.get("datasets", {})
    if dataset_name not in entries:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Registered: {sorted(entries)}"
        )
    entry = entries[dataset_name]

    env_var = entry.get("root_env")
    if env_var and os.environ.get(env_var):
        return DatasetLocation(dataset_name, Path(os.environ[env_var]).resolve(), "env")

    if entry.get("default_root"):
        candidate = (PROJECT_ROOT / entry["default_root"]).resolve()
        if candidate.exists() or not must_exist:
            return DatasetLocation(dataset_name, candidate, "default")

    raise DatasetNotAvailableError(
        f"Dataset {dataset_name!r} is not available. Set the {env_var!s} "
        "environment variable to its root directory. The platform never "
        "downloads datasets automatically."
    )


def resolve_experiment_output_dir(relative_or_absolute: str | os.PathLike[str]) -> Path:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
