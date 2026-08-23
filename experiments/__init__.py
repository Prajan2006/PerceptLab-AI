"""Experiment layer: configuration schema for reproducible research runs."""

from .config import (
    ExperimentConfig,
    ExperimentConfigError,
    load_experiment_config,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentConfigError",
    "load_experiment_config",
]
