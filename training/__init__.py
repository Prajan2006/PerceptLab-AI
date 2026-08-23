"""Training layer: minimal baseline trainer for the ResNet-50 gaze model."""

from .baseline import (
    BASELINE_CONFIG_PATH,
    BaselineConfigError,
    EngineeringBaseline,
    load_engineering_baseline,
)
from .trainer import BaselineTrainer, TrainingConfig, apply_seed

__all__ = [
    "BASELINE_CONFIG_PATH",
    "BaselineConfigError",
    "BaselineTrainer",
    "EngineeringBaseline",
    "TrainingConfig",
    "apply_seed",
    "load_engineering_baseline",
]
