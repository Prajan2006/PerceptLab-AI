"""Model layer: registry of architectures and their implementations."""

from .registry import (
    MODEL_REGISTRY,
    ModelBuildError,
    ModelSpec,
    build_model,
    get_model_spec,
    list_models,
)
from .resnet50 import ResNet50Gaze
from .resnet50_face_eyes import ResNet50FaceEyes

__all__ = [
    "MODEL_REGISTRY",
    "ModelBuildError",
    "ModelSpec",
    "ResNet50Gaze",
    "ResNet50FaceEyes",
    "build_model",
    "get_model_spec",
    "list_models",
]
