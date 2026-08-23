"""Model registry.

Models are declared with their input contract so experiment configurations
and the UI can reference them today; builders are wired in as each model's
architecture is implemented. This keeps model selection config-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    inputs: dict[str, tuple[int, ...]]          # logical name → CHW shape
    outputs: tuple[str, ...] = ("gaze",)
    params: dict = field(default_factory=dict)
    builder: Callable | None = None             # set once the architecture exists


class ModelBuildError(NotImplementedError):
    """Raised when a registered-but-not-yet-implemented model is built."""


def _build_resnet50(**params):
    from .resnet50 import ResNet50Gaze

    return ResNet50Gaze(**params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "resnet50": ModelSpec(
        name="resnet50",
        description=(
            "ResNet-50 appearance-based gaze estimation baseline "
            "(primary baseline for MPIIFaceGaze LOPO evaluation)."
        ),
        inputs={"face": (3, 224, 224)},
        params={"pretrained_backbone": True},
        builder=_build_resnet50,
    ),
    "gazetr_hybrid": ModelSpec(
        name="gazetr_hybrid",
        description="GazeTR hybrid (CNN eye encoder + transformer) comparison model.",
        inputs={"left_eye": (3, 36, 60), "right_eye": (3, 36, 60), "face": (3, 224, 224)},
        params={"pretrained_backbone": True},
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def list_models() -> list[dict]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "inputs": {key: list(shape) for key, shape in spec.inputs.items()},
            "implemented": spec.builder is not None,
        }
        for spec in sorted(MODEL_REGISTRY.values(), key=lambda s: s.name)
    ]


def build_model(name: str, **params):
    spec = get_model_spec(name)
    if spec.builder is None:
        raise ModelBuildError(
            f"Model {name!r} is registered but its architecture is not implemented yet."
        )
    return spec.builder(**{**spec.params, **params})
