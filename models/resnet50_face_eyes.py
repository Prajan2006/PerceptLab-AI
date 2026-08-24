"""Face + localized eye-region gaze model (controlled input-representation arm).

Implements the minimal architecture change required by the research question:

    "Under fixed preprocessing, architecture, training settings, and
     subject-independent LOPO evaluation, does adding localized eye-region
     information to a full-face RGB gaze-estimation baseline improve mean
     angular error on MPIIFaceGaze?"

Design:
- The face pathway is EXACTLY the existing ``resnet50`` backbone (torchvision
  ImageNet initialization when enabled) with its classifier removed; its
  mathematics are unchanged from the frozen baseline model.
- Localized eye regions reuse the already-validated preprocessing crops —
  the ``(3, 36, 60)`` left/right patches present in every ``GazeBatch``. No new
  cropping is introduced.
- A compact shared-weight CNN encodes each eye patch; the two embeddings are
  concatenated with the face feature and mapped to a raw ``(B, 3)`` gaze
  direction by a linear head (same output contract as the baseline; the
  evaluator normalizes).

The single experimental variable versus the reference run is the presence of
eye-region input; loss, optimizer, schedule, batch size, seed, splits, labels,
and evaluation are untouched.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import ModelBuildError

EXPECTED_FACE_SHAPE = (3, 224, 224)
EXPECTED_EYE_SHAPE = (3, 36, 60)
EYE_FEATURES = 32


class EyeEncoder(nn.Module):
    """Compact shared-weight CNN for one 36x60 eye patch."""

    def __init__(self, out_features: int = EYE_FEATURES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.features(eye).flatten(1)


class ResNet50FaceEyes(nn.Module):
    """Full-face ResNet-50 + shared eye-region encoder → 3-D gaze direction."""

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:  # download/cache/network problems
            raise ModelBuildError(
                "could not construct ResNet-50 face backbone "
                f"(pretrained_backbone={pretrained_backbone!r}): {exc}"
            ) from exc
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.face_backbone = backbone
        self.eye_encoder = EyeEncoder()
        self.head = nn.Linear(in_features + 2 * EYE_FEATURES, 3)

    def forward(self, face: torch.Tensor, left_eye: torch.Tensor, right_eye: torch.Tensor) -> torch.Tensor:
        _validate_inputs(face, left_eye, right_eye)
        face_features = self.face_backbone(face)
        eyes = torch.cat(
            [self.eye_encoder(left_eye), self.eye_encoder(right_eye)], dim=1
        )
        return self.head(torch.cat([face_features, eyes], dim=1))


def _validate_inputs(face, left_eye, right_eye) -> None:
    for name, tensor, expected in (
        ("face", face, EXPECTED_FACE_SHAPE),
        ("left_eye", left_eye, EXPECTED_EYE_SHAPE),
        ("right_eye", right_eye, EXPECTED_EYE_SHAPE),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape (B, {', '.join(map(str, expected))}); "
                f"got {tuple(tensor.shape)}"
            )
