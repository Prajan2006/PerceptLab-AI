"""ResNet-50 gaze-estimation architecture (face branch only).

Implements the registered ``resnet50`` model spec: the appearance-based
MPIIFaceGaze baseline that consumes exactly the preprocessed face tensor
``(B, 3, 224, 224)`` (ImageNet-normalized CHW, as produced by
``RawPreprocessor`` / ``collate_gaze_samples``) and emits an unnormalized
3-D gaze-direction prediction ``(B, 3)``. The locked angular-error evaluator
normalizes predictions internally, so no normalization layer lives here.

Design notes:
- The backbone is torchvision's canonical ResNet-50; its final classifier is
  replaced by a single linear head mapping the 2048-D pooled feature to the
  3-D gaze vector.
- ``pretrained_backbone=True`` loads ImageNet weights at construction time;
  failures surface as :class:`~models.registry.ModelBuildError` with an
  actionable message. Tests run with random init (no network access).
- Head pose, eye patches, and the gaze label are NOT inputs: they stay in
  the batch as metadata/labels per the registry input contract.

No training, loss, optimizer, or checkpointing logic belongs here.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import ModelBuildError

EXPECTED_FACE_SHAPE = (3, 224, 224)


class ResNet50Gaze(nn.Module):
    """Face-only ResNet-50 → 3-D gaze direction."""

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:  # download/cache/network problems
            raise ModelBuildError(
                "could not construct ResNet-50 backbone "
                f"(pretrained_backbone={pretrained_backbone!r}): {exc}"
            ) from exc
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 3)
        self.backbone = backbone

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        _validate_face(face)
        return self.backbone(face)


def _validate_face(face: torch.Tensor) -> None:
    if not isinstance(face, torch.Tensor):
        raise ValueError(f"face must be a torch.Tensor, got {type(face)!r}")
    if face.ndim != 4 or tuple(face.shape[1:]) != EXPECTED_FACE_SHAPE:
        raise ValueError(
            f"face must have shape (B, {', '.join(map(str, EXPECTED_FACE_SHAPE))}); "
            f"got {tuple(face.shape)}"
        )
