"""GazeHub-compatible preprocessing for appearance-based gaze estimation.

Implements the standardized MPIIFaceGaze recipe used by the GazeHub
protocol collection:

- face patch: expand the annotated bounding box to a square by
  ``expand_ratio``, crop, resize to ``face_size`` (224×224 for ResNet)
- eye patches: kept at the dataset-native 36×60 resolution
- gaze vector: re-normalized to unit length
- images: RGB float32, ImageNet mean/std normalized (CHW layout)

The interface is framework-free (NumPy in/out) so training frameworks can
be introduced later without touching dataset or evaluation layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..gaze3d import normalize_vector
from ..mpiifacegaze.adapter import GazeSample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PreprocessConfig:
    face_size: int = 224
    expand_ratio: float = 2.0
    imagenet_normalization: bool = True
    # Eye-patch options (raw-layout pipeline; the .mat adapter always emits eyes).
    include_eyes: bool = True
    eye_size: tuple[int, int] = (36, 60)      # (height, width), dataset-native shape
    eye_expand_ratio: float = 3.0


@dataclass(frozen=True)
class PreprocessedSample:
    sample_id: str
    subject_id: str
    face: np.ndarray | None      # (3,H,W) float32 or None if source lacks full image
    left_eye: np.ndarray | None  # (3,H,W) float32 or None when unavailable/disabled
    right_eye: np.ndarray | None # (3,H,W) float32 or None when unavailable/disabled
    gaze: np.ndarray             # (3,) float64 unit vector
    head_pose: np.ndarray | None
    meta: dict = field(default_factory=dict)


class PreprocessingProtocol(ABC):
    """Contract between dataset adapters and model code."""

    @abstractmethod
    def process(self, sample: GazeSample) -> PreprocessedSample: ...


def _to_rgb_chw(image: np.ndarray) -> np.ndarray:
    """HxW(xC) uint8 → 3xHxW float32 in [0,1], replicated to 3 channels if gray."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array.astype(np.float32).transpose(2, 0, 1) / 255.0


def _apply_imagenet(chw: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return chw.astype(np.float32)
    return ((chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(np.float32)


def _resize_nearest(chw: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize on CHW float arrays."""
    channels, src_h, src_w = chw.shape
    row_idx = np.clip((np.arange(height) * src_h / max(height, 1)).astype(int), 0, src_h - 1)
    col_idx = np.clip((np.arange(width) * src_w / max(width, 1)).astype(int), 0, src_w - 1)
    return chw[:, row_idx][:, :, col_idx]


def _square_expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Expand ``bbox`` to an exactly-square box by ``ratio``, clamped in-frame."""
    x, y, w, h = bbox
    center_x, center_y = x + w / 2.0, y + h / 2.0
    side = max(1, int(round(max(w, h) * ratio)))
    height, width = frame_shape[:2]
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(left, 0), max(0, width - side))
    top = min(max(top, 0), max(0, height - side))
    return left, top, side, side


class GazeHubPreprocessor(PreprocessingProtocol):
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def process(self, sample: GazeSample) -> PreprocessedSample:
        cfg = self.config

        face_chw: np.ndarray | None = None
        crop_info: dict = {"face_source": "unavailable"}
        if sample.face_bbox is not None and sample.full_image is not None:
            full = np.asarray(sample.full_image)
            x, y, w, h = _square_expand_bbox(sample.face_bbox, full.shape, cfg.expand_ratio)
            crop = full[y : y + h, x : x + w]
            face_chw = _resize_nearest(_to_rgb_chw(crop), cfg.face_size, cfg.face_size)
            crop_info = {"face_source": "full_image", "crop": [x, y, w, h]}
        elif sample.face_image is not None:
            face_chw = _resize_nearest(
                _to_rgb_chw(np.asarray(sample.face_image)), cfg.face_size, cfg.face_size
            )
            crop_info = {"face_source": "embedded"}

        left_chw = _resize_nearest(
            _to_rgb_chw(np.asarray(sample.left_eye)), sample.left_eye.shape[0], sample.left_eye.shape[1]
        )
        right_chw = _resize_nearest(
            _to_rgb_chw(np.asarray(sample.right_eye)), sample.right_eye.shape[0], sample.right_eye.shape[1]
        )

        return PreprocessedSample(
            sample_id=sample.sample_id,
            subject_id=sample.subject_id,
            face=_apply_imagenet(face_chw, cfg.imagenet_normalization) if face_chw is not None else None,
            left_eye=_apply_imagenet(left_chw, cfg.imagenet_normalization),
            right_eye=_apply_imagenet(right_chw, cfg.imagenet_normalization),
            gaze=normalize_vector(sample.gaze),
            head_pose=None if sample.head_pose is None else np.asarray(sample.head_pose, dtype=np.float64),
            meta={"filename": sample.filename, "session_id": sample.session_id, **crop_info},
        )
