"""Preprocessing for the RAW MPIIFaceGaze layout (annotation → model input).

Per-sample, stateless pipeline built on the existing GazeHub conventions:

1. load the annotated RGB frame (``RawAnnotation.image_path``)
2. derive the gaze direction exactly as documented by the dataset:
       gaze_direction = gaze_target - face_center
   then L2-normalise to unit length (project-wide convention shared with
   ``data.gaze3d`` and the angular-error evaluator)
3. crop the face from the six 2-D landmarks: bounding box of the landmark
   extent, square-ified and expanded by ``expand_ratio``, clamped to the
   frame, resized to ``face_size`` (224×224 for ResNet-50)
4. optionally crop eye patches around the four eye-corner landmarks
   (dataset order 0-3) resized to the dataset-native 36×60
5. ImageNet mean/std normalisation (CHW float32), matching the .mat path

Identity — subject_id, session_id, sample_id, filename, source line — is
preserved so LOPO evaluation stays exact. The pipeline is stateless per
sample: no statistics are fitted across samples or subjects, therefore no
information from a held-out test subject can leak into preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..gaze3d import normalize_vector
from ..mpiifacegaze.raw_adapter import RawAnnotation
from .gazehub import (
    PreprocessConfig,
    PreprocessedSample,
    _apply_imagenet,
    _resize_nearest,
    _square_expand_bbox,
    _to_rgb_chw,
)

# MPIIFaceGaze six-landmark order (dataset documentation): indices 0-3 are
# the four eye corners — right outer, right inner, left inner, left outer.
RIGHT_EYE_CORNERS = (0, 1)
LEFT_EYE_CORNERS = (2, 3)

BBox = tuple[int, int, int, int]  # x, y, w, h


class RawPreprocessError(ValueError):
    """A sample could not be preprocessed; reported, never silently skipped."""


def load_image_rgb(path) -> np.ndarray:
    """Read an image file as RGB uint8 (H, W, 3)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RawPreprocessError(f"unreadable or missing image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def derive_gaze_direction(
    face_center: tuple[float, float, float],
    gaze_target: tuple[float, float, float],
) -> np.ndarray:
    """Official relation: direction = gt − fc, then unit-length normalised."""
    vector = np.asarray(gaze_target, dtype=np.float64) - np.asarray(face_center, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise RawPreprocessError("gaze direction contains non-finite values")
    try:
        return normalize_vector(vector)
    except ValueError as exc:
        raise RawPreprocessError(
            "degenerate gaze geometry: face_center equals gaze_target"
        ) from exc


def face_bbox_from_landmarks(
    landmarks,
    frame_shape: tuple[int, ...],
    ratio: float,
) -> BBox:
    """Square-expanded bbox around the six landmarks, clamped to the frame."""
    points = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return _square_expand_bbox(
        (int(round(x_min)), int(round(y_min)), int(round(x_max - x_min)), int(round(y_max - y_min))),
        frame_shape,
        ratio,
    )


def eye_box_from_corners(
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    frame_shape: tuple[int, ...],
    ratio: float,
    min_side: int = 24,
) -> BBox:
    """Square-expanded bbox spanning two eye-corner landmarks."""
    ax, ay = corner_a
    bx, by = corner_b
    width = abs(bx - ax)
    height = abs(by - ay)
    side = max(width, height, float(min_side))
    center_x, center_y = (ax + bx) / 2.0, (ay + by) / 2.0
    half = side * ratio / 2.0
    return _square_expand_bbox(
        (
            int(round(center_x - half)),
            int(round(center_y - half)),
            int(round(2 * half)),
            int(round(2 * half)),
        ),
        frame_shape,
        1.0,
    )


def _crop_resize_normalize(
    image_rgb: np.ndarray,
    bbox: BBox,
    size: tuple[int, int],
    imagenet: bool,
) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise RawPreprocessError(f"degenerate crop box {bbox}")
    crop = image_rgb[y : y + h, x : x + w]
    chw = _to_rgb_chw(crop)
    return _apply_imagenet(_resize_nearest(chw, size[0], size[1]), imagenet)


@dataclass(frozen=True)
class RawPreprocessor:
    """Stateless per-sample processor for raw-layout annotations."""

    config: PreprocessConfig = PreprocessConfig()

    def process_raw(self, annotation: RawAnnotation, image_rgb: np.ndarray | None = None) -> PreprocessedSample:
        cfg = self.config
        if image_rgb is None:
            image_rgb = load_image_rgb(annotation.image_path)
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise RawPreprocessError(f"expected RGB HxWx3 frame, got {image_rgb.shape}")

        gaze_direction = derive_gaze_direction(annotation.face_center, annotation.gaze_target)

        face_box = face_bbox_from_landmarks(annotation.landmarks, image_rgb.shape, cfg.expand_ratio)
        face = _crop_resize_normalize(image_rgb, face_box, (cfg.face_size, cfg.face_size), cfg.imagenet_normalization)

        left_eye = right_eye = None
        boxes: dict[str, BBox | None] = {"face": face_box, "left_eye": None, "right_eye": None}
        if cfg.include_eyes:
            left_box = eye_box_from_corners(
                annotation.landmarks[LEFT_EYE_CORNERS[0]],
                annotation.landmarks[LEFT_EYE_CORNERS[1]],
                image_rgb.shape,
                cfg.eye_expand_ratio,
            )
            right_box = eye_box_from_corners(
                annotation.landmarks[RIGHT_EYE_CORNERS[0]],
                annotation.landmarks[RIGHT_EYE_CORNERS[1]],
                image_rgb.shape,
                cfg.eye_expand_ratio,
            )
            left_eye = _crop_resize_normalize(image_rgb, left_box, cfg.eye_size, cfg.imagenet_normalization)
            right_eye = _crop_resize_normalize(image_rgb, right_box, cfg.eye_size, cfg.imagenet_normalization)
            boxes["left_eye"] = left_box
            boxes["right_eye"] = right_box

        return PreprocessedSample(
            sample_id=f"{annotation.subject_id}:{annotation.source_line}",
            subject_id=annotation.subject_id,
            face=face,
            left_eye=left_eye,
            right_eye=right_eye,
            gaze=gaze_direction,
            head_pose=np.asarray(annotation.head_rotation, dtype=np.float64),
            meta={
                "filename": annotation.relative_path,
                "session_id": annotation.session_id,
                "source_line": annotation.source_line,
                "eye_side": annotation.eye_side,
                "boxes": boxes,
                "image_shape": list(image_rgb.shape[:2]),
            },
        )

    # PreprocessingProtocol compatibility: accept GazeSample-shaped inputs
    # that carry a full_image (e.g., produced by annotation_to_sample).
    def process(self, sample) -> PreprocessedSample:  # type: ignore[override]
        raise TypeError(
            "RawPreprocessor processes RawAnnotation via process_raw(); "
            "use GazeHubPreprocessor for GazeSample inputs."
        )


def annotation_to_sample(annotation: RawAnnotation, image_rgb: np.ndarray):
    """Bridge to the .mat-era GazeSample for consumers expecting it."""
    from ..mpiifacegaze.adapter import GazeSample

    return GazeSample(
        sample_id=f"{annotation.subject_id}:{annotation.source_line}",
        subject_id=annotation.subject_id,
        session_id=annotation.session_id,
        filename=annotation.relative_path,
        gaze=derive_gaze_direction(annotation.face_center, annotation.gaze_target),
        head_pose=np.asarray(annotation.head_rotation, dtype=np.float64),
        face_bbox=None,
        left_eye=None,
        right_eye=None,
        full_image=image_rgb,
    )
