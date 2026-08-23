"""Deterministic tests for raw-layout MPIIFaceGaze preprocessing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from data.mpiifacegaze.raw_adapter import RawAnnotation
from data.preprocessing.gazehub import PreprocessConfig
from data.preprocessing.raw_pipeline import (
    RawPreprocessError,
    RawPreprocessor,
    derive_gaze_direction,
    eye_box_from_corners,
    face_bbox_from_landmarks,
    load_image_rgb,
)

FC = (27.792112, 23.422692, 524.537075)
GT = (11.040978, 166.869249, -27.728178)


def make_annotation(tmp_path: Path, line: int = 1, subject: str = "p00") -> RawAnnotation:
    return RawAnnotation(
        subject_id=subject,
        session_id="day01",
        relative_path="day01/0005.jpg",
        image_path=tmp_path / subject / "day01" / "0005.jpg",
        gaze_screen_location=(476.0, 758.0),
        landmarks=((100.0, 200.0), (110.0, 205.0), (120.0, 210.0), (130.0, 215.0), (140.0, 220.0), (150.0, 225.0)),
        head_rotation=(-0.232179, 0.055685, 0.018205),
        head_translation=(28.351504, 1.174807, 529.783734),
        face_center=FC,
        gaze_target=GT,
        eye_side="right",
        source_line=line,
    )


def gradient_image(height: int = 480, width: int = 640) -> np.ndarray:
    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    cols = np.linspace(255, 0, width, dtype=np.uint8)[None, :]
    base = (rows + cols) // 2
    return np.stack([base, base // 2, 255 - base], axis=-1)


class TestGazeDirection:
    def test_exact_formula_gt_minus_fc(self):
        direction = derive_gaze_direction((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        assert np.allclose(direction, [0.6, 0.8, 0.0], atol=1e-12)

    def test_translation_invariant(self):
        a = derive_gaze_direction((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        b = derive_gaze_direction((10.0, 20.0, 30.0), (13.0, 24.0, 30.0))
        assert np.allclose(a, b, atol=1e-12)

    def test_real_annotation_values_unit_length(self):
        direction = derive_gaze_direction(FC, GT)
        assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-12)
        expected = np.asarray(GT) - np.asarray(FC)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(direction, expected, atol=1e-12)

    def test_degenerate_geometry_rejected(self):
        with pytest.raises(RawPreprocessError, match="degenerate"):
            derive_gaze_direction(FC, FC)

    def test_non_finite_rejected(self):
        with pytest.raises(RawPreprocessError):
            derive_gaze_direction((np.nan, 0, 0), (1, 0, 0))


class TestCropGeometry:
    LANDMARKS = ((100, 200), (110, 205), (120, 210), (130, 215), (140, 220), (150, 225))

    def test_face_bbox_square_and_centered(self):
        box = face_bbox_from_landmarks(self.LANDMARKS, (480, 640, 3), ratio=2.0)
        x, y, w, h = box
        assert w == h == 100
        assert (x, y) == (75, 162)  # centre (125, 212.5), exact square

    def test_face_bbox_clamped_to_frame(self):
        near_corner = tuple((x + 10, y + 10) for x, y in self.LANDMARKS[:3]) + self.LANDMARKS[3:]
        shifted = tuple((px - 90, py - 190) for px, py in self.LANDMARKS)
        box = face_bbox_from_landmarks(shifted, (480, 640, 3), ratio=2.0)
        x, y, w, h = box
        assert x >= 0 and y >= 0 and x + w <= 640 and y + h <= 480
        _ = near_corner

    def test_eye_box_spans_corners_with_ratio(self):
        box = eye_box_from_corners((100, 200), (140, 200), (480, 640, 3), ratio=3.0)
        x, y, w, h = box
        assert w == h == 120
        assert (x, y) == (60, 140)


class TestRawPreprocessor:
    @pytest.fixture()
    def annotation(self, tmp_path):
        path = tmp_path / "p00" / "day01" / "0005.jpg"
        path.parent.mkdir(parents=True)
        ok, buffer = cv2.imencode(".jpg", gradient_image())
        assert ok
        buffer.tofile(path)
        return make_annotation(tmp_path)

    def test_output_shapes_dtypes_and_identity(self, annotation):
        sample = RawPreprocessor().process_raw(annotation)

        assert sample.sample_id == "p00:1"
        assert sample.subject_id == "p00"
        assert sample.meta["session_id"] == "day01"
        assert sample.meta["filename"] == "day01/0005.jpg"
        assert sample.meta["source_line"] == 1
        assert sample.meta["eye_side"] == "right"

        assert sample.face.shape == (3, 224, 224)
        assert sample.face.dtype == np.float32
        assert np.isfinite(sample.face).all()

        assert sample.left_eye is not None and sample.left_eye.shape == (3, 36, 60)
        assert sample.right_eye is not None and sample.right_eye.shape == (3, 36, 60)

        assert sample.gaze.dtype == np.float64
        assert np.isclose(np.linalg.norm(sample.gaze), 1.0, atol=1e-12)
        assert np.allclose(sample.gaze, derive_gaze_direction(FC, GT), atol=1e-12)

        boxes = sample.meta["boxes"]
        for name in ("face", "left_eye", "right_eye"):
            bx, by, bw, bh = boxes[name]
            assert 0 <= bx and 0 <= by and bx + bw <= 640 and by + bh <= 480

    def test_label_is_pure_gt_minus_fc(self, annotation):
        sample = RawPreprocessor().process_raw(annotation)
        manual = np.asarray(GT, dtype=np.float64) - np.asarray(FC, dtype=np.float64)
        manual /= np.linalg.norm(manual)
        assert np.array_equal(sample.gaze, manual)

    def test_head_pose_carries_raw_rvec(self, annotation):
        sample = RawPreprocessor().process_raw(annotation)
        assert np.allclose(sample.head_pose, (-0.232179, 0.055685, 0.018205))

    def test_deterministic(self, annotation):
        first = RawPreprocessor().process_raw(annotation)
        second = RawPreprocessor().process_raw(annotation)
        assert np.array_equal(first.face, second.face)
        assert np.array_equal(first.left_eye, second.left_eye)
        assert np.array_equal(first.right_eye, second.right_eye)
        assert np.array_equal(first.gaze, second.gaze)

    def test_stateless_order_independence_no_leakage(self, tmp_path):
        """Per-sample independence: processing order cannot change outputs."""
        image = gradient_image()
        first_annotation = make_annotation(tmp_path, line=1, subject="p00")
        second_annotation = make_annotation(tmp_path, line=2, subject="p13")

        forward_a = RawPreprocessor().process_raw(first_annotation, image_rgb=image.copy())
        forward_b = RawPreprocessor().process_raw(second_annotation, image_rgb=image.copy())
        reverse_b = RawPreprocessor().process_raw(second_annotation, image_rgb=image.copy())
        reverse_a = RawPreprocessor().process_raw(first_annotation, image_rgb=image.copy())

        assert np.array_equal(forward_a.face, reverse_a.face)
        assert np.array_equal(forward_a.gaze, reverse_a.gaze)
        assert np.array_equal(forward_b.face, reverse_b.face)
        # Identity travels with each sample regardless of order.
        assert forward_a.sample_id == reverse_a.sample_id
        assert forward_b.sample_id == reverse_b.sample_id
        assert forward_a.subject_id == "p00" and forward_b.subject_id == "p13"

    def test_eyes_disabled(self, annotation):
        config = PreprocessConfig(include_eyes=False)
        sample = RawPreprocessor(config).process_raw(annotation)
        assert sample.face is not None
        assert sample.left_eye is None and sample.right_eye is None

    def test_normalization_disabled_keeps_unit_range(self, annotation):
        config = PreprocessConfig(imagenet_normalization=False)
        sample = RawPreprocessor(config).process_raw(annotation)
        assert float(sample.face.min()) >= 0.0
        assert float(sample.face.max()) <= 1.0

    def test_unreadable_image_rejected(self, tmp_path):
        annotation = make_annotation(tmp_path)
        with pytest.raises(RawPreprocessError, match="unreadable"):
            RawPreprocessor().process_raw(annotation)

    def test_bad_image_shape_rejected(self, annotation):
        with pytest.raises(RawPreprocessError, match="RGB HxWx3"):
            RawPreprocessor().process_raw(annotation, image_rgb=np.zeros((10, 10), dtype=np.uint8))


class TestImageLoading:
    def test_loads_rgb(self, tmp_path):
        path = tmp_path / "frame.jpg"
        ok, buffer = cv2.imencode(".jpg", gradient_image(32, 48))
        assert ok
        buffer.tofile(path)

        image = load_image_rgb(path)
        assert image.shape == (32, 48, 3)
        # OpenCV loads BGR; RGB conversion must swap channels.
        bgr = cv2.imread(str(path))
        assert np.array_equal(image[:, :, 0], bgr[:, :, 2])
