"""Unit tests for the GazeHub-compatible preprocessing pipeline."""

import numpy as np
import pytest

from data.mpiifacegaze.adapter import GazeSample
from data.preprocessing.gazehub import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    GazeHubPreprocessor,
    PreprocessConfig,
)


def make_sample(with_full_image: bool = False, with_embedded_face: bool = False) -> GazeSample:
    rng = np.random.default_rng(7)
    full_image = (
        rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) if with_full_image else None
    )
    embedded = (
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) if with_embedded_face else None
    )
    return GazeSample(
        sample_id="p00:0",
        subject_id="p00",
        session_id="day01",
        filename="p00/day01/0000.jpg",
        gaze=np.array([2.0, 0.0, 2.0]),  # deliberately not unit length
        head_pose=np.array([0.1, -0.2, 1.0]),
        face_bbox=(200, 100, 120, 140),
        left_eye=rng.integers(0, 256, size=(36, 60), dtype=np.uint8),
        right_eye=rng.integers(0, 256, size=(36, 60), dtype=np.uint8),
        face_image=embedded,
        full_image=full_image,
    )


class TestGazeHubPreprocessor:
    def test_eye_patches_shape_and_normalization(self):
        output = GazeHubPreprocessor().process(make_sample())
        assert output.left_eye.shape == (3, 36, 60)
        assert output.right_eye.shape == (3, 36, 60)
        assert np.isfinite(output.left_eye).all()
        # Normalized values must leave the [0,1] range (mean-shifted).
        assert output.left_eye.min() < 0.0 or output.left_eye.max() > 1.0

    def test_gaze_renormalized_to_unit(self):
        output = GazeHubPreprocessor().process(make_sample())
        assert np.allclose(output.gaze, [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)], atol=1e-9)

    def test_face_from_full_image_when_available(self):
        output = GazeHubPreprocessor().process(make_sample(with_full_image=True))
        assert output.face is not None and output.face.shape == (3, 224, 224)
        assert output.meta["face_source"] == "full_image"
        assert len(output.meta["crop"]) == 4

    def test_embedded_face_used_as_fallback(self):
        output = GazeHubPreprocessor().process(make_sample(with_embedded_face=True))
        assert output.face is not None and output.face.shape == (3, 224, 224)
        assert output.meta["face_source"] == "embedded"

    def test_no_face_source_yields_none(self):
        output = GazeHubPreprocessor().process(make_sample())
        assert output.face is None
        assert output.meta["face_source"] == "unavailable"

    def test_normalization_disabled_keeps_unit_range(self):
        config = PreprocessConfig(imagenet_normalization=False)
        output = GazeHubPreprocessor(config).process(make_sample(with_embedded_face=True))
        assert 0.0 <= float(output.face.min()) and float(output.face.max()) <= 1.0

    def test_imagenet_constants_match_reference(self):
        assert np.allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406])
        assert np.allclose(IMAGENET_STD, [0.229, 0.224, 0.225])

    def test_subject_and_session_metadata_preserved(self):
        output = GazeHubPreprocessor().process(make_sample())
        assert output.sample_id == "p00:0"
        assert output.subject_id == "p00"
        assert output.meta["session_id"] == "day01"

    def test_deterministic(self):
        first = GazeHubPreprocessor().process(make_sample(with_embedded_face=True))
        second = GazeHubPreprocessor().process(make_sample(with_embedded_face=True))
        assert np.array_equal(first.left_eye, second.left_eye)
        assert np.array_equal(first.face, second.face)
