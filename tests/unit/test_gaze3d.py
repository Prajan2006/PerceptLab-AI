"""Unit tests for 3D gaze-vector representation and geometry helpers."""

import numpy as np
import pytest

from data.gaze3d import (
    angular_error_deg,
    gaze_to_yaw_pitch_deg,
    normalize_vector,
    yaw_pitch_to_gaze_deg,
)


class TestNormalize:
    def test_scales_to_unit_length(self):
        unit = normalize_vector([3.0, 0.0, 0.0])
        assert np.allclose(unit, [1.0, 0.0, 0.0])
        assert np.isclose(np.linalg.norm(unit), 1.0)

    def test_arbitrary_vector(self):
        unit = normalize_vector([1.0, 1.0, 1.0])
        assert np.isclose(np.linalg.norm(unit), 1.0)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError):
            normalize_vector([0.0, 0.0, 0.0])


class TestAngularError:
    def test_identical_vectors_zero(self):
        assert angular_error_deg([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_ninety(self):
        assert angular_error_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0, abs=1e-9)

    def test_opposite_vectors_180(self):
        assert angular_error_deg([0, 0, 1], [0, 0, -1]) == pytest.approx(180.0, abs=1e-9)

    def test_known_angle(self):
        # 45 degrees between z-axis and (1, 0, 1).
        assert angular_error_deg([1, 0, 1], [0, 0, 1]) == pytest.approx(45.0, abs=1e-9)

    def test_unnormalized_inputs_handled(self):
        a = angular_error_deg([10, 0, 0], [0, 5, 0])
        assert a == pytest.approx(90.0, abs=1e-9)

    def test_clamping_avoids_nan(self):
        value = angular_error_deg([1, 1e-12, 0], [-1, -1e-12, 0])
        assert np.isfinite(value)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError):
            angular_error_deg([0, 0, 0], [0, 0, 1])


class TestYawPitchRoundTrip:
    @pytest.mark.parametrize(
        "yaw,pitch",
        [(0, 0), (15, -10), (-25, 8), (45, -30), (90, 0), (0, 40)],
    )
    def test_roundtrip(self, yaw, pitch):
        gaze = yaw_pitch_to_gaze_deg(yaw, pitch)
        recovered_yaw, recovered_pitch = gaze_to_yaw_pitch_deg(gaze)
        assert recovered_yaw == pytest.approx(yaw, abs=1e-9)
        assert recovered_pitch == pytest.approx(pitch, abs=1e-9)
