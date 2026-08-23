"""3D gaze-vector representation and geometry helpers.

Conventions (matching MPIIFaceGaze / GazeHub):
- A gaze direction is a 3-vector expressed in the camera coordinate system
  and normalized to unit length.
- Head pose is carried as the dataset provides it; conversion helpers live
  here so every layer shares one implementation.
"""

from __future__ import annotations

import math

import numpy as np

Vector3 = np.ndarray  # shape (3,), float64


def normalize_vector(vector: Vector3) -> Vector3:
    """Return ``vector`` scaled to unit length. Raises on zero-length input."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def angular_error_deg(prediction: Vector3, target: Vector3) -> float:
    """Angle in degrees between two (not necessarily unit) 3-vectors."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    prediction_norm = float(np.linalg.norm(prediction))
    target_norm = float(np.linalg.norm(target))
    if prediction_norm == 0.0 or target_norm == 0.0:
        raise ValueError("Angular error is undefined for zero-length vectors.")
    cosine = float(np.dot(prediction, target) / (prediction_norm * target_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def gaze_to_yaw_pitch_deg(gaze: Vector3) -> tuple[float, float]:
    """Human-readable (yaw, pitch) in degrees for logging/plots.

    yaw:   rotation around vertical axis (atan2(x, z))
    pitch: elevation from horizontal plane (asin(-y / |v|))
    """
    gaze = normalize_vector(gaze)
    yaw = math.degrees(math.atan2(float(gaze[0]), float(gaze[2])))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(gaze[1])))))
    return yaw, pitch


def yaw_pitch_to_gaze_deg(yaw_deg: float, pitch_deg: float) -> Vector3:
    """Inverse of :func:`gaze_to_yaw_pitch_deg` — handy for synthetic fixtures."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return normalize_vector(
        np.array(
            [math.cos(pitch) * math.sin(yaw), -math.sin(pitch), math.cos(pitch) * math.cos(yaw)],
            dtype=np.float64,
        )
    )
