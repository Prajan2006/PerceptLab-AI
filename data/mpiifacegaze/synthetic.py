"""Deterministic synthetic MPIIFaceGaze subject files for testing.

Produces ``pYY.mat`` files that mirror the official structure (``data``
struct array with left/right/rect/gaze/head + ``filenames``) so the real
adapter, index builder, and preprocessing pipeline are all exercised
without the 45k-sample dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

EYE_HEIGHT, EYE_WIDTH = 36, 60
SESSIONS = ("day01", "day02", "day03", "day04")


def _gaze_for(index: int) -> np.ndarray:
    yaw = -15.0 + (index % 7) * 5.0
    pitch = -10.0 + (index % 5) * 5.0
    pitch_rad, yaw_rad = np.deg2rad(pitch), np.deg2rad(yaw)
    vector = np.array(
        [
            np.cos(pitch_rad) * np.sin(yaw_rad),
            -np.sin(pitch_rad),
            np.cos(pitch_rad) * np.cos(yaw_rad),
        ],
        dtype=np.float64,
    )
    return vector / np.linalg.norm(vector)


def build_synthetic_subject_file(
    root: Path,
    subject_id: str,
    num_samples: int = 24,
    seed: int = 0,
) -> Path:
    """Write ``<root>/<subject_id>.mat`` with deterministic content."""
    rng = np.random.default_rng(seed)
    subject_id = subject_id.lower()

    records = []
    filenames = []
    for index in range(num_samples):
        session = SESSIONS[index % len(SESSIONS)]
        filenames.append(f"{subject_id}/{session}/{index:04d}.jpg")
        records.append(
            (
                rng.integers(0, 256, size=(EYE_HEIGHT, EYE_WIDTH), dtype=np.uint8),   # left
                rng.integers(0, 256, size=(EYE_HEIGHT, EYE_WIDTH), dtype=np.uint8),   # right
                np.array([120, 80, 400, 400], dtype=np.float64),                      # rect x,y,w,h
                _gaze_for(index).astype(np.float64),                                  # gaze
                np.array([0.1 * ((index % 9) - 4), 0.2, 1.0], dtype=np.float64),      # head
            )
        )

    data = np.empty((1, num_samples), dtype=object)
    for i, record in enumerate(records):
        entry = np.empty((1, 1), dtype=[("left", "O"), ("right", "O"), ("rect", "O"), ("gaze", "O"), ("head", "O")])
        entry[0, 0] = record
        data[0, i] = entry[0, 0]

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{subject_id}.mat"
    savemat(
        str(target),
        {
            "filenames": np.array([filenames], dtype=object),
            "data": data,
        },
    )
    return target


def build_synthetic_dataset(root: Path, subject_ids=("p00", "p01", "p02")) -> Path:
    """Create a tiny multi-subject dataset mirroring the official layout."""
    for position, subject in enumerate(subject_ids):
        build_synthetic_subject_file(root, subject, num_samples=18 + position * 3, seed=position)
    return Path(root)

