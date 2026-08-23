"""Deterministic synthetic fixtures for the RAW MPIIFaceGaze layout.

Builds a miniature tree that mirrors the verified real structure::

    <root>/pYY/Calibration/          (empty placeholder)
    <root>/pYY/dayXX/NNNN.jpg        (tiny valid JPEGs)
    <root>/pYY/pYY.txt               (28-field annotation rows)
    <root>/__MACOSX/, .DS_Store      (archive artifacts to be ignored)

Used by unit tests so parser, matching, artifact filtering, and LOPO
generation are exercised without the 45k-sample dataset.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

IGNORED_DIR = "__MACOSX"


def format_annotation_row(session: str, filename: str, index: int = 0, eye_side: str | None = None) -> str:
    """Emit one 28-token annotation row per the official field layout."""
    screen_location = (500 + index, 300 + index % 7)
    landmarks = [(100 + i * 10, 200 + i * 5) for i in range(6)]
    head_rotation = (-0.232179 + index * 0.001, 0.055685, 0.018205)
    head_translation = (28.351504, 1.174807, 529.783734)
    face_center = (27.792112, 23.422692, 524.537075)
    gaze_target = (11.040978, 166.869249, -27.728178)
    side = eye_side or ("right" if index % 2 == 0 else "left")

    tokens = [f"{session}/{filename}"]
    tokens.extend(str(value) for value in screen_location)
    for pair in landmarks:
        tokens.extend(str(value) for value in pair)
    tokens.extend(f"{value:.6f}" for value in head_rotation)
    tokens.extend(f"{value:.6f}" for value in head_translation)
    tokens.extend(f"{value:.6f}" for value in face_center)
    tokens.extend(f"{value:.6f}" for value in gaze_target)
    tokens.append(side)
    return " ".join(tokens)


def _write_tiny_jpeg(path: Path, level: int) -> None:
    image = np.full((48, 64, 3), max(0, min(level, 255)), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("fixture JPEG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(path)


def build_synthetic_raw_subject(
    root: Path,
    subject_id: str = "p00",
    sessions: tuple[str, ...] = ("day01", "day02"),
    frames_per_session: int = 3,
    include_missing_image: bool = False,
    include_malformed_row: bool = False,
    include_artifacts: bool = True,
) -> Path:
    """Create one subject directory with images + annotation file."""
    root = Path(root)
    subject_dir = root / subject_id.lower()
    calibration_dir = subject_dir / "Calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    sequence = 0
    for session_index, session in enumerate(sessions):
        for frame_index in range(frames_per_session):
            filename = f"{frame_index:04d}.jpg"
            image_path = subject_dir / session / filename
            missing = include_missing_image and session_index == len(sessions) - 1 and frame_index == 0
            if not missing:
                _write_tiny_jpeg(image_path, session_index * 40 + frame_index * 10)
            rows.append(format_annotation_row(session, filename, sequence))
            sequence += 1

    if include_malformed_row:
        rows.append("day01/broken.jpg 1 2 3")  # wrong token count on purpose

    (subject_dir / f"{subject_id.lower()}.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    if include_artifacts:
        artifact_dir = root / IGNORED_DIR / subject_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"._{subject_id}.jpg").write_bytes(b"\x00")
        (root / ".DS_Store").write_bytes(b"\x00\x00")

    return subject_dir


def build_synthetic_raw_dataset(
    root: Path,
    subjects: tuple[str, ...] = ("p00", "p01"),
    sessions: tuple[str, ...] = ("day01", "day02"),
    frames_per_session: int = 3,
) -> Path:
    for position, subject in enumerate(subjects):
        build_synthetic_raw_subject(
            root,
            subject,
            sessions=sessions[: 1 + position % len(sessions)],
            frames_per_session=frames_per_session,
            include_missing_image=(position == 0),
            include_malformed_row=(position == len(subjects) - 1),
        )
    return Path(root)
