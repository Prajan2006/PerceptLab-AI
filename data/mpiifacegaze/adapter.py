"""MPIIFaceGaze dataset adapter.

Official layout: one MATLAB file per subject (``p00.mat`` … ``p14.mat``).
Each file contains:

- ``filenames`` — cell array of image paths such as ``"p00/day01/0000.jpg"``
- ``data``      — struct array with fields:
    * ``left``  36×60 uint8 left-eye patch
    * ``right`` 36×60 uint8 right-eye patch
    * ``rect``  face bounding box ``[x, y, w, h]``
    * ``gaze``  unit-length 3D gaze vector
    * ``head``  head-pose vector

The adapter preserves subject identity and the day/session component of
each filename so LOPO grouping is always exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SUBJECT_FILE_PATTERN = re.compile(r"^(p\d{2})\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class GazeSample:
    sample_id: str
    subject_id: str
    session_id: str | None
    filename: str
    gaze: np.ndarray          # (3,) float64 unit vector
    head_pose: np.ndarray | None
    face_bbox: tuple[int, int, int, int] | None
    left_eye: np.ndarray | None   # HxW uint8 (RGB channel order preserved as stored)
    right_eye: np.ndarray | None
    # Optional richer imagery for future dataset variants / re-cropping:
    # MPIIFaceGaze .mat files embed only the eye patches.
    face_image: np.ndarray | None = None   # embedded face crop if a source provides one
    full_image: np.ndarray | None = None   # original frame, enables bbox-based face crops


def list_subject_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MPIIFaceGaze root does not exist: {root}")
    files = [p for p in sorted(root.glob("*.mat")) if _SUBJECT_FILE_PATTERN.match(p.name)]
    return files


def discover_subjects(root: Path) -> list[str]:
    """Sorted subject ids (p00…pNN) discovered from ``<root>/pYY.mat``."""
    return [
        _SUBJECT_FILE_PATTERN.match(path.name).group(1).lower()
        for path in list_subject_files(root)
    ]


def parse_session(filename: str) -> str | None:
    parts = Path(filename.replace("\\", "/")).parts
    return parts[1] if len(parts) >= 3 else None


class MPIIFaceGazeSubjectReader:
    """Loads a single subject's samples from its ``pYY.mat``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def load(self, subject_id: str):
        from scipy.io import loadmat  # local import keeps heavy dep optional at import time

        mat_path = self.root / f"{subject_id.lower()}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Subject file not found: {mat_path}")

        metadata = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        records = np.atleast_1d(metadata["data"])
        filenames = np.atleast_1d(metadata.get("filenames", np.empty(0, dtype=object)))

        samples: list[GazeSample] = []
        for index in range(records.shape[0]):
            record = records[index]
            filename = (
                str(filenames[index])
                if index < filenames.shape[0]
                else f"{subject_id}/unknown/{index:04d}.jpg"
            )
            samples.append(
                GazeSample(
                    sample_id=f"{subject_id}:{index}",
                    subject_id=subject_id.lower(),
                    session_id=parse_session(filename),
                    filename=filename,
                    gaze=np.asarray(record.gaze, dtype=np.float64).reshape(3),
                    head_pose=(
                        np.asarray(record.head, dtype=np.float64).reshape(3)
                        if getattr(record, "head", None) is not None
                        else None
                    ),
                    face_bbox=tuple(int(v) for v in np.asarray(record.rect).reshape(-1)[:4])
                    if getattr(record, "rect", None) is not None
                    else None,
                    left_eye=_as_uint8_image(getattr(record, "left", None)),
                    right_eye=_as_uint8_image(getattr(record, "right", None)),
                )
            )
        return samples


def _as_uint8_image(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    return array.astype(np.uint8)


class MPIIFaceGazeDataset:
    """Facade over all subjects with lazy per-subject loading."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._reader = MPIIFaceGazeSubjectReader(self.root)
        self._subjects = discover_subjects(self.root)

    @property
    def subjects(self) -> list[str]:
        return list(self._subjects)

    def load_subject(self, subject_id: str) -> list[GazeSample]:
        if subject_id not in self._subjects:
            raise KeyError(
                f"Unknown subject {subject_id!r}. Available: {self._subjects}"
            )
        return self._reader.load(subject_id)

    def __iter__(self):
        for subject in self._subjects:
            yield from self.load_subject(subject)
