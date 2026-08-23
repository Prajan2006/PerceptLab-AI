"""Parser + adapter for the RAW MPIIFaceGaze layout.

Verified dataset root contains one directory per subject::

    <root>/pYY/                     (p00 … p14)
        Calibration/                (Camera.mat, monitorPose.mat, screenSize.mat)
        day01/ … dayNN/             original JPEG frames
        pYY.txt                     one annotation row per frame

Annotation rows are space-separated (28 fields) per the OFFICIAL
MPIIFaceGaze annotation specification::

    field  1      relative image path              e.g. "day01/0005.jpg"
    fields 2-3    gaze location on the screen      (pixel coordinates)
    fields 4-15   six facial landmarks             (x, y) × 6, dataset order
    fields 16-21  estimated 3D head pose           rotation (rvec) + translation (tvec)
    fields 22-24  face centre (fc)                 3D camera coordinates
    fields 25-27  gaze target (gt)                 3D camera coordinates
    field  28     annotated eye side               'left' | 'right'

The official documentation defines::

    gaze direction = gt - fc

Deriving that direction (and any normalisation) belongs to the
preprocessing layer — this parser preserves every raw value verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_SUBJECT_DIR_PATTERN = re.compile(r"^p(\d{2})$", re.IGNORECASE)

EXPECTED_TOKENS = 28
LANDMARK_COUNT = 6

IGNORED_FILE_NAMES = {".DS_Store"}
IGNORED_DIR_NAMES = {"__MACOSX"}

VALID_EYE_SIDES = {"left", "right"}


class AnnotationParseError(ValueError):
    """A single annotation row failed strict parsing."""


@dataclass(frozen=True)
class RawAnnotation:
    subject_id: str
    session_id: str                                   # e.g. "day01"
    relative_path: str                                # e.g. "day01/0005.jpg"
    image_path: Path                                  # resolved against the subject dir
    gaze_screen_location: tuple[float, float]         # fields 2-3 (pixels)
    landmarks: tuple[tuple[float, float], ...]        # fields 4-15, six (x, y), raw order
    head_rotation: tuple[float, float, float]         # fields 16-18 (rvec, raw)
    head_translation: tuple[float, float, float]      # fields 19-21 (tvec, raw)
    face_center: tuple[float, float, float]           # fields 22-24 (fc, camera coords)
    gaze_target: tuple[float, float, float]           # fields 25-27 (gt, camera coords)
    eye_side: str                                     # field 28 ('left' | 'right')
    source_line: int                                  # 1-based line number in pYY.txt

    @property
    def session_key(self) -> str:
        return f"{self.subject_id}/{self.session_id}"


@dataclass(frozen=True)
class MalformedAnnotation:
    subject_id: str
    source_line: int
    reason: str


def parse_annotation_line(
    line: str,
    subject_dir: Path,
    source_line: int,
) -> RawAnnotation:
    """Strict positional parser per the official 28-field specification."""
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name.lower()
    tokens = line.split()

    if len(tokens) != EXPECTED_TOKENS:
        raise AnnotationParseError(
            f"expected {EXPECTED_TOKENS} whitespace-separated fields, got {len(tokens)}"
        )

    relative_path = tokens[0]
    path_parts = relative_path.split("/")
    if len(path_parts) != 2 or "\\" in relative_path:
        raise AnnotationParseError(
            f"path must be '<session>/<file>' with '/' separators, got {relative_path!r}"
        )

    def number(index: int) -> float:
        try:
            return float(tokens[index])
        except ValueError as exc:
            raise AnnotationParseError(
                f"non-numeric field #{index + 1}: {tokens[index]!r}"
            ) from exc

    try:
        eye_side = tokens[27]
        if eye_side not in VALID_EYE_SIDES:
            raise AnnotationParseError(f"invalid eye side {eye_side!r}")
    except IndexError as exc:  # pragma: no cover — guarded by token count
        raise AnnotationParseError("missing eye-side field") from exc

    return RawAnnotation(
        subject_id=subject_id,
        session_id=path_parts[0],
        relative_path=relative_path,
        image_path=(subject_dir / relative_path).resolve(),
        # fields 2-3: gaze location on screen (pixels)
        gaze_screen_location=(number(1), number(2)),
        # fields 4-15: six facial landmarks
        landmarks=tuple((number(3 + i * 2), number(4 + i * 2)) for i in range(LANDMARK_COUNT)),
        # fields 16-21: head pose = rotation + translation
        head_rotation=(number(15), number(16), number(17)),
        head_translation=(number(18), number(19), number(20)),
        # fields 22-24: face centre (fc)
        face_center=(number(21), number(22), number(23)),
        # fields 25-27: gaze target (gt)
        gaze_target=(number(24), number(25), number(26)),
        # field 28
        eye_side=eye_side,
        source_line=source_line,
    )


def read_subject_annotations(subject_dir: Path) -> tuple[list[RawAnnotation], list[MalformedAnnotation]]:
    """Parse ``<subject>/pYY.txt``. Corrupt rows are reported, never skipped silently."""
    subject_dir = Path(subject_dir)
    annotation_file = subject_dir / f"{subject_dir.name}.txt"
    annotations: list[RawAnnotation] = []
    malformed: list[MalformedAnnotation] = []

    if not annotation_file.exists():
        malformed.append(
            MalformedAnnotation(
                subject_dir.name.lower(), 0, f"missing annotation file {annotation_file.name}"
            )
        )
        return annotations, malformed

    with annotation_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                annotations.append(parse_annotation_line(line, subject_dir, line_number))
            except AnnotationParseError as error:
                malformed.append(
                    MalformedAnnotation(subject_dir.name.lower(), line_number, str(error))
                )
    return annotations, malformed


def discover_raw_subjects(root: Path) -> list[str]:
    """Sorted ``pYY`` directories, ignoring archive artifacts."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return [
        entry.name.lower()
        for entry in sorted(root.iterdir())
        if entry.is_dir()
        and entry.name not in IGNORED_DIR_NAMES
        and _SUBJECT_DIR_PATTERN.match(entry.name)
    ]


@dataclass
class DatasetValidationReport:
    dataset_root: Path
    subjects: list[str] = field(default_factory=list)
    sessions: set[str] = field(default_factory=set)
    annotation_records: int = 0
    images_on_disk: int = 0
    valid_matches: int = 0
    missing_images_total: int = 0
    missing_images_sample: list[str] = field(default_factory=list)
    malformed_annotations: list[str] = field(default_factory=list)
    ignored_artifacts: int = 0

    def summary(self) -> dict:
        return {
            "dataset_root": str(self.dataset_root),
            "num_subjects": len(self.subjects),
            "subjects": self.subjects,
            "num_sessions": len(self.sessions),
            "annotation_records": self.annotation_records,
            "images_on_disk": self.images_on_disk,
            "valid_matches": self.valid_matches,
            "missing_images": self.missing_images_total,
            "malformed_annotations": len(self.malformed_annotations),
            "ignored_artifacts": self.ignored_artifacts,
        }


def count_ignored_artifacts(root: Path) -> int:
    """Count __MACOSX entries and .DS_Store files anywhere under root."""
    ignored = 0
    for entry in Path(root).rglob("*"):
        if any(part in IGNORED_DIR_NAMES for part in entry.relative_to(root).parts):
            ignored += 1
            continue
        if entry.name in IGNORED_FILE_NAMES:
            ignored += 1
    return ignored


def validate_raw_dataset(root: Path, max_missing_sample: int = 20) -> DatasetValidationReport:
    """Walk the dataset and verify every annotation ↔ image pairing."""
    root = Path(root)
    report = DatasetValidationReport(dataset_root=root)

    report.subjects = discover_raw_subjects(root)
    report.ignored_artifacts = count_ignored_artifacts(root)

    for subject in report.subjects:
        subject_dir = root / subject
        annotations, malformed = read_subject_annotations(subject_dir)

        report.malformed_annotations.extend(
            f"{subject}:{problem.source_line}: {problem.reason}" for problem in malformed
        )

        for annotation in annotations:
            report.annotation_records += 1
            report.sessions.add(annotation.session_key)
            if annotation.image_path.exists():
                report.images_on_disk += 1
                report.valid_matches += 1
            else:
                report.missing_images_total += 1
                if len(report.missing_images_sample) < max_missing_sample:
                    report.missing_images_sample.append(str(annotation.image_path))

    return report
