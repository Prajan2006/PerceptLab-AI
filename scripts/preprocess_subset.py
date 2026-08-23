"""Preprocess a small real MPIIFaceGaze subset and report statistics.

Usage::

    python scripts/preprocess_subset.py --subject p00 --limit 40

Read-only over the raw dataset; prints a JSON report with processed/rejected
counts, tensor shapes, label representation, coordinate convention, and the
normalization formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.config import resolve_dataset_root  # noqa: E402
from data.mpiifacegaze.raw_adapter import read_subject_annotations  # noqa: E402
from data.preprocessing.gazehub import PreprocessConfig  # noqa: E402
from data.preprocessing.raw_pipeline import RawPreprocessError, RawPreprocessor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mpii_facegaze")
    parser.add_argument("--subject", default="p00")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--sessions", default="", help="comma-separated dayXX filter (empty = all)")
    args = parser.parse_args()

    location = resolve_dataset_root(args.dataset)
    subject_dir = location.root / args.subject
    annotations, malformed = read_subject_annotations(subject_dir)

    wanted_sessions = {s.strip() for s in args.sessions.split(",") if s.strip()}
    if wanted_sessions:
        annotations = [a for a in annotations if a.session_id in wanted_sessions]
    annotations = annotations[: max(0, args.limit)]

    preprocessor = RawPreprocessor(PreprocessConfig())
    processed = 0
    rejected: list[dict] = []
    session_counts: Counter[str] = Counter()
    first_sample: dict | None = None

    for annotation in annotations:
        try:
            sample = preprocessor.process_raw(annotation)
        except RawPreprocessError as error:
            rejected.append(
                {
                    "sample_id": f"{annotation.subject_id}:{annotation.source_line}",
                    "reason": str(error),
                }
            )
            continue

        processed += 1
        session_counts[annotation.session_id] += 1
        if first_sample is None:
            first_sample = {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "session_id": sample.meta["session_id"],
                "filename": sample.meta["filename"],
                "face_shape": list(sample.face.shape),
                "left_eye_shape": None if sample.left_eye is None else list(sample.left_eye.shape),
                "right_eye_shape": (
                    None if sample.right_eye is None else list(sample.right_eye.shape)
                ),
                "gaze_label": [round(float(v), 6) for v in sample.gaze],
                "gaze_label_norm": round(float(np.linalg.norm(sample.gaze)), 12),
                "face_dtype": str(sample.face.dtype),
            }

    report = {
        "dataset_root": str(location.root),
        "subject": args.subject,
        "requested": len(annotations),
        "processed": processed,
        "rejected": len(rejected),
        "rejected_detail": rejected,
        "malformed_annotation_rows": len(malformed),
        "sessions_seen": dict(sorted(session_counts.items())),
        "first_sample": first_sample,
        "tensor_shapes": {
            "face": [3, 224, 224],
            "left_eye": [3, 36, 60],
            "right_eye": [3, 36, 60],
        },
        "label_representation": "unit-length 3D gaze direction vector (float64)",
        "coordinate_convention": (
            "camera coordinates exactly as annotated by MPIIFaceGaze "
            "(no axis remapping performed)"
        ),
        "normalization_formula": "gaze_direction = normalize(gaze_target - face_center)  # L2",
        "identity_preserved": ["subject_id", "session_id", "sample_id", "filename", "source_line"],
        "leakage_note": (
            "stateless per-sample transform; no cross-sample or cross-subject "
            "statistics are fitted"
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if not rejected and not malformed else 1


if __name__ == "__main__":
    raise SystemExit(main())
