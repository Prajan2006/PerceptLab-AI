"""Dataset metadata / index generation.

The index is a lightweight JSON manifest (subject counts, session list,
schema version) used by the UI and by experiment tooling without touching
the heavy MATLAB files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

INDEX_SCHEMA_VERSION = 1


def build_index(dataset_root: Path, output_path: Path | None = None) -> dict:
    from .adapter import discover_subjects, parse_session
    from .adapter import MPIIFaceGazeSubjectReader

    dataset_root = Path(dataset_root)
    subjects = discover_subjects(dataset_root)
    reader = MPIIFaceGazeSubjectReader(dataset_root)

    subject_entries: dict[str, dict] = {}
    total = 0
    for subject in subjects:
        samples = reader.load(subject)
        sessions = sorted({sample.session_id for sample in samples if sample.session_id})
        subject_entries[subject] = {
            "num_samples": len(samples),
            "sessions": sessions,
            "first_filename": samples[0].filename,
            "last_filename": samples[-1].filename,
        }
        total += len(samples)

    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset": "MPIIFaceGaze",
        "root": str(dataset_root),
        "num_subjects": len(subjects),
        "subjects": subjects,
        "total_samples": total,
        "per_subject": subject_entries,
        "built_with_env": {
            key: value
            for key, value in (("PERCEPTLAB_MPIIFACE_GAZE_ROOT", os.environ.get("PERCEPTLAB_MPIIFACE_GAZE_ROOT")),)
            if value
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def load_index(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset") != "MPIIFaceGaze":
        raise ValueError(f"Not an MPIIFaceGaze index: {path}")
    return payload
