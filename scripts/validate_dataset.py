"""Validate the configured MPIIFaceGaze dataset (real data).

Usage::

    python scripts/validate_dataset.py [--dataset mpii_facegaze]

Reads the root from config/datasets.json / PERCEPTLAB_MPIIFACE_GAZE_ROOT
(never hard-coded), verifies every annotation ↔ image pairing, checks the
15-fold LOPO generation, and prints a JSON report. Read-only: the raw
dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.config import resolve_dataset_root  # noqa: E402
from data.mpiifacegaze.raw_adapter import validate_raw_dataset  # noqa: E402
from data.splits import lopo_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mpii_facegaze")
    args = parser.parse_args()

    location = resolve_dataset_root(args.dataset)
    print(f"dataset root: {location.root} (resolved via {location.source})", file=sys.stderr)

    report = validate_raw_dataset(location.root)
    folds = lopo_folds(report.subjects)

    payload = report.summary()
    payload["lopo_fold_count"] = len(folds)
    payload["lopo_deterministic_order"] = [fold.test_subjects[0] for fold in folds]
    if report.missing_images_sample:
        payload["missing_images_sample"] = report.missing_images_sample
    if report.malformed_annotations:
        payload["malformed_detail"] = report.malformed_annotations[:10]

    print(json.dumps(payload, indent=2))
    return 0 if report.missing_images_total == 0 and not report.malformed_annotations else 1


if __name__ == "__main__":
    raise SystemExit(main())
