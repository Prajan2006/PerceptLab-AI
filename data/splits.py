"""Deterministic leave-one-person-out (LOPO) split generation.

Subject identity — never random sampling of frames — drives the protocol,
matching the standardized MPIIFaceGaze evaluation: 15 subjects, each one
held out exactly once. Ordering is fully deterministic (sorted subject ids)
so any run reproduces the same folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPLIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LOPOFold:
    fold_index: int
    test_subjects: tuple[str, ...]
    train_subjects: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "test_subjects": list(self.test_subjects),
            "train_subjects": list(self.train_subjects),
        }


def lopo_folds(subjects: list[str]) -> list[LOPOFold]:
    """One fold per subject; deterministic ordering; strict disjointness."""
    ordered = sorted(set(subjects))
    if len(ordered) != len(subjects):
        raise ValueError("Duplicate subject ids are not allowed.")
    if not ordered:
        raise ValueError("At least one subject is required for LOPO.")
    return [
        LOPOFold(
            fold_index=index,
            test_subjects=(subject,),
            train_subjects=tuple(other for other in ordered if other != subject),
        )
        for index, subject in enumerate(ordered)
    ]


def save_lopo_folds(folds: list[LOPOFold], path: Path) -> Path:
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "protocol": "leave-one-person-out",
        "num_subjects": len({subject for fold in folds for subject in fold.test_subjects}),
        "folds": [fold.to_dict() for fold in folds],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_lopo_folds(path: Path) -> list[LOPOFold]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "leave-one-person-out":
        raise ValueError(f"Unexpected split protocol: {payload.get('protocol')!r}")
    return [
        LOPOFold(
            fold_index=int(entry["fold_index"]),
            test_subjects=tuple(entry["test_subjects"]),
            train_subjects=tuple(entry["train_subjects"]),
        )
        for entry in payload["folds"]
    ]
