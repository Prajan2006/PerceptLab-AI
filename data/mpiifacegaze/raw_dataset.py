"""PyTorch Dataset/DataLoader layer over the RAW MPIIFaceGaze layout.

Composes the already-validated pieces — ``discover_raw_subjects``,
``read_subject_annotations`` and the stateless ``RawPreprocessor`` — into a
lazy, map-style :class:`torch.utils.data.Dataset`. No preprocessing logic is
duplicated: ``__getitem__`` delegates every pixel to ``RawPreprocessor`` and
returns its :class:`PreprocessedSample` unchanged (NumPy in, NumPy out).
Tensor conversion happens once, deterministically, in
:func:`collate_gaze_samples`.

Identity — sample_id/subject_id/session_id/source_line/filename/eye_side —
is carried from the annotation row through preprocessing into every batch.

Single-eye policy (verified real-data reality, 2026-08):
    Every annotation row contains both eye landmarks, so BOTH eye patches are
    cropped for EVERY sample. The annotated ``eye_side`` (field 28) records
    which eye the gaze label refers to and is preserved as metadata; it is a
    label attribute, never a filter. Subject p14 annotates only the left eye
    and 21 of 519 subject/sessions carry one side — none of that data is
    dropped or rewritten here. Corpus-wide both sides remain represented.

Leakage policy: LOPO folds are materialized only via
:func:`build_lopo_fold_datasets`, which constructs train/test datasets from
disjoint subject allowlists and verifies sample-ID disjointness before the
datasets are handed out. The dataset itself is read-only over the raw tree;
nothing under the dataset root is ever written.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..preprocessing import PreprocessConfig, PreprocessedSample, RawPreprocessor
from ..splits import LOPOFold
from .raw_adapter import (
    RawAnnotation,
    discover_raw_subjects,
    read_subject_annotations,
)


@dataclass(frozen=True)
class GazeBatch:
    """Deterministic batch structure produced by :func:`collate_gaze_samples`.

    Tensor dtypes mirror the validated preprocessing exactly:
    face/eyes float32 (ImageNet-normalized CHW), gaze/head_pose float64.
    Identity fields keep batch order aligned with the tensor rows.
    """

    face: torch.Tensor        # (B, 3, 224, 224) float32
    left_eye: torch.Tensor    # (B, 3, 36, 60)   float32
    right_eye: torch.Tensor   # (B, 3, 36, 60)   float32
    gaze: torch.Tensor        # (B, 3)           float64 unit vectors
    head_pose: torch.Tensor   # (B, 3)           float64
    sample_ids: list[str] = field(default_factory=list)
    subject_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    source_lines: list[int] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    eye_sides: list[str] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.face.shape[0])

    def __len__(self) -> int:
        return self.batch_size

    def to_dict(self) -> dict:
        return {
            "face": self.face,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "gaze": self.gaze,
            "head_pose": self.head_pose,
            "sample_ids": list(self.sample_ids),
            "subject_ids": list(self.subject_ids),
            "session_ids": list(self.session_ids),
            "source_lines": list(self.source_lines),
            "filenames": list(self.filenames),
            "eye_sides": list(self.eye_sides),
        }


def collate_gaze_samples(samples: Sequence[PreprocessedSample]) -> GazeBatch:
    """Stack preprocessed samples into a :class:`GazeBatch` without altering values."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    head_poses = [s.head_pose for s in samples]
    if any(pose is None for pose in head_poses):
        missing = [s.sample_id for s, pose in zip(samples, head_poses) if pose is None]
        raise ValueError(
            "head_pose missing for raw-layout sample(s); "
            f"annotation fields 16-21 are mandatory: {missing}"
        )
    return GazeBatch(
        face=torch.from_numpy(np.stack([np.asarray(s.face) for s in samples])),
        left_eye=torch.from_numpy(np.stack([np.asarray(s.left_eye) for s in samples])),
        right_eye=torch.from_numpy(np.stack([np.asarray(s.right_eye) for s in samples])),
        gaze=torch.from_numpy(np.stack([np.asarray(s.gaze, dtype=np.float64) for s in samples])),
        head_pose=torch.from_numpy(np_stack(head_poses)),
        sample_ids=[s.sample_id for s in samples],
        subject_ids=[s.subject_id for s in samples],
        session_ids=[str(s.meta["session_id"]) for s in samples],
        source_lines=[int(s.meta["source_line"]) for s in samples],
        filenames=[str(s.meta["filename"]) for s in samples],
        eye_sides=[str(s.meta["eye_side"]) for s in samples],
    )


def np_stack(arrays: Sequence):
    return np.stack([np.asarray(a) for a in arrays], axis=0)


_ANNOTATION_CACHE: dict = {}


def _load_subject_annotations(subject_dir: Path):
    """Parse one subject's annotation file with an mtime/size-keyed memo cache.

    LOPO tooling rebuilds datasets per fold, re-parsing the same immutable
    rows up to 15x. ``RawAnnotation`` is frozen, so cached rows are safe to
    share; the cache key includes size and nanosecond mtime so any edit to an
    annotation file invalidates its entries. This caches identity data only —
    no image bytes and nothing derived across samples.
    """
    txt_path = subject_dir / f"{subject_dir.name}.txt"
    try:
        stat = txt_path.stat()
        key = (str(txt_path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in _ANNOTATION_CACHE:
        return _ANNOTATION_CACHE[key]
    rows, problems = read_subject_annotations(subject_dir)
    payload = (tuple(rows), tuple(problems))
    if key is not None:
        _ANNOTATION_CACHE[key] = payload
    return payload


class RawMPIIFaceGazeDataset(Dataset):
    """Lazy map-style dataset over the raw-txt MPIIFaceGaze layout.

    Args:
        root: dataset root; when omitted the configured registry entry
            (``config/datasets.json`` / env override) is resolved.
        subjects: explicit subject allowlist. Unknown ids raise ``KeyError``.
            Omit to load every discovered subject.
        config: preprocessing configuration passed through to
            :class:`RawPreprocessor` unchanged.

    Indexing returns the untouched :class:`PreprocessedSample`; access is
    deterministic and order-independent because preprocessing is stateless.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        subjects: Sequence[str] | None = None,
        config: PreprocessConfig | None = None,
        dataset_name: str = "mpii_facegaze",
        require_images: bool = True,
    ) -> None:
        """See class docstring.

        ``require_images=False`` skips the annotated-image existence sweep
        (used by split-layer tooling that only inspects identity/splits);
        preprocessing still fails loudly on unreadable files. The default
        ``True`` keeps construction fail-fast for training use.
        """
        from ..config import resolve_dataset_root

        if root is None:
            root = resolve_dataset_root(dataset_name).root
        self.root = Path(root)

        discovered = discover_raw_subjects(self.root)
        if subjects is None:
            selected = list(discovered)
        else:
            unknown = [s for s in subjects if s not in discovered]
            if unknown:
                raise KeyError(
                    f"Unknown subject(s) {unknown}; available: {discovered}"
                )
            selected = [s for s in discovered if s in set(subjects)]

        annotations: list[RawAnnotation] = []
        malformed: list[str] = []
        missing: list[str] = []
        for subject in selected:
            rows, problems = _load_subject_annotations(self.root / subject)
            malformed.extend(f"{subject}:{p.source_line}: {p.reason}" for p in problems)
            for row in rows:
                if require_images and not row.image_path.exists():
                    missing.append(str(row.image_path))
                annotations.append(row)
        if malformed:
            raise ValueError(f"malformed annotation rows: {malformed[:5]}")
        if missing:
            raise FileNotFoundError(f"{len(missing)} annotated images missing, e.g. {missing[0]}")

        self._annotations = tuple(annotations)
        self._subjects = tuple(selected)
        self._sample_ids = tuple(
            f"{a.subject_id}:{a.source_line}" for a in annotations
        )
        self._preprocessor = RawPreprocessor(config or PreprocessConfig())

    @property
    def subjects(self) -> tuple[str, ...]:
        return self._subjects

    @property
    def annotations(self) -> tuple[RawAnnotation, ...]:
        """Read-only annotation rows backing this dataset (no image I/O)."""
        return self._annotations

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> PreprocessedSample:
        if isinstance(index, slice):
            raise TypeError("RawMPIIFaceGazeDataset supports integer indexing only")
        return self._preprocessor.process_raw(self._annotations[index])


def build_lopo_fold_datasets(
    fold: LOPOFold,
    root: str | Path | None = None,
    config: PreprocessConfig | None = None,
    require_images: bool = True,
) -> tuple[RawMPIIFaceGazeDataset, RawMPIIFaceGazeDataset]:
    """Materialize (train_dataset, test_dataset) for one LOPO fold.

    Train/test subject sets come straight from the fold; sample-ID disjointness
    is verified here so a mis-built fold fails loudly instead of leaking the
    held-out subject into training.
    """
    train = RawMPIIFaceGazeDataset(
        root=root, subjects=fold.train_subjects, config=config, require_images=require_images
    )
    test = RawMPIIFaceGazeDataset(
        root=root, subjects=fold.test_subjects, config=config, require_images=require_images
    )
    assert_fold_disjointness(train, test)
    return train, test


def assert_fold_disjointness(
    train: RawMPIIFaceGazeDataset,
    test: RawMPIIFaceGazeDataset,
) -> None:
    """Raise unless train/test share neither subjects nor sample IDs."""
    shared_subjects = sorted(set(train.subjects) & set(test.subjects))
    if shared_subjects:
        raise ValueError(f"held-out subject(s) present in training: {shared_subjects}")
    overlap = sorted(set(train.sample_ids) & set(test.sample_ids))
    if overlap:
        raise ValueError(
            f"{len(overlap)} sample ID(s) appear in both train and test, e.g. {overlap[:5]}"
        )


def make_gaze_dataloader(
    dataset: RawMPIIFaceGazeDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    **loader_kwargs,
) -> DataLoader:
    """DataLoader with the deterministic gaze collate wired in.

    Shuffling is deliberately a caller decision (training-time concern); the
    dataset itself never reorders samples.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gaze_samples,
        **loader_kwargs,
    )
