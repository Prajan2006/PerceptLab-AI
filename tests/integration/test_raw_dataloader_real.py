"""Real-data integration tests for the raw-layout dataset/DataLoader layer.

Skipped automatically when the configured MPIIFaceGaze root is absent.
Verifies the PyTorch layer against the actual corpus: length, unique sample
IDs, LOPO fold construction (train/test disjoint, held-out exactly once),
leakage guards, and real DataLoader batches cross-checked against the
underlying annotation rows.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Subset

from data.mpiifacegaze.raw_adapter import discover_raw_subjects
from data.mpiifacegaze.raw_dataset import (
    RawMPIIFaceGazeDataset,
    assert_fold_disjointness,
    build_lopo_fold_datasets,
    make_gaze_dataloader,
)
from data.splits import lopo_folds

EXPECTED_SUBJECTS = [f"p{i:02d}" for i in range(15)]
EXPECTED_CORPUS_SIZE = 37667


def _dataset_available() -> bool:
    try:
        from data.config import resolve_dataset_root

        return resolve_dataset_root("mpii_facegaze").root.exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _dataset_available(), reason="real MPIIFaceGaze dataset not configured"
)


@pytest.fixture(scope="module")
def corpus():
    """One strict full-corpus construction; folds partition these subjects."""
    return RawMPIIFaceGazeDataset()


class TestRealCorpusDataset:
    def test_discovers_all_subjects(self, corpus):
        assert list(corpus.subjects) == EXPECTED_SUBJECTS

    def test_corpus_length_is_37667(self, corpus):
        assert len(corpus) == EXPECTED_CORPUS_SIZE

    def test_sample_ids_unique(self, corpus):
        ids = corpus.sample_ids
        assert len(set(ids)) == EXPECTED_CORPUS_SIZE

    def test_getitem_matches_annotation_row(self, corpus):
        for index in (0, 12345, len(corpus) - 1):
            annotation = corpus.annotations[index]
            sample = corpus[index]
            assert sample.sample_id == f"{annotation.subject_id}:{annotation.source_line}"
            assert sample.meta["filename"] == annotation.relative_path
            assert sample.meta["session_id"] == annotation.session_id
            assert sample.face.shape == (3, 224, 224)
            assert sample.left_eye.shape == sample.right_eye.shape == (3, 36, 60)


class TestRealLopoFolds:
    def test_fifteen_folds_held_out_exactly_once(self, corpus):
        subjects = list(corpus.subjects)
        folds = lopo_folds(subjects)
        held_out = [f.test_subjects[0] for f in folds]
        assert sorted(held_out) == EXPECTED_SUBJECTS
        assert len(set(held_out)) == 15

    def test_every_fold_test_ids_partition_corpus(self, corpus):
        """Across all folds each real sample ID becomes a test ID exactly once."""
        folds = lopo_folds(list(corpus.subjects))
        tested: list[str] = []
        for fold in folds:
            subject = fold.test_subjects[0]
            subject_indices = [
                i for i, a in enumerate(corpus.annotations) if a.subject_id == subject
            ]
            tested.extend(corpus.sample_ids[i] for i in subject_indices)
            train_ids = {
                corpus.sample_ids[i]
                for i, a in enumerate(corpus.annotations)
                if a.subject_id in set(fold.train_subjects)
            }
            test_ids = set(tested[-len(subject_indices):])
            assert test_ids.isdisjoint(train_ids), f"fold {subject} leaked"

        assert len(tested) == len(corpus.sample_ids)
        assert set(tested) == set(corpus.sample_ids)

    def test_factory_end_to_end_on_real_fold(self, corpus):
        """Full factory path incl. image sweep for one representative fold."""
        fold = next(f for f in lopo_folds(list(corpus.subjects)) if f.test_subjects[0] == "p07")
        train, test = build_lopo_fold_datasets(fold)

        assert test.subjects == ("p07",)
        assert set(train.subjects) == set(fold.train_subjects)
        assert not (set(train.sample_ids) & set(test.sample_ids))
        assert len(train) + len(test) == EXPECTED_CORPUS_SIZE
        # explicit requirement-11 check: held-out subject never in training
        assert all(not sid.startswith("p07:") for sid in train.sample_ids)
        assert all(sid.startswith("p07:") for sid in test.sample_ids)

    def test_factory_covers_left_only_subject_fold(self, corpus):
        """p14 (left-eye-only annotations) still forms a valid disjoint fold."""
        fold = next(f for f in lopo_folds(list(corpus.subjects)) if f.test_subjects[0] == "p14")
        train, test = build_lopo_fold_datasets(fold, require_images=False)
        assert_fold_disjointness(train, test)
        assert test.subjects == ("p14",)
        sides = {a.eye_side for a in test.annotations}
        assert sides == {"left"}

    def test_all_folds_construct_disjoint_fast(self, corpus):
        """Every fold through the factory; identity-only (images proven above)."""
        for fold in lopo_folds(list(corpus.subjects)):
            train, test = build_lopo_fold_datasets(fold, require_images=False)
            assert not (set(train.sample_ids) & set(test.sample_ids)), fold.fold_index
            assert len(train) + len(test) == EXPECTED_CORPUS_SIZE


@pytest.fixture(scope="module")
def smoke_batches(corpus):
    """p00/p01/p02, day01+day02, first left/right per session, batch_size=4."""
    subset_indices = []
    wanted_sessions = {"day01", "day02"}
    seen: dict[tuple[str, str], set[str]] = {}
    for subject in ("p00", "p01", "p02"):
        rows = [
            (i, a)
            for i, a in enumerate(corpus.annotations)
            if a.subject_id == subject and a.session_id in wanted_sessions
        ]
        by_session: dict[str, list] = {}
        for i, a in rows:
            by_session.setdefault(a.session_id, []).append((i, a))
        for session in sorted(by_session):
            group = by_session[session]
            first_left = next(((i, a) for i, a in group if a.eye_side == "left"), None)
            first_right = next(((i, a) for i, a in group if a.eye_side == "right"), None)
            for pick in (first_left, first_right):
                if pick is None:
                    continue
                i, a = pick
                key = (subject, session)
                seen.setdefault(key, set())
                if a.eye_side not in seen[key]:
                    seen[key].add(a.eye_side)
                    subset_indices.append(i)
    dataset = Subset(corpus, subset_indices)
    return make_gaze_dataloader(dataset, batch_size=4), corpus


class TestRealDataLoaderSmoke:
    """p00/p01/p02, multiple sessions, both eye sides, batch_size=4."""

    def test_at_least_three_batches_with_expected_shapes(self, smoke_batches):
        loader, _ = smoke_batches
        batches = list(loader)
        assert len(batches) >= 3
        assert all(batch.batch_size == 4 for batch in batches)
        for batch in batches:
            assert batch.face.shape == (batch.batch_size, 3, 224, 224)
            assert batch.left_eye.shape == batch.right_eye.shape == (
                batch.batch_size, 3, 36, 60)
            assert batch.gaze.shape == (batch.batch_size, 3)
            norms = batch.gaze.pow(2).sum(dim=1).sqrt()
            assert bool(((norms - 1.0).abs() < 1e-9).all())
            finite = all(
                bool(torch.isfinite(t).all())
                for t in (batch.face, batch.left_eye, batch.right_eye, batch.gaze, batch.head_pose)
            )
            assert finite
            assert {s.split(":")[0] for s in batch.sample_ids} <= {"p00", "p01", "p02"}

    def test_batch_metadata_matches_actual_annotations(self, smoke_batches):
        loader, corpus = smoke_batches
        checked = 0
        root = corpus.root
        for batch in loader:
            for row in range(batch.batch_size):
                subject = batch.subject_ids[row]
                line = batch.source_lines[row]
                with (root / subject / f"{subject}.txt").open(
                    encoding="utf-8", errors="replace"
                ) as handle:
                    for number, raw in enumerate(handle, start=1):
                        if number == line:
                            tokens = raw.strip().split()
                            assert tokens[0] == batch.filenames[row]
                            assert batch.session_ids[row] == tokens[0].split("/")[0]
                            assert tokens[27] == batch.eye_sides[row]
                            break
                    else:
                        pytest.fail(f"line {line} missing in {subject}.txt")
                checked += 1
        assert checked >= 12
