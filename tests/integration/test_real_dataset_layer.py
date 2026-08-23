"""Integration tests: dataset/split layer against the REAL MPIIFaceGaze corpus.

Skipped automatically when the configured dataset root is absent, so unit
suites and CI without the dataset still run. These tests lock in the
critical invariants that synthetic fixtures cannot prove:

- all 15 subjects are discovered via config-driven root resolution
- every annotation row parses cleanly and its image exists on disk
- LOPO folds built from the real subject set keep test/train disjoint at
  sample-ID level, and the per-fold test sets partition the whole corpus
"""

from __future__ import annotations

import pytest

from data.config import resolve_dataset_root
from data.mpiifacegaze.raw_adapter import discover_raw_subjects, read_subject_annotations
from data.splits import lopo_folds

EXPECTED_SUBJECTS = [f"p{i:02d}" for i in range(15)]


def _dataset_available() -> bool:
    try:
        return resolve_dataset_root("mpii_facegaze").root.exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _dataset_available(), reason="real MPIIFaceGaze dataset not configured"
)


@pytest.fixture(scope="module")
def real_root():
    return resolve_dataset_root("mpii_facegaze").root


@pytest.fixture(scope="module")
def parsed_corpus(real_root):
    subjects = discover_raw_subjects(real_root)
    corpus = {}
    for subject in subjects:
        annotations, malformed = read_subject_annotations(real_root / subject)
        corpus[subject] = (annotations, malformed)
    return subjects, corpus


class TestRealSubjectDiscovery:
    def test_discovers_all_15_subjects(self, parsed_corpus):
        subjects, _ = parsed_corpus
        assert subjects == EXPECTED_SUBJECTS


class TestRealAnnotationParsing:
    def test_every_row_parses_cleanly_across_all_subjects(self, parsed_corpus):
        _, corpus = parsed_corpus
        for subject, (annotations, malformed) in corpus.items():
            assert not malformed, f"{subject}: {malformed[:3]}"
            assert len(annotations) > 0

    def test_total_records_and_unique_sample_ids(self, parsed_corpus):
        _, corpus = parsed_corpus
        sample_ids = [
            f"{a.subject_id}:{a.source_line}"
            for (annotations, _) in corpus.values()
            for a in annotations
        ]
        assert len(sample_ids) == 37667
        assert len(set(sample_ids)) == len(sample_ids)

    def test_every_annotated_image_exists(self, parsed_corpus):
        _, corpus = parsed_corpus
        missing = [
            str(a.image_path)
            for (annotations, _) in corpus.values()
            for a in annotations
            if not a.image_path.exists()
        ]
        assert missing == []

    def test_both_eye_sides_represented(self, parsed_corpus):
        """Corpus-wide both sides exist; every subject contributes both sides.

        Known real-dataset property (verified 2026-08): subject p14 annotates
        only the left eye, and 21 of 519 subject/sessions carry a single side.
        Session-level symmetry is therefore NOT asserted; if the allowlist
        below starts matching other subjects, the dataset changed.
        """
        _, corpus = parsed_corpus
        sides_by_subject: dict[str, set[str]] = {}
        sides_by_session: dict[tuple[str, str], set[str]] = {}
        for (annotations, _) in corpus.values():
            for a in annotations:
                sides_by_subject.setdefault(a.subject_id, set()).add(a.eye_side)
                sides_by_session.setdefault((a.subject_id, a.session_id), set()).add(a.eye_side)

        assert {"left", "right"} <= set().union(*sides_by_subject.values())
        single_side_subjects = {s for s, v in sides_by_subject.items() if len(v) < 2}
        assert single_side_subjects == {"p14"}, (
            f"unexpected single-side subjects: {sorted(single_side_subjects)}"
        )
        single_side_sessions = {k for k, v in sides_by_session.items() if len(v) < 2}
        assert {s for s, _ in single_side_sessions} <= {"p01", "p04", "p13", "p14"}


class TestRealLopoProtocol:
    def test_fifteen_deterministic_folds(self, parsed_corpus):
        subjects, _ = parsed_corpus
        folds = lopo_folds(subjects)
        assert [f.test_subjects[0] for f in folds] == EXPECTED_SUBJECTS
        assert len({f.fold_index for f in folds}) == 15

    def test_within_fold_sample_ids_disjoint(self, parsed_corpus):
        subjects, corpus = parsed_corpus
        folds = lopo_folds(subjects)
        ids_by_subject: dict[str, set[str]] = {
            subject: {f"{a.subject_id}:{a.source_line}" for (annotations, _) in [corpus[subject]]
                      for a in annotations}
            for subject in subjects
        }
        for fold in folds:
            train_ids: set[str] = set()
            for subject in fold.train_subjects:
                train_ids |= ids_by_subject[subject]
            test_ids = ids_by_subject[fold.test_subjects[0]]
            assert test_ids.isdisjoint(train_ids), f"fold {fold.test_subjects} leaked"

    def test_test_sets_partition_corpus_exactly_once(self, parsed_corpus):
        subjects, corpus = parsed_corpus
        folds = lopo_folds(subjects)
        all_ids = {
            f"{a.subject_id}:{a.source_line}"
            for (annotations, _) in corpus.values()
            for a in annotations
        }
        tested: list[str] = []
        for fold in folds:
            tested.extend(sorted({
                f"{a.subject_id}:{a.source_line}"
                for a in corpus[fold.test_subjects[0]][0]
            }))
        assert len(tested) == len(all_ids)
        assert set(tested) == all_ids

    def test_train_union_covers_rest_of_corpus(self, parsed_corpus):
        subjects, corpus = parsed_corpus
        folds = lopo_folds(subjects)
        for fold in folds:
            train_subjects = set(fold.train_subjects)
            expected_n = sum(len(corpus[s][0]) for s in train_subjects)
            assert expected_n > 0
            assert set(fold.train_subjects).isdisjoint(fold.test_subjects)
