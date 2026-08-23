"""Unit tests for deterministic LOPO split generation."""

import json

import pytest

from data.splits import load_lopo_folds, lopo_folds, save_lopo_folds

SUBJECTS = ["p00", "p01", "p02"]


class TestLopoFolds:
    def test_one_fold_per_subject(self):
        folds = lopo_folds(SUBJECTS)
        assert len(folds) == 3

    def test_deterministic_ordering(self):
        shuffled = ["p02", "p00", "p01"]
        assert [f.test_subjects for f in lopo_folds(shuffled)] == [
            ("p00",),
            ("p01",),
            ("p02",),
        ]

    def test_train_test_disjoint_and_complete(self):
        folds = lopo_folds(SUBJECTS)
        for fold in folds:
            assert fold.test_subjects not in ([],) and len(fold.train_subjects) == 2
            assert set(fold.test_subjects).isdisjoint(fold.train_subjects)
            assert set(fold.test_subjects) | set(fold.train_subjects) == set(SUBJECTS)

    def test_every_subject_held_out_exactly_once(self):
        held_out = [fold.test_subjects[0] for fold in lopo_folds(SUBJECTS)]
        assert sorted(held_out) == SUBJECTS

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError):
            lopo_folds(["p00", "p00", "p01"])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            lopo_folds([])


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        folds = lopo_folds(SUBJECTS)
        path = tmp_path / "splits" / "lopo.json"
        save_lopo_folds(folds, path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["protocol"] == "leave-one-person-out"

        loaded = load_lopo_folds(path)
        assert loaded == folds

    def test_load_rejects_wrong_protocol(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"protocol": "random", "folds": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_lopo_folds(path)
