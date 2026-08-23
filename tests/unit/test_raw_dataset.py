"""Unit tests for the raw-layout PyTorch dataset/batching layer (synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.mpiifacegaze.raw_dataset import (
    GazeBatch,
    RawMPIIFaceGazeDataset,
    assert_fold_disjointness,
    build_lopo_fold_datasets,
    collate_gaze_samples,
    make_gaze_dataloader,
)
from data.mpiifacegaze.raw_synthetic import (
    _write_tiny_jpeg,
    build_synthetic_raw_subject,
    format_annotation_row,
)
from data.preprocessing import PreprocessedSample
from data.splits import lopo_folds

SUBJECTS = ("p00", "p01")
SESSIONS = ("day01", "day02")
FRAMES = 3


@pytest.fixture()
def raw_root(tmp_path):
    root = tmp_path / "Data"
    for subject in SUBJECTS:
        build_synthetic_raw_subject(
            root, subject, sessions=SESSIONS, frames_per_session=FRAMES
        )
    return root


@pytest.fixture()
def dataset(raw_root):
    return RawMPIIFaceGazeDataset(root=raw_root)


class TestIndexing:
    def test_len_covers_all_annotations(self, dataset):
        assert len(dataset) == len(SUBJECTS) * len(SESSIONS) * FRAMES

    def test_getitem_returns_preprocessed_sample(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, PreprocessedSample)
        assert sample.face.shape == (3, 224, 224)
        assert sample.left_eye.shape == (3, 36, 60)
        assert sample.right_eye.shape == (3, 36, 60)

    def test_slice_indexing_rejected(self, dataset):
        with pytest.raises(TypeError):
            dataset[0:2]


class TestIdentityPreservation:
    def test_sample_id_matches_annotation(self, dataset):
        for index, annotation in enumerate(dataset.annotations):
            sample = dataset[index]
            assert sample.sample_id == f"{annotation.subject_id}:{annotation.source_line}"

    def test_metadata_matches_annotation(self, dataset):
        sample = dataset[1]
        annotation = dataset.annotations[1]
        assert sample.subject_id == annotation.subject_id == "p00"
        assert sample.meta["session_id"] == annotation.session_id
        assert sample.meta["source_line"] == annotation.source_line
        assert sample.meta["filename"] == annotation.relative_path
        assert sample.meta["eye_side"] == annotation.eye_side
        assert sample.sample_id in dataset.sample_ids

    def test_subjects_property_sorted(self, dataset):
        assert dataset.subjects == tuple(sorted(SUBJECTS))


class TestSubjectFiltering:
    def test_filter_keeps_only_requested_subject(self, raw_root):
        single = RawMPIIFaceGazeDataset(root=raw_root, subjects=("p01",))
        assert single.subjects == ("p01",)
        assert len(single) == len(SESSIONS) * FRAMES
        assert all(sid.startswith("p01:") for sid in single.sample_ids)

    def test_unknown_subject_rejected(self, raw_root):
        with pytest.raises(KeyError):
            RawMPIIFaceGazeDataset(root=raw_root, subjects=("p42",))


class TestLopoDisjointness:
    def test_factory_builds_disjoint_train_test(self, raw_root):
        fold = lopo_folds(list(SUBJECTS))[0]
        train, test = build_lopo_fold_datasets(fold, root=raw_root)

        assert set(fold.test_subjects) == set(test.subjects)
        assert set(train.subjects).isdisjoint(test.subjects)
        assert not (set(train.sample_ids) & set(test.sample_ids))
        assert set(train.sample_ids) | set(test.sample_ids) == {
            f"{s}:{i}"
            for s in SUBJECTS
            for i in range(1, len(SESSIONS) * FRAMES + 1)
        }

    def test_held_out_subject_exclusively_in_test(self, raw_root):
        fold = lopo_folds(list(SUBJECTS))[0]
        held_out = fold.test_subjects[0]
        train, test = build_lopo_fold_datasets(fold, root=raw_root)

        assert held_out not in train.subjects
        assert all(sid.startswith(held_out) for sid in test.sample_ids)
        assert all(not sid.startswith(held_out) for sid in train.sample_ids)

    def test_overlapping_datasets_raise(self, raw_root):
        first = RawMPIIFaceGazeDataset(root=raw_root)
        second = RawMPIIFaceGazeDataset(root=raw_root, subjects=("p00",))
        with pytest.raises(ValueError):
            assert_fold_disjointness(first, second)


class TestDeterminism:
    def test_repeated_access_bitwise_identical(self, dataset):
        first = dataset[3]
        second = dataset[3]
        assert np.array_equal(first.face, second.face)
        assert np.array_equal(first.left_eye, second.left_eye)
        assert np.array_equal(first.right_eye, second.right_eye)
        assert np.array_equal(first.gaze, second.gaze)

    def test_access_order_does_not_alter_contents(self, dataset):
        forward = {dataset[i].sample_id: dataset[i].face.copy() for i in range(len(dataset))}
        backward = {
            dataset[i].sample_id: dataset[i].face.copy()
            for i in range(len(dataset) - 1, -1, -1)
        }
        assert forward.keys() == backward.keys()
        assert all(np.array_equal(forward[k], backward[k]) for k in forward)

    def test_fresh_instance_same_contents(self, raw_root, dataset):
        other = RawMPIIFaceGazeDataset(root=raw_root)
        for i in (0, 4, len(dataset) - 1):
            assert np.array_equal(dataset[i].face, other[i].face)
            assert dataset[i].sample_id == other[i].sample_id


class TestBatching:
    def test_dataloader_batch_shapes_dtypes_identity(self, dataset):
        loader = make_gaze_dataloader(dataset, batch_size=5)
        batches = list(loader)

        assert [len(b) for b in batches] == [5, 5, 2]
        for batch in batches:
            assert isinstance(batch, GazeBatch)
            assert batch.face.shape == (batch.batch_size, 3, 224, 224)
            assert batch.left_eye.shape == batch.right_eye.shape == (
                batch.batch_size, 3, 36, 60)
            assert batch.gaze.shape == batch.head_pose.shape == (batch.batch_size, 3)
            assert batch.face.dtype == torch.float32
            assert batch.left_eye.dtype == batch.right_eye.dtype == torch.float32
            assert batch.gaze.dtype == batch.head_pose.dtype == torch.float64
            norms = torch_norms(batch.gaze)
            assert torch_all_close(norms, 1.0)
            assert len(batch.sample_ids) == batch.batch_size

    def test_batch_order_follows_dataset_order_without_shuffle(self, dataset):
        batch = next(iter(make_gaze_dataloader(dataset, batch_size=4)))
        assert batch.sample_ids == list(dataset.sample_ids[:4])

    def test_collate_preserves_values(self, dataset):
        samples = [dataset[0], dataset[len(dataset) - 1]]
        batch = collate_gaze_samples(samples)
        assert batch.sample_ids == [s.sample_id for s in samples]
        assert np.array_equal(batch.face[1].numpy(), samples[1].face)
        assert np.array_equal(batch.gaze[1].numpy(), samples[1].gaze)
        assert batch.session_ids == [samples[0].meta["session_id"], samples[1].meta["session_id"]]

    def test_collate_rejects_missing_head_pose(self, dataset):
        sample = dataset[0]
        broken = PreprocessedSample(
            sample_id=sample.sample_id,
            subject_id=sample.subject_id,
            face=sample.face,
            left_eye=sample.left_eye,
            right_eye=sample.right_eye,
            gaze=sample.gaze,
            head_pose=None,
            meta=dict(sample.meta),
        )
        with pytest.raises(ValueError):
            collate_gaze_samples([broken])


class TestSingleEyePolicy:
    def test_left_only_subject_kept_with_both_patches(self, tmp_path):
        """p14-style single-eye rows are never dropped; eye_side is metadata."""
        root = tmp_path / "Data"
        subject_dir = root / "p14"
        rows = []
        sequence = 0
        for session in SESSIONS:
            for frame in range(FRAMES):
                filename = f"{frame:04d}.jpg"
                _write_tiny_jpeg(subject_dir / session / filename, sequence * 10)
                rows.append(format_annotation_row(session, filename, sequence, eye_side="left"))
                sequence += 1
        (subject_dir / "p14.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

        dataset = RawMPIIFaceGazeDataset(root=root)
        assert len(dataset) == len(SESSIONS) * FRAMES
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample.left_eye is not None and sample.left_eye.shape == (3, 36, 60)
            assert sample.right_eye is not None and sample.right_eye.shape == (3, 36, 60)
            assert sample.meta["eye_side"] == "left"


class TestFailFastConstruction:
    def test_malformed_rows_raise(self, tmp_path):
        root = build_synthetic_raw_subject(
            tmp_path / "Data",
            "p05",
            include_malformed_row=True,
            include_artifacts=False,
        )
        with pytest.raises(ValueError):
            RawMPIIFaceGazeDataset(root=root.parent)

    def test_missing_images_raise(self, tmp_path):
        root = build_synthetic_raw_subject(
            tmp_path / "Data",
            "p06",
            sessions=("day01",),
            include_missing_image=True,
            include_artifacts=False,
        )
        with pytest.raises(FileNotFoundError):
            RawMPIIFaceGazeDataset(root=root.parent)


def torch_norms(tensor):
    return tensor.pow(2).sum(dim=1).sqrt()


def torch_all_close(actual, expected, atol=1e-9):
    return bool((actual - expected).abs().max() <= atol)
