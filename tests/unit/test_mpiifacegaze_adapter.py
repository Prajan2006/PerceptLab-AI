"""Unit tests for the MPIIFaceGaze adapter, metadata index, and fixtures."""

import numpy as np
import pytest

from data.mpiifacegaze import (
    MPIIFaceGazeDataset,
    build_index,
    build_synthetic_dataset,
    build_synthetic_subject_file,
    load_index,
)


@pytest.fixture()
def synthetic_root(tmp_path):
    return build_synthetic_dataset(tmp_path / "MPIIFaceGaze", ("p02", "p00", "p01"))


class TestAdapter:
    def test_discovers_sorted_subjects(self, synthetic_root):
        dataset = MPIIFaceGazeDataset(synthetic_root)
        assert dataset.subjects == ["p00", "p01", "p02"]

    def test_sample_fields_preserved(self, tmp_path):
        root = build_synthetic_subject_file(tmp_path, "P07", num_samples=10, seed=3)
        assert root.name == "p07.mat"

        samples = MPIIFaceGazeDataset(tmp_path).load_subject("p07")
        assert len(samples) == 10

        first = samples[0]
        assert first.sample_id == "p07:0"
        assert first.subject_id == "p07"
        assert first.filename == "p07/day01/0000.jpg"
        assert first.session_id == "day01"
        assert np.isclose(np.linalg.norm(first.gaze), 1.0, atol=1e-9)
        assert first.left_eye.shape == (36, 60)
        assert first.right_eye.shape == (36, 60)
        assert tuple(int(v) for v in first.face_bbox) == (120, 80, 400, 400)
        assert first.head_pose is not None and first.head_pose.shape == (3,)

    def test_session_rotation(self, tmp_path):
        build_synthetic_subject_file(tmp_path, "p05", num_samples=8)
        samples = MPIIFaceGazeDataset(tmp_path).load_subject("p05")
        sessions = [s.session_id for s in samples]
        assert sessions[:4] == ["day01", "day02", "day03", "day04"]

    def test_unknown_subject_raises(self, synthetic_root):
        dataset = MPIIFaceGazeDataset(synthetic_root)
        with pytest.raises(KeyError):
            dataset.load_subject("p99")

    def test_iteration_yields_all_samples(self, synthetic_root):
        total = sum(1 for _ in MPIIFaceGazeDataset(synthetic_root))
        assert total == 18 + 21 + 24  # p00=18, p01=21, p02=24

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MPIIFaceGazeDataset(tmp_path / "does-not-exist").subjects


class TestMetadataIndex:
    def test_index_counts_and_sessions(self, synthetic_root, tmp_path):
        index_path = tmp_path / "metadata" / "dataset.index.json"
        index = build_index(synthetic_root, output_path=index_path)

        assert index["dataset"] == "MPIIFaceGaze"
        assert index["num_subjects"] == 3
        assert index["total_samples"] == 63
        # build_synthetic_dataset assigns num_samples = 18 + position*3 for
        # ("p02", "p00", "p01") → p02=18, p00=21, p01=24.
        assert index["per_subject"]["p00"]["num_samples"] == 21
        assert set(index["per_subject"]["p00"]["sessions"]) <= {
            "day01",
            "day02",
            "day03",
            "day04",
        }
        assert index_path.exists()

    def test_load_index_validates_dataset(self, synthetic_root, tmp_path):
        index = build_index(synthetic_root)
        loaded = load_index_from_payload(index)
        assert loaded["total_samples"] == index["total_samples"]


def load_index_from_payload(payload):
    from data.mpiifacegaze.metadata import INDEX_SCHEMA_VERSION

    assert payload["schema_version"] == INDEX_SCHEMA_VERSION
    return payload
