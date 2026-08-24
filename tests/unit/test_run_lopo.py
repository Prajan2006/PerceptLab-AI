"""Focused runner tests on synthetic data — no real 15-fold training occurs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from data.mpiifacegaze import build_lopo_fold_datasets, build_synthetic_raw_subject
from data.splits import lopo_folds
from scripts.train_lopo import (
    SMOKE_RUN_DIR_NAME,
    fold_dir,
    is_fold_complete,
    run_lopo,
)
from training import load_engineering_baseline

SUBJECTS = ("p00", "p01", "p02")


class TinyGaze(nn.Module):
    """Shape-compatible stand-in for ResNet50Gaze (face (B,3,224,224) -> (B,3))."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 3)

    def forward(self, face):
        return self.fc(face.flatten(start_dim=1))


def _build_synthetic_root(base: Path) -> Path:
    root = base / "Data"
    for subject in SUBJECTS:
        build_synthetic_raw_subject(
            root,
            subject,
            sessions=("day01",),
            frames_per_session=2,
            include_artifacts=False,
        )
    return root


def _synthetic_factory(synthetic_root: Path):
    """Serve any LOPO fold synthetically, creating subjects on first request."""

    def factory(fold, root=None):
        needed = set(fold.train_subjects) | set(fold.test_subjects)
        for subject in needed:
            if not (synthetic_root / subject / f"{subject}.txt").exists():
                build_synthetic_raw_subject(
                    synthetic_root,
                    subject,
                    sessions=("day01",),
                    frames_per_session=2,
                    include_artifacts=False,
                )
        return build_lopo_fold_datasets(fold, root=synthetic_root)

    return factory


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    return _build_synthetic_root(tmp_path_factory.mktemp("lopo_runner"))


def _run(output_root: Path, synthetic_root: Path, **overrides):
    return run_lopo(
        output_root=output_root,
        seed=overrides.pop("seed", 0),
        subjects=list(overrides.pop("subjects", SUBJECTS)),
        model_builder=overrides.pop("model_builder", TinyGaze),
        fold_datasets_factory=_synthetic_factory(synthetic_root),
        progress=lambda message: None,
        **overrides,
    )


class TestEnumeration:
    def test_deterministic_subject_sorted_order(self, tmp_path, synthetic_root):
        summary = _run(tmp_path / "out", synthetic_root, subjects=["p02", "p00", "p01"])
        assert [f["test_subject"] for f in summary["folds"]] == ["p00", "p01", "p02"]
        assert [f["status"] for f in summary["folds"]] == ["completed"] * 3

    def test_full_protocol_b_enumeration_is_15_folds(self, tmp_path, synthetic_root):
        summary = _run(
            tmp_path / "out15",
            synthetic_root,
            subjects=[f"p{i:02d}" for i in range(15)],
        )
        assert summary["counts"] == {"completed": 15, "skipped": 0, "failed": 0}
        for index in range(15):
            assert fold_dir(Path(summary["output_root"]), lopo_folds(
                [f"p{i:02d}" for i in range(15)])[index]).is_dir()


class TestFoldIsolation:
    def test_per_fold_directories_are_isolated(self, tmp_path, synthetic_root):
        summary = _run(tmp_path / "out", synthetic_root)
        root = Path(summary["output_root"])
        dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        assert dirs == ["fold_p00", "fold_p01", "fold_p02"]
        for name in dirs:
            fold_files = {p.name for p in (root / name).iterdir()}
            assert {"checkpoint.pt", "history.json", "metrics.json"} <= fold_files


class TestProtocolBPropagation:
    def test_metrics_record_frozen_settings_exactly(self, tmp_path, synthetic_root):
        _run(tmp_path / "out", synthetic_root)
        metrics = json.loads((tmp_path / "out" / "fold_p00" / "metrics.json").read_text())
        baseline = load_engineering_baseline()
        frozen = baseline.document["settings"]

        assert metrics["loss"] == frozen["loss"]["value"]
        assert metrics["optimizer"] == frozen["optimizer"]["type"]
        assert metrics["scheduler"] == frozen["scheduler"]["type"]
        assert metrics["learning_rate"] == frozen["optimizer"]["lr"]
        assert metrics["weight_decay"] == frozen["optimizer"]["weight_decay"]
        assert metrics["epochs"] == frozen["epochs"]["value"] == 1
        assert metrics["batch_size"] == frozen["batch_size"]["value"] == 8
        assert metrics["seed"] == 0

    def test_epochs_one_propagates_to_history_length(self, tmp_path, synthetic_root):
        _run(tmp_path / "out", synthetic_root)
        for subject in SUBJECTS:
            history = json.loads(
                (tmp_path / "out" / f"fold_{subject}" / "history.json").read_text()
            )
            assert len(history) == 1


class TestSeedReproducibility:
    def test_same_seed_identical_histories(self, tmp_path, synthetic_root):
        _run(tmp_path / "run_a", synthetic_root)
        _run(tmp_path / "run_b", synthetic_root)

        for subject in SUBJECTS:
            a = json.loads((tmp_path / "run_a" / f"fold_{subject}" / "history.json").read_text())
            b = json.loads((tmp_path / "run_b" / f"fold_{subject}" / "history.json").read_text())
            assert a == b


class TestSkipIfComplete:
    def test_completed_folds_are_skipped_and_not_rewritten(self, tmp_path, synthetic_root):
        _run(tmp_path / "out", synthetic_root)
        checkpoint_before = {
            subject: (tmp_path / "out" / f"fold_{subject}" / "checkpoint.pt").stat()
            for subject in SUBJECTS
        }

        summary = _run(tmp_path / "out", synthetic_root)

        assert summary["counts"] == {"completed": 0, "skipped": 3, "failed": 0}
        for subject in SUBJECTS:
            stat_after = (tmp_path / "out" / f"fold_{subject}" / "checkpoint.pt").stat()
            assert stat_after.st_mtime_ns == checkpoint_before[subject].st_mtime_ns

    def test_partial_completion_only_reruns_missing_fold(self, tmp_path, synthetic_root):
        _run(tmp_path / "out", synthetic_root)
        # corrupt/remove one fold's completion marker
        (tmp_path / "out" / "fold_p01" / "metrics.json").unlink()

        summary = _run(tmp_path / "out", synthetic_root)
        statuses = {f["test_subject"]: f["status"] for f in summary["folds"]}
        assert statuses == {"p00": "skipped", "p01": "completed", "p02": "skipped"}
        assert is_fold_complete(tmp_path / "out", lopo_folds(list(SUBJECTS))[1])


class TestFailureReporting:
    def test_failed_fold_reported_without_silent_success(self, tmp_path, synthetic_root):
        attempts = {"count": 0}

        def failing_builder():
            attempts["count"] += 1
            if attempts["count"] == 2:  # second fold construction fails
                raise RuntimeError("synthetic fold failure")
            return TinyGaze()

        summary = _run(
            tmp_path / "out", synthetic_root, model_builder=failing_builder
        )

        assert summary["counts"]["failed"] == 1
        failed = next(f for f in summary["folds"] if f["status"] == "failed")
        assert failed["test_subject"] == "p01"
        assert "synthetic fold failure" in failed["error"]

        statuses = {f["test_subject"]: f["status"] for f in summary["folds"]}
        assert statuses["p00"] == statuses["p02"] == "completed"
        # failed fold must not be marked complete
        assert not (tmp_path / "out" / "fold_p01" / "metrics.json").exists()


class TestSmokeArtifactGuard:
    def test_refuses_to_write_into_smoke_directory(self, tmp_path):
        with pytest.raises(ValueError):
            run_lopo(
                output_root=tmp_path / SMOKE_RUN_DIR_NAME,
                subjects=list(SUBJECTS),
            )

    def test_default_output_root_is_separate_from_smoke_dir(self):
        from scripts.train_lopo import _default_output_root

        resolved = str(_default_output_root().resolve())
        assert SMOKE_RUN_DIR_NAME not in resolved.split("\\")[-1]
        assert "smoke" not in Path(resolved).name.lower()


class TestEpochsOverride:
    """--epochs is an explicit runtime override; the frozen config stays at 1."""

    def test_epochs_three_propagates_to_history_and_metrics(self, tmp_path, synthetic_root):
        summary = _run(tmp_path / "e3", synthetic_root, epochs=3)

        assert summary["epochs_per_fold"] == 3
        for subject in SUBJECTS:
            directory = tmp_path / "e3" / f"fold_{subject}"
            history = json.loads((directory / "history.json").read_text())
            metrics = json.loads((directory / "metrics.json").read_text())
            assert [record["epoch"] for record in history] == [1, 2, 3]
            assert metrics["epochs"] == 3

    def test_all_other_protocol_b_settings_unchanged_under_override(
        self, tmp_path, synthetic_root
    ):
        _run(tmp_path / "e3", synthetic_root, epochs=3)
        frozen = load_engineering_baseline().document["settings"]
        for subject in SUBJECTS:
            metrics = json.loads(
                (tmp_path / "e3" / f"fold_{subject}" / "metrics.json").read_text()
            )
            assert metrics["loss"] == frozen["loss"]["value"]
            assert metrics["optimizer"] == frozen["optimizer"]["type"]
            assert metrics["scheduler"] == frozen["scheduler"]["type"]
            assert metrics["learning_rate"] == frozen["optimizer"]["lr"]
            assert metrics["weight_decay"] == frozen["optimizer"]["weight_decay"]
            assert metrics["batch_size"] == frozen["batch_size"]["value"]
            assert metrics["seed"] == 0

    def test_default_without_flag_remains_one_epoch(self, tmp_path, synthetic_root):
        summary = _run(tmp_path / "default", synthetic_root)

        assert summary["epochs_per_fold"] == 1
        for subject in SUBJECTS:
            directory = tmp_path / "default" / f"fold_{subject}"
            history = json.loads((directory / "history.json").read_text())
            metrics = json.loads((directory / "metrics.json").read_text())
            assert len(history) == 1
            assert metrics["epochs"] == 1

    def test_fold_isolation_preserved_under_epochs_override(self, tmp_path, synthetic_root):
        _run(tmp_path / "e3", synthetic_root, epochs=3)
        root = tmp_path / "e3"
        dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
        assert dirs == ["fold_p00", "fold_p01", "fold_p02"]
        for name in dirs:
            files = {p.name for p in (root / name).iterdir()}
            assert {"checkpoint.pt", "history.json", "metrics.json"} <= files

    def test_skip_if_complete_unchanged_under_epochs_override(self, tmp_path, synthetic_root):
        _run(tmp_path / "e3", synthetic_root, epochs=3)
        mtimes = {
            subject: (tmp_path / "e3" / f"fold_{subject}" / "checkpoint.pt").stat().st_mtime_ns
            for subject in SUBJECTS
        }

        summary = _run(tmp_path / "e3", synthetic_root, epochs=3)

        assert summary["counts"] == {"completed": 0, "skipped": 3, "failed": 0}
        for subject in SUBJECTS:
            after = (tmp_path / "e3" / f"fold_{subject}" / "checkpoint.pt").stat().st_mtime_ns
            assert after == mtimes[subject]

    def test_non_positive_epochs_rejected(self, tmp_path, synthetic_root):
        with pytest.raises(ValueError):
            run_lopo(
                output_root=tmp_path / "bad",
                seed=0,
                subjects=list(SUBJECTS),
                model_builder=TinyGaze,
                fold_datasets_factory=_synthetic_factory(synthetic_root),
                progress=lambda message: None,
                epochs=0,
            )


class TestModelNamePropagation:
    def test_model_name_recorded_in_summary_and_metrics(self, tmp_path, synthetic_root):
        summary = _run(
            tmp_path / "out", synthetic_root, model_name="resnet50_face_eyes"
        )

        assert summary["model"] == "resnet50_face_eyes"
        for subject in SUBJECTS:
            metrics = json.loads(
                (tmp_path / "out" / f"fold_{subject}" / "metrics.json").read_text()
            )
            assert metrics["model"] == "resnet50_face_eyes"

    def test_default_model_is_frozen_baseline_resnet50(self, tmp_path, synthetic_root):
        summary = _run(tmp_path / "out", synthetic_root)
        assert summary["model"] == "resnet50"
