"""Tests for the controlled eye-region input-representation experiment arm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from data.mpiifacegaze import (
    RawMPIIFaceGazeDataset,
    build_synthetic_raw_subject,
    make_gaze_dataloader,
)
from models.registry import (
    MODEL_REGISTRY,
    ModelBuildError,
    build_model,
    get_model_spec,
)
from models.resnet50_face_eyes import EXPECTED_EYE_SHAPE, EXPECTED_FACE_SHAPE, ResNet50FaceEyes

EXPERIMENT_CONFIG = Path("config/experiments/eye_region_E3.json")
RESEARCH_QUESTION = (
    "Under fixed preprocessing, architecture, training settings, and subject-independent "
    "LOPO evaluation, does adding localized eye-region information to a full-face RGB "
    "gaze-estimation baseline improve mean angular error on MPIIFaceGaze?"
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return ResNet50FaceEyes(pretrained_backbone=False)


@pytest.fixture(scope="module")
def real_batch(tmp_path_factory):
    root = build_synthetic_raw_subject(
        tmp_path_factory.mktemp("eye_data") / "Data",
        "p00",
        sessions=("day01",),
        frames_per_session=4,
        include_artifacts=False,
    )
    dataset = RawMPIIFaceGazeDataset(root=root.parent)
    return next(iter(make_gaze_dataloader(dataset, batch_size=2)))


class TestConstruction:
    def test_registered_and_buildable(self):
        model = build_model("resnet50_face_eyes", pretrained_backbone=False)
        assert isinstance(model, ResNet50FaceEyes)
        assert isinstance(model, nn.Module)

    def test_spec_contract_matches_design(self):
        spec = get_model_spec("resnet50_face_eyes")
        assert spec.inputs == {
            "face": (3, 224, 224),
            "left_eye": (3, 36, 60),
            "right_eye": (3, 36, 60),
        }
        assert spec.outputs == ("gaze",)

    def test_gazetr_hybrid_still_unimplemented(self):
        with pytest.raises(ModelBuildError):
            build_model("gazetr_hybrid")

    def test_baseline_resnet50_contract_unchanged(self):
        spec = get_model_spec("resnet50")
        assert spec.inputs == {"face": (3, 224, 224)}
        baseline = build_model("resnet50", pretrained_backbone=False)
        with pytest.raises(TypeError):
            baseline(torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 36, 60))


class TestForward:
    def test_output_shape_dtype_finite(self, model, real_batch):
        model.eval()
        with torch.no_grad():
            out = model(real_batch.face, real_batch.left_eye, real_batch.right_eye)
        assert out.shape == (real_batch.batch_size, 3)
        assert out.dtype == torch.float32
        assert bool(torch.isfinite(out).all())

    def test_random_weights_reproducible(self):
        torch.manual_seed(11)
        first = ResNet50FaceEyes()
        torch.manual_seed(11)
        second = ResNet50FaceEyes()
        for p1, p2 in zip(first.parameters(), second.parameters()):
            assert torch.equal(p1, p2)


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"left_eye": torch.zeros(1, 3, 30, 30)},
            {"right_eye": torch.zeros(1, 1, 36, 60)},
            {"face": torch.zeros(1, 3, 64, 64)},
        ],
    )
    def test_wrong_shapes_rejected(self, model, kwargs):
        inputs = {
            "face": torch.zeros(1, *EXPECTED_FACE_SHAPE),
            "left_eye": torch.zeros(1, *EXPECTED_EYE_SHAPE),
            "right_eye": torch.zeros(1, *EXPECTED_EYE_SHAPE),
        }
        inputs.update(kwargs)
        with pytest.raises(ValueError):
            model(**inputs)

    def test_error_message_names_expected_eye_shape(self, model):
        with pytest.raises(ValueError) as excinfo:
            model(torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 10, 10), torch.zeros(1, 3, 36, 60))
        assert "3, 36, 60" in str(excinfo.value)

    def test_device_consistency(self, model):
        model.to("cpu")
        assert {p.device.type for p in model.parameters()} == {"cpu"}


class TestLabelsAndConventionsUnchanged:
    def test_gaze_label_is_normalize_gt_minus_fc(self, tmp_path):
        root = build_synthetic_raw_subject(
            tmp_path / "Data", "p00", sessions=("day01",), frames_per_session=2,
            include_artifacts=False,
        )
        dataset = RawMPIIFaceGazeDataset(root=root.parent)
        sample = dataset[0]
        annotation = dataset.annotations[0]

        vector = np.asarray(annotation.gaze_target, dtype=np.float64) - np.asarray(
            annotation.face_center, dtype=np.float64
        )
        expected = vector / np.linalg.norm(vector)
        assert np.allclose(sample.gaze, expected, atol=1e-12)
        assert abs(np.linalg.norm(sample.gaze) - 1.0) < 1e-9
        # eye-region crops keep the validated representation
        assert sample.left_eye.shape == (3, 36, 60)
        assert sample.right_eye.shape == (3, 36, 60)
        assert np.isfinite(sample.left_eye).all() and np.isfinite(sample.right_eye).all()


class TestExperimentConfiguration:
    def test_config_identifies_experiment_unambiguously(self):
        payload = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        identity = payload["experiment"]
        assert identity == "eye_region_vs_full_face_E3"
        assert payload["identity"]["research_question"] == RESEARCH_QUESTION
        assert "eye-region" in payload["identity"]["changed_variable"]
        assert payload["identity"]["output_directory"] == (
            "data/experiments/engineering_baseline_v1_E3_eye_region"
        )

    def test_reference_and_frozen_dirs_declared(self):
        payload = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        identity = payload["identity"]
        assert identity["comparison_reference"] == "data/experiments/engineering_baseline_v1_E3"
        assert "data/experiments/engineering_baseline_v1" in identity["frozen_baseline_do_not_modify"]
        assert "data/experiments/engineering_baseline_v1_E3" in identity["frozen_baseline_do_not_modify"]

    def test_fixed_variables_pin_the_recipe(self):
        fixed = "\n".join(json.loads(EXPERIMENT_CONFIG.read_text())["identity"]["fixed_variables"])
        for token in ("seed: 0", "batch_size: 8", "learning_rate: 1e-3", "weight_decay: 0.0",
                      "scheduler: none", "MSELoss", "3 epochs"):
            assert token in fixed, token

    def test_future_command_present_but_marked_not_executed(self):
        payload = json.loads(EXPERIMENT_CONFIG.read_text())
        command = payload["identity"]["future_command_not_executed"]
        assert "--model resnet50_face_eyes" in command
        assert "--epochs 3" in command
        assert "--seed 0" in command
        assert "--device cuda" in command
        assert "engineering_baseline_v1_E3_eye_region" in command

    def test_settings_provenance_labels_valid(self):
        payload = json.loads(EXPERIMENT_CONFIG.read_text())
        allowed = set(payload["allowed_status_labels"]) | {"SPLIT"}
        for key, entry in payload["settings"].items():
            assert entry.get("status") in allowed, key


class TestNoSplitLeakage:
    def test_training_keeps_fold_disjointness(self, tmp_path):
        root = build_synthetic_raw_subject(
            tmp_path / "Data", "p00", sessions=("day01",), frames_per_session=2,
            include_artifacts=False,
        )
        root2 = build_synthetic_raw_subject(
            tmp_path / "Data", "p01", sessions=("day01",), frames_per_session=2,
            include_artifacts=False,
        )
        del root2
        dataset = RawMPIIFaceGazeDataset(root=tmp_path / "Data")

        from data.splits import lopo_folds

        fold = lopo_folds(list(dataset.subjects))[0]
        test_ids = {
            f"{a.subject_id}:{a.source_line}"
            for a in dataset.annotations
            if a.subject_id in set(fold.test_subjects)
        }
        train_ids = {
            f"{a.subject_id}:{a.source_line}"
            for a in dataset.annotations
            if a.subject_id in set(fold.train_subjects)
        }
        assert len(dataset) == 4
        assert test_ids.isdisjoint(train_ids)
        assert MODEL_REGISTRY["resnet50_face_eyes"].inputs["face"] == (3, 224, 224)
