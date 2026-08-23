"""Unit tests for the ResNet-50 gaze architecture (synthetic inputs only)."""

from __future__ import annotations

import pytest
import torch
from torch.nn import Module, Linear

from data.mpiifacegaze import (
    RawMPIIFaceGazeDataset,
    build_synthetic_raw_subject,
    make_gaze_dataloader,
)
from models.registry import ModelBuildError, build_model, get_model_spec
from models.resnet50 import EXPECTED_FACE_SHAPE, ResNet50Gaze


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return ResNet50Gaze(pretrained_backbone=False)


def face_batch(batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    return torch.rand((batch_size, *EXPECTED_FACE_SHAPE), generator=generator)


class TestConstruction:
    def test_build_via_registry(self):
        model = build_model("resnet50", pretrained_backbone=False)
        assert isinstance(model, ResNet50Gaze)
        assert isinstance(model, Module)

    def test_head_maps_2048_to_gaze3(self, model):
        head = model.backbone.fc
        assert isinstance(head, Linear)
        assert head.in_features == 2048
        assert head.out_features == 3

    def test_spec_contract_matches_module(self, model):
        spec = get_model_spec("resnet50")
        assert model.backbone.conv1.weight.shape[1] == spec.inputs["face"][0]
        assert tuple(EXPECTED_FACE_SHAPE) == spec.inputs["face"]
        assert "gaze" in spec.outputs

    def test_pretrained_failure_is_actionable(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("no network in unit tests")

        import torchvision.models as tvm

        monkeypatch.setattr(tvm, "resnet50", boom)
        with pytest.raises(ModelBuildError):
            ResNet50Gaze(pretrained_backbone=True)

    def test_seed_identical_init(self):
        torch.manual_seed(123)
        first = ResNet50Gaze()
        torch.manual_seed(123)
        second = ResNet50Gaze()
        for p1, p2 in zip(first.parameters(), second.parameters()):
            assert torch.equal(p1, p2)


class TestForward:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_output_shape_and_dtype(self, model, batch_size):
        model.eval()
        with torch.no_grad():
            out = model(face_batch(batch_size))
        assert out.shape == (batch_size, 3)
        assert out.dtype == torch.float32

    def test_outputs_finite(self, model):
        model.eval()
        with torch.no_grad():
            out = model(face_batch(2))
        assert bool(torch.isfinite(out).all())

    def test_accepts_real_dataloader_face_tensor(self, tmp_path, model):
        root = build_synthetic_raw_subject(
            tmp_path / "Data", "p00", sessions=("day01",), frames_per_session=4,
            include_artifacts=False,
        )
        dataset = RawMPIIFaceGazeDataset(root=root.parent)
        batch = next(iter(make_gaze_dataloader(dataset, batch_size=2)))
        model.eval()
        with torch.no_grad():
            out = model(batch.face)
        assert out.shape == (2, 3)
        assert bool(torch.isfinite(out).all())


class TestDeterminism:
    def test_eval_mode_forward_bitwise_reproducible(self, model):
        model.eval()
        face = face_batch(3)
        with torch.no_grad():
            first = model(face)
            second = model(face)
        assert torch.equal(first, second)

    def test_eval_mode_survives_device_roundtrip_cpu(self, model):
        face = face_batch(2)
        model.eval()
        with torch.no_grad():
            direct = model(face)
        moved = model.to("cpu")
        assert moved is model
        with torch.no_grad():
            after_move = moved(face)
        assert torch.equal(direct, after_move)


class TestInvalidInput:
    def test_wrong_channel_count_rejected(self, model):
        with pytest.raises(ValueError):
            model(torch.zeros((1, 1, 224, 224)))

    def test_wrong_spatial_size_rejected(self, model):
        with pytest.raises(ValueError):
            model(torch.zeros((1, 3, 64, 64)))

    def test_missing_batch_dimension_rejected(self, model):
        with pytest.raises(ValueError):
            model(torch.zeros(EXPECTED_FACE_SHAPE))

    def test_non_tensor_rejected(self, model):
        with pytest.raises(ValueError):
            model("not a tensor")

    def test_error_message_states_expected_shape(self, model):
        with pytest.raises(ValueError) as excinfo:
            model(torch.zeros((1, 3, 100, 100)))
        assert "3, 224, 224" in str(excinfo.value)


class TestParameterDeviceConsistency:
    def test_all_parameters_share_device(self, model):
        devices = {p.device for p in model.parameters()}
        assert len(devices) == 1
        assert next(iter(devices)).type == "cpu"

    def test_parameters_require_grad(self, model):
        parameters = list(model.parameters())
        assert parameters
        assert all(p.requires_grad for p in parameters)

    def test_to_device_moves_every_parameter(self, model):
        model.to("cpu")
        assert {p.device for p in model.parameters()} == {torch.device("cpu")}
