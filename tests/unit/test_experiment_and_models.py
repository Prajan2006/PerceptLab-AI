"""Unit tests for model registry and experiment configuration schema."""

import json

import pytest

from experiments.config import (
    ExperimentConfigError,
    load_experiment_config,
    parse_experiment_config,
)
from models.registry import MODEL_REGISTRY, ModelBuildError, build_model, get_model_spec, list_models


class TestModelRegistry:
    def test_locked_models_registered(self):
        assert {"resnet50", "gazetr_hybrid"} <= set(MODEL_REGISTRY)

    def test_resnet50_input_contract(self):
        spec = get_model_spec("resnet50")
        assert spec.inputs["face"] == (3, 224, 224)
        assert "gaze" in spec.outputs

    def test_build_unimplemented_model_raises(self):
        # gazetr_hybrid has no implemented architecture yet; resnet50 does.
        with pytest.raises(ModelBuildError):
            build_model("gazetr_hybrid")

    def test_build_resnet50_returns_module(self):
        model = build_model("resnet50", pretrained_backbone=False)
        import torch.nn as nn

        assert isinstance(model, nn.Module)

    def test_unknown_model_rejected(self):
        with pytest.raises(KeyError):
            get_model_spec("vgg16")

    def test_list_models_reports_implementation_state(self):
        entries = {entry["name"]: entry for entry in list_models()}
        assert entries["resnet50"]["implemented"] is True
        assert entries["gazetr_hybrid"]["implemented"] is False


def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "test-run",
        "seed": 7,
        "dataset": {"name": "mpii_facegaze"},
        "model": {"name": "resnet50", "params": {}},
        "preprocessing": {"recipe": "gazehub", "params": {"face_size": 224}},
        "protocol": {"type": "lopo"},
        "evaluation": {"metric": "mean_angular_error_deg"},
        "output_dir": "data/experiments/test-run",
    }


class TestExperimentConfig:
    def test_example_config_loads(self):
        config = load_experiment_config("config/experiments/resnet50_mpiifacegaze_lopo.example.json")
        assert config.name == "resnet50-mpii-lopo-baseline"
        assert config.model.name == "resnet50"
        assert config.dataset.name == "mpii_facegaze"
        assert config.protocol.type == "lopo"

    def test_roundtrip_to_dict(self):
        config = parse_experiment_config(valid_payload())
        restored = parse_experiment_config(config.to_dict())
        assert restored == config

    @pytest.mark.parametrize(
        "mutation,expected_fragment",
        [
            (lambda p: p.update(name="Bad Name!"), "name"),
            (lambda p: p["dataset"].update(name="unknown_dataset"), "dataset.name"),
            (lambda p: p["model"].update(name="vgg16"), "model.name"),
            (lambda p: p["preprocessing"].update(recipe="custom"), "preprocessing recipe"),
            (lambda p: p["protocol"].update(type="kfold"), "protocol.type"),
            (lambda p: p["evaluation"].update(metric="accuracy"), "evaluation.metric"),
            (lambda p: p.update(seed=-3), "seed"),
        ],
    )
    def test_invalid_fields_rejected(self, mutation, expected_fragment):
        payload = valid_payload()
        mutation(payload)
        with pytest.raises(ExperimentConfigError) as excinfo:
            parse_experiment_config(payload)
        assert expected_fragment in str(excinfo.value)

    def test_multiple_errors_reported_together(self):
        payload = valid_payload()
        payload["name"] = "!!"
        payload["model"]["name"] = "nope"
        with pytest.raises(ExperimentConfigError) as excinfo:
            parse_experiment_config(payload)
        assert len(excinfo.value.errors) >= 2

    def test_output_dir_defaults(self):
        payload = valid_payload()
        payload.pop("output_dir")
        config = parse_experiment_config(payload)
        assert config.output_dir == "data/experiments"
