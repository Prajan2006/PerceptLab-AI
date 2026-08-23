"""Configuration-freeze validation for Protocol B (Project Engineering Baseline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    BaselineConfigError,
    TrainingConfig,
    load_engineering_baseline,
)

REQUIRED_SETTINGS = (
    "loss",
    "optimizer",
    "scheduler",
    "epochs",
    "batch_size",
    "seed",
    "pretrained_initialization",
    "checkpoint",
    "augmentation",
    "shuffling",
)


@pytest.fixture()
def baseline():
    return load_engineering_baseline()


class TestIdentity:
    def test_identifies_as_engineering_baseline(self, baseline):
        identity = baseline.document["identity"]
        assert identity["protocol_name"] == "Protocol B - Engineering Baseline"
        assert identity["terminology"] == "Project Engineering Baseline"

    def test_never_claims_official_reproduction(self, baseline):
        assert baseline.document["identity"]["is_official_reproduction"] is False
        assert baseline.document["identity"]["reproduction_target"] is None


class TestProtocolBValues:
    def test_all_required_settings_present(self, baseline):
        for key in REQUIRED_SETTINGS:
            assert key in baseline.document["settings"], key

    def test_loss_is_mse_on_raw_prediction_and_unit_label(self, baseline):
        loss = baseline.document["settings"]["loss"]
        assert loss["value"] == "MSELoss(raw_prediction, unit_gaze_label)"
        assert "raw/unnormalized" in loss["target_handling"]

    def test_optimizer_matches_protocol_b_exactly(self, baseline):
        optimizer = baseline.document["settings"]["optimizer"]
        assert optimizer["type"] == "Adam"
        assert optimizer["lr"] == 1e-3
        assert optimizer["weight_decay"] == 0.0
        assert optimizer["betas"] == [0.9, 0.999]
        assert optimizer["eps"] == 1e-8
        assert optimizer["amsgrad"] is False

    def test_scheduler_explicitly_none(self, baseline):
        assert baseline.document["settings"]["scheduler"]["type"] == "none"

    def test_epochs_batch_size_match_protocol_b(self, baseline):
        settings = baseline.document["settings"]
        assert settings["epochs"]["value"] == 1
        assert settings["batch_size"]["value"] == 8

    def test_augmentation_none_and_shuffling_preserved(self, baseline):
        settings = baseline.document["settings"]
        assert settings["augmentation"]["value"] == "none"
        assert settings["shuffling"]["dataset_level"] is False
        assert "shuffle=False" in settings["shuffling"]["loader_behavior"]

    def test_pretrained_request_true_with_split_provenance(self, baseline):
        pretrained = baseline.document["settings"]["pretrained_initialization"]
        assert pretrained["registry_request"] is True
        assert pretrained["status"] == "SPLIT"

    def test_checkpoint_has_no_best_selection(self, baseline):
        checkpoint = baseline.document["settings"]["checkpoint"]
        assert checkpoint["best_selection"] == "none implemented"


class TestProvenanceLabels:
    def test_every_setting_carries_a_documented_status(self, baseline):
        allowed = set(baseline.document["allowed_status_labels"]) | {"SPLIT"}
        entries = {
            **baseline.document["settings"],
            **baseline.document["verified_context"],
        }
        for key, entry in entries.items():
            assert entry.get("status") in allowed, f"{key}: {entry.get('status')!r}"

    def test_expected_classification_partition(self, baseline):
        statuses = {
            key: entry["status"] for key, entry in baseline.document["settings"].items()
        }
        project_chosen = {k for k, s in statuses.items() if s.startswith("PROJECT-CHOSEN")}
        verified = {k for k, s in statuses.items() if s.startswith("VERIFIED")}
        splits = {k for k, s in statuses.items() if s == "SPLIT"}

        assert project_chosen == {
            "loss", "optimizer", "scheduler", "epochs", "batch_size",
            "checkpoint", "augmentation",
        }
        assert verified == {"seed", "shuffling"}
        assert splits == {"pretrained_initialization"}

    def test_verified_context_entries_are_source_backed(self, baseline):
        for key, entry in baseline.document["verified_context"].items():
            assert entry["status"] == "VERIFIED / SOURCE-BACKED", key


class TestNoHiddenSettings:
    def test_optimizer_enumerates_all_torch_defaults(self, baseline):
        optimizer_keys = set(baseline.document["settings"]["optimizer"])
        assert {"type", "lr", "weight_decay", "betas", "eps", "amsgrad"} <= optimizer_keys

    def test_loader_rejects_missing_optimizer_field(self, tmp_path, baseline):
        document = json.loads(json.dumps(baseline.document))
        del document["settings"]["optimizer"]["betas"]
        path = Path(tmp_path) / "broken.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BaselineConfigError):
            load_engineering_baseline(path)

    def test_loader_rejects_undocumented_setting(self, tmp_path, baseline):
        document = json.loads(json.dumps(baseline.document))
        document["settings"]["momentum"] = {"value": 0.9}
        path = Path(tmp_path) / "undocumented.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BaselineConfigError):
            load_engineering_baseline(path)

    def test_loader_rejects_unrecognized_status_label(self, tmp_path, baseline):
        document = json.loads(json.dumps(baseline.document))
        document["settings"]["loss"]["status"] = "TRUSTED BECAUSE COMMON"
        path = Path(tmp_path) / "badlabel.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BaselineConfigError):
            load_engineering_baseline(path)

    def test_loader_rejects_official_reproduction_claim(self, tmp_path, baseline):
        document = json.loads(json.dumps(baseline.document))
        document["identity"]["is_official_reproduction"] = True
        path = Path(tmp_path) / "claimant.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BaselineConfigError):
            load_engineering_baseline(path)


class TestConsistencyWithTrainerDefaults:
    def test_frozen_values_equal_trainer_defaults(self, baseline):
        assert baseline.training_config == TrainingConfig()

    def test_batch_size_accessor(self, baseline):
        assert baseline.batch_size == 8
