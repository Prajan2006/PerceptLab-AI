"""Unit tests for the baseline training pipeline (synthetic fixtures, CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from data.mpiifacegaze import (
    RawMPIIFaceGazeDataset,
    build_lopo_fold_datasets,
    build_synthetic_raw_subject,
    make_gaze_dataloader,
)
from data.splits import lopo_folds
from models.resnet50 import ResNet50Gaze
from training import BaselineTrainer, TrainingConfig, apply_seed

SUBJECTS = ("p00", "p01")


@pytest.fixture(scope="module")
def fold_loaders(tmp_path_factory):
    """One LOPO fold over a tiny synthetic corpus; loaders from the real API."""
    root = tmp_path_factory.mktemp("train_data") / "Data"
    for subject in SUBJECTS:
        build_synthetic_raw_subject(
            root,
            subject,
            sessions=("day01", "day02"),
            frames_per_session=1,
            include_artifacts=False,
        )
    fold = lopo_folds(list(SUBJECTS))[0]
    train_ds, val_ds = build_lopo_fold_datasets(fold, root=root)
    train_loader = make_gaze_dataloader(train_ds, batch_size=2)
    val_loader = make_gaze_dataloader(val_ds, batch_size=2)
    return fold, train_ds, val_ds, train_loader, val_loader


@pytest.fixture()
def trainer(fold_loaders):
    _, _, _, train_loader, val_loader = fold_loaders
    return BaselineTrainer(
        ResNet50Gaze(pretrained_backbone=False),
        train_loader,
        val_loader,
        config=TrainingConfig(seed=0),
    )


class TestConstruction:
    def test_components_wired(self, trainer):
        assert isinstance(trainer.optimizer, Adam)
        assert isinstance(trainer.loss_fn, MSELoss)
        assert trainer.device.type == "cpu"
        assert list(trainer.optimizer.param_groups[0]["params"]) == list(
            trainer.model.parameters()
        )

    def test_default_config(self, trainer):
        assert trainer.config.epochs == 1
        assert trainer.config.learning_rate == 1e-3
        assert trainer.config.weight_decay == 0.0

    def test_seed_makes_rng_reproducible(self):
        apply_seed(123)
        first = torch.rand(4)
        apply_seed(123)
        second = torch.rand(4)
        assert torch.equal(first, second)


class TestTrainingStep:
    def test_one_epoch_returns_finite_loss(self, trainer):
        mean_loss = trainer.train_epoch()
        assert math.isfinite(mean_loss)
        assert mean_loss > 0.0

    def test_parameters_actually_update(self, trainer):
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.train_epoch()
        for p_before, p_after in zip(before, trainer.model.parameters()):
            assert not torch.equal(p_before, p_after)

    def test_fit_history_records_losses(self, trainer):
        records = trainer.fit()
        assert len(records) == 1
        assert records[0]["epoch"] == 1
        assert math.isfinite(records[0]["train_loss"])
        assert records[0]["val_loss"] is None or math.isfinite(records[0]["val_loss"])

    def test_fit_with_validation_reports_val_loss(self, trainer):
        history = trainer.fit()
        assert history[0]["val_loss"] is not None
        assert math.isfinite(history[0]["val_loss"])

    def test_consumes_real_gazebatch_loaders(self, fold_loaders):
        _, _, _, train_loader, _ = fold_loaders
        batch = next(iter(train_loader))
        assert batch.gaze.dtype == torch.float64
        assert batch.face.shape[1:] == (3, 224, 224)


class TestValidationSemantics:
    def test_validation_never_updates_parameters(self, trainer):
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.validate()
        for p_before, p_after in zip(before, trainer.model.parameters()):
            assert torch.equal(p_before, p_after)

    def test_validation_is_repeatable(self, trainer):
        first = trainer.validate()
        second = trainer.validate()
        assert first == second

    def test_training_mode_restored_after_validation(self, trainer):
        trainer.model.train()
        trainer.validate()
        assert trainer.model.training is True

    def test_validate_without_loader_raises(self, fold_loaders):
        _, _, _, train_loader, _ = fold_loaders
        bare = BaselineTrainer(ResNet50Gaze(), train_loader)
        with pytest.raises(ValueError):
            bare.validate()


class TestDeterminism:
    def test_same_seed_same_history(self, fold_loaders):
        _, _, _, train_loader, val_loader = fold_loaders

        def run():
            # Seed BEFORE model construction so parameter init is reproducible;
            # the trainer then re-applies the seed for everything afterwards.
            apply_seed(7)
            trainer = BaselineTrainer(
                ResNet50Gaze(pretrained_backbone=False),
                train_loader,
                val_loader,
                config=TrainingConfig(seed=7),
            )
            return trainer.fit()

        first, second = run(), run()
        assert first == second


class TestCheckpointing:
    def test_roundtrip_restores_exact_state(self, trainer, fold_loaders, tmp_path):
        trainer.fit()
        path = trainer.save_checkpoint(tmp_path / "ckpt" / "baseline.pt")

        _, _, _, train_loader, val_loader = fold_loaders
        restored = BaselineTrainer(
            ResNet50Gaze(pretrained_backbone=False),
            train_loader,
            val_loader,
        )
        payload = restored.load_checkpoint(path)

        assert restored.epoch == trainer.epoch
        assert len(restored.history) == len(trainer.history)
        for p_saved, p_restored in zip(trainer.model.parameters(), restored.model.parameters()):
            assert torch.equal(p_saved.detach().cpu(), p_restored)

    def test_loaded_model_predicts_identically(self, trainer, fold_loaders, tmp_path):
        trainer.fit()
        path = trainer.save_checkpoint(tmp_path / "baseline.pt")

        face = torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(5))
        expected = trainer.model.eval()(face).detach()

        _, _, _, train_loader, _ = fold_loaders
        fresh_trainer = BaselineTrainer(ResNet50Gaze(), train_loader)
        fresh_trainer.load_checkpoint(path)
        actual = fresh_trainer.model.eval()(face).detach()

        assert torch.allclose(expected, actual, atol=0.0, rtol=0.0)


class TestCpuAndLopoIntegrity:
    def test_runs_on_cpu_device_config(self, fold_loaders):
        _, _, _, train_loader, val_loader = fold_loaders
        cpu_trainer = BaselineTrainer(
            ResNet50Gaze(),
            train_loader,
            val_loader,
            config=TrainingConfig(device="cpu", seed=1),
        )
        losses = cpu_trainer.fit()
        assert all(math.isfinite(r["train_loss"]) for r in losses)

    def test_lopo_split_untouched_by_training(self, fold_loaders):
        fold, train_ds, val_ds, _, _ = fold_loaders
        train_ids_before = set(train_ds.sample_ids)
        val_ids_before = set(val_ds.sample_ids)
        folds_before = [(f.fold_index, f.test_subjects, f.train_subjects) for f in lopo_folds(list(SUBJECTS))]

        trainer = BaselineTrainer(
            ResNet50Gaze(pretrained_backbone=False),
            make_gaze_dataloader(train_ds, batch_size=2),
            make_gaze_dataloader(val_ds, batch_size=2),
            config=TrainingConfig(seed=3),
        )
        trainer.fit()

        assert set(train_ds.sample_ids) == train_ids_before
        assert set(val_ds.sample_ids) == val_ids_before
        assert train_ids_before.isdisjoint(val_ids_before)
        folds_after = [(f.fold_index, f.test_subjects, f.train_subjects) for f in lopo_folds(list(SUBJECTS))]
        assert folds_after == folds_before


class TestAuditGaps:
    """Behaviors identified as untested during the baseline-reproduction audit."""

    def test_multi_epoch_history_accumulates(self, fold_loaders):
        _, _, _, train_loader, _ = fold_loaders
        trainer = BaselineTrainer(
            ResNet50Gaze(pretrained_backbone=False),
            train_loader,
            config=TrainingConfig(epochs=3, seed=5),
        )
        history = trainer.fit()
        assert [record["epoch"] for record in history] == [1, 2, 3]
        assert trainer.epoch == 3
        assert trainer.history == history

    def test_resume_from_checkpoint_continues_epoch_numbering(
        self, trainer, fold_loaders, tmp_path
    ):
        trainer.fit()
        path = trainer.save_checkpoint(tmp_path / "mid.pt")

        _, _, _, train_loader, _ = fold_loaders
        resumed = BaselineTrainer(ResNet50Gaze(pretrained_backbone=False), train_loader)
        payload = resumed.load_checkpoint(path)
        assert payload["epoch"] == 1

        continuation = resumed.fit()
        assert [record["epoch"] for record in continuation] == [2]
        assert resumed.epoch == 2

    def test_optimizer_state_survives_roundtrip(self, trainer, fold_loaders, tmp_path):
        trainer.train_epoch()
        path = trainer.save_checkpoint(tmp_path / "optimizer.pt")

        _, _, _, train_loader, _ = fold_loaders
        restored = BaselineTrainer(ResNet50Gaze(pretrained_backbone=False), train_loader)
        restored.load_checkpoint(path)

        original_state = trainer.optimizer.state_dict()["state"]
        restored_state = restored.optimizer.state_dict()["state"]
        assert set(original_state) == set(restored_state)
        for param_id, entry in original_state.items():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value.cpu(), restored_state[param_id][key].cpu())
                else:
                    assert value == restored_state[param_id][key]

    def test_empty_train_loader_raises_clear_error(self):
        from torch.utils.data import DataLoader

        from data.mpiifacegaze import collate_gaze_samples

        class _Empty(torch.utils.data.Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError

        empty_loader = DataLoader(_Empty(), batch_size=4, collate_fn=collate_gaze_samples)
        trainer = BaselineTrainer(ResNet50Gaze(pretrained_backbone=False), empty_loader)
        with pytest.raises(ValueError):
            trainer.train_epoch()


def test_apply_seed_covers_numpy_and_python():
    apply_seed(9)
    np_first = np.random.rand(3)
    apply_seed(9)
    np_second = np.random.rand(3)
    assert np.array_equal(np_first, np_second)
