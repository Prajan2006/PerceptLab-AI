"""Baseline training pipeline for the ResNet-50 gaze model.

Minimal, deliberate scope: an epoch loop over the EXISTING
``RawMPIIFaceGazeDataset``/``make_gaze_dataloader`` batches, per-epoch
train/validation loss reporting, optional seeding, and small checkpoint
helpers. No augmentation, no sweeps, no orchestration, no metrics beyond
validation loss.

Choices and their grounding (the repository specifies no training
hyperparameters anywhere):

- **Loss** — ``nn.MSELoss`` between the model's raw ``(B, 3)`` prediction and
  the gaze label exactly as produced by the validated pipeline (already a
  unit-length vector). MSE-to-label regression is the objective used by the
  appearance-based ResNet-50 baselines this project's GazeHub-style recipe
  mirrors, and it preserves the raw-output contract: the locked angular-error
  evaluator normalizes predictions itself, so no normalization layer is added
  here. The target representation is NOT reformulated.
- **Optimizer** — ``torch.optim.Adam`` (default lr ``1e-3``, weight decay
  ``0.0``): a self-adapting baseline that needs no LR schedule.
- **Scheduler** — none, because nothing in the repository requires one.
- **Seeding** — applied when ``TrainingConfig.seed`` is set (the experiment
  schema already carries a seed): Python/NumPy/Torch RNGs are seeded at
  trainer construction, covering everything from that point on (e.g. loader
  shuffling). Parameter initialization happens when the caller builds the
  model, so fully reproducible runs must seed *before* constructing it —
  ``apply_seed(config.seed)`` then ``Model(...)`` — as the tests demonstrate.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.mpiifacegaze import GazeBatch

# Canonical order of the input-image tensor fields a model may consume from a
# ``GazeBatch``. Labels/metadata (gaze, head_pose, identity fields) are never
# model inputs by project convention.
_MODEL_INPUT_FIELDS = ("face", "left_eye", "right_eye")


def model_input_names(model: nn.Module) -> tuple[str, ...]:
    """Input-tensor fields (in canonical order) that ``model.forward`` consumes.

    Single source of truth for routing ``GazeBatch`` tensors into any
    registered model: face-only models get exactly ``("face",)``, the
    face+eyes arm gets ``("face", "left_eye", "right_eye")``. Derived from the
    model's own forward signature, so it stays correct for any builder,
    including test stand-ins, without hard-coded special cases.
    """
    parameters = inspect.signature(model.forward).parameters
    names = tuple(name for name in _MODEL_INPUT_FIELDS if name in parameters)
    if not names:
        raise ValueError(
            f"{type(model).__name__}.forward consumes none of the supported "
            f"model inputs {list(_MODEL_INPUT_FIELDS)}; cannot route a GazeBatch."
        )
    return names


@dataclass(frozen=True)
class TrainingConfig:
    """Minimal baseline knobs. Defaults are documented in the module docstring."""

    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int | None = None


def apply_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches (python / numpy / torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class BaselineTrainer:
    """Epoch-level trainer over existing GazeBatch loaders.

    Model inputs are routed per ``model_input_names(model)``: face-only
    models receive ``batch.face`` exactly as before, multi-input models (the
    eye-region arm) additionally receive the existing preprocessed eye
    patches. Loss/optimizer/labels are unaffected by routing.
    """

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None = None
    config: TrainingConfig = field(default_factory=TrainingConfig)
    input_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            apply_seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.input_names = self.input_names or model_input_names(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self.epoch = 0
        self.history: list[dict] = []

    # ---- batch handling ---------------------------------------------------
    def _loss_for_batch(self, batch: GazeBatch) -> torch.Tensor:
        gaze = batch.gaze.to(device=self.device, dtype=torch.float32)
        prediction = self.model(
            **{name: getattr(batch, name).to(self.device) for name in self.input_names}
        )
        return self.loss_fn(prediction, gaze)

    # ---- phases ------------------------------------------------------------
    def train_epoch(self) -> float:
        """One optimization pass over ``train_loader``; returns mean batch loss."""
        self.model.train()
        total, batches = 0.0, 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._loss_for_batch(batch)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            batches += 1
        if batches == 0:
            raise ValueError("train_loader yielded no batches")
        return total / batches

    def validate(self) -> float:
        """Loss over ``val_loader`` under no_grad; parameters never update."""
        if self.val_loader is None:
            raise ValueError("no validation loader configured")
        was_training = self.model.training
        self.model.eval()
        try:
            total, batches = 0.0, 0
            with torch.no_grad():
                for batch in self.val_loader:
                    total += float(self._loss_for_batch(batch))
                    batches += 1
        finally:
            self.model.train(was_training)
        if batches == 0:
            raise ValueError("val_loader yielded no batches")
        return total / batches

    def fit(self) -> list[dict]:
        """Run ``config.epochs`` epochs; returns per-epoch loss history."""
        records: list[dict] = []
        for _ in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate() if self.val_loader is not None else None
            self.epoch += 1
            record = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.history.append(record)
            records.append(record)
        return records

    # ---- checkpointing -----------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
                "config": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "device": self.config.device,
                    "seed": self.config.seed,
                },
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path, map_location: str = "cpu") -> dict:
        payload = torch.load(Path(path), map_location=map_location)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.epoch = int(payload["epoch"])
        self.history = list(payload.get("history", []))
        self.model.to(self.device)
        return payload
