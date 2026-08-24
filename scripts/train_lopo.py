"""Thin LOPO orchestration for the Project Engineering Baseline (Protocol B).

Composes ONLY existing components — no new hyperparameters, no trainer/model/
data/split/evaluator changes:

    load_engineering_baseline -> lopo_folds -> build_lopo_fold_datasets
    -> make_gaze_dataloader(batch_size=8) -> build_model("resnet50")
    -> BaselineTrainer.fit() -> MeanAngularErrorEvaluator -> save_checkpoint

Behavior:
- deterministic fold order p00..p14 (existing lopo_folds ordering)
- per-fold isolated directory:  <output-root>/fold_pYY/
- final-only checkpointing via the existing trainer mechanism; history.json
  and metrics.json are written AFTER the checkpoint so their presence implies
  a fully completed fold
- skip-if-complete: a fold whose checkpoint.pt + history.json + metrics.json
  all exist is reported as "skipped" and never rewritten
- failures are recorded per fold and reported in the run summary; the run
  exits non-zero if any fold failed — a failed fold is never silent and never
  marked complete
- refuses to write into the smoke-run directory

Usage (after explicit approval to launch):
    python scripts/train_lopo.py --seed 0 --device cuda \
        --output data/experiments/engineering_baseline_v1

Optional explicit duration override (the frozen configuration itself is
never modified; the default remains the Protocol B value of 1 epoch/fold):
    python scripts/train_lopo.py --seed 0 --device cuda --epochs 3 \
        --output data/experiments/engineering_baseline_v1_E3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.evaluation.angular_error import MeanAngularErrorEvaluator  # noqa: E402
from data.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT  # noqa: E402
from data.mpiifacegaze import (  # noqa: E402
    build_lopo_fold_datasets,
    make_gaze_dataloader,
)
from data.splits import LOPOFold, lopo_folds  # noqa: E402
from models.registry import build_model  # noqa: E402
from training import BaselineTrainer, TrainingConfig, load_engineering_baseline  # noqa: E402

SMOKE_RUN_DIR_NAME = "smoke_protocol_b_fold_p07"
COMPLETION_FILES = ("checkpoint.pt", "history.json", "metrics.json")


@dataclass
class FoldResult:
    fold_index: int
    test_subject: str
    status: str  # "completed" | "skipped" | "failed"
    detail: str = ""


def _default_output_root() -> Path:
    return CONFIG_PROJECT_ROOT / "data" / "experiments" / "engineering_baseline_v1"


def fold_dir(output_root: Path, fold: LOPOFold) -> Path:
    return Path(output_root) / f"fold_{fold.test_subjects[0]}"


def is_fold_complete(output_root: Path, fold: LOPOFold) -> bool:
    directory = fold_dir(output_root, fold)
    return all((directory / name).exists() for name in COMPLETION_FILES)


def _evaluate(model, val_loader, device) -> dict:
    evaluator = MeanAngularErrorEvaluator()
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            prediction = model(batch.face.to(device))
            evaluator.add(
                prediction.detach().cpu().numpy(),
                batch.gaze.detach().cpu().numpy(),
                subject_ids=list(batch.subject_ids),
            )
    summary = evaluator.summary()
    return {
        "mean_angular_error_deg": summary.mean_deg,
        "num_eval_samples": summary.num_samples,
        "per_subject_deg": summary.per_subject_deg,
    }


def run_fold(
    fold: LOPOFold,
    *,
    root,
    output_root: Path,
    training_config: TrainingConfig,
    batch_size: int,
    seed: int,
    model_name: str = "resnet50",
    model_builder: Callable[[], object] | None = None,
    fold_datasets_factory=None,
    progress: Callable[[str], None] = print,
) -> FoldResult:
    from training.trainer import apply_seed

    if is_fold_complete(output_root, fold):
        progress(f"fold {fold.test_subjects[0]}: skipped (already complete)")
        return FoldResult(fold.fold_index, fold.test_subjects[0], "skipped")

    directory = fold_dir(output_root, fold)
    started = time.perf_counter()
    try:
        apply_seed(seed)
        factory = fold_datasets_factory or build_lopo_fold_datasets
        train_ds, val_ds = factory(fold, root=root)

        train_loader = make_gaze_dataloader(train_ds, batch_size=batch_size)
        val_loader = make_gaze_dataloader(val_ds, batch_size=batch_size)

        builder = model_builder or (lambda: build_model(model_name))
        model = builder()
        trainer = BaselineTrainer(model, train_loader, val_loader, config=training_config)

        history = trainer.fit()
        metrics = _evaluate(trainer.model, val_loader, trainer.device.type)
        metrics.update(
            {
                "fold_index": fold.fold_index,
                "test_subject": fold.test_subjects[0],
                "model": model_name,
                "train_samples": len(train_ds),
                "test_samples": len(val_ds),
                "epochs": training_config.epochs,
                "batch_size": batch_size,
                "learning_rate": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
                "optimizer": "Adam",
                "scheduler": "none",
                "loss": "MSELoss(raw_prediction, unit_gaze_label)",
                "seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

        directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(directory / "checkpoint.pt")
        (directory / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        progress(
            f"fold {fold.test_subjects[0]}: completed "
            f"(train_loss={history[-1]['train_loss']:.4f}, "
            f"val_loss={history[-1]['val_loss']:.4f}, "
            f"angular={metrics['mean_angular_error_deg']:.3f} deg)"
        )
        return FoldResult(fold.fold_index, fold.test_subjects[0], "completed")
    except Exception as exc:  # noqa: BLE001 - fold failure must be reported, not raised away
        progress(f"fold {fold.test_subjects[0]}: FAILED - {type(exc).__name__}: {exc}")
        return FoldResult(fold.fold_index, fold.test_subjects[0], "failed",
                          detail=f"{type(exc).__name__}: {exc}")


def run_lopo(
    *,
    output_root: str | Path | None = None,
    seed: int = 0,
    device: str = "cpu",
    dataset_name: str = "mpii_facegaze",
    subjects: list[str] | None = None,
    folds_filter: list[str] | None = None,
    model_builder: Callable[[], object] | None = None,
    fold_datasets_factory=None,
    baseline=None,
    progress: Callable[[str], None] = print,
    epochs: int | None = None,
    model_name: str = "resnet50",
) -> dict:
    output_root = Path(output_root) if output_root is not None else _default_output_root()
    resolved = output_root.resolve()
    if resolved.name == SMOKE_RUN_DIR_NAME or SMOKE_RUN_DIR_NAME in resolved.parts:
        raise ValueError("refusing to write into the smoke-run directory")

    baseline = baseline or load_engineering_baseline()
    # Epochs come from the frozen configuration unless explicitly overridden
    # at launch time (the frozen file itself is never modified).
    requested_epochs = baseline.training_config.epochs if epochs is None else int(epochs)
    if requested_epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {requested_epochs}")
    training_config = TrainingConfig(
        epochs=requested_epochs,
        learning_rate=baseline.training_config.learning_rate,
        weight_decay=baseline.training_config.weight_decay,
        device=device,
        seed=seed,
    )

    from data.config import resolve_dataset_root
    from data.mpiifacegaze.raw_adapter import discover_raw_subjects

    root = resolve_dataset_root(dataset_name).root
    discovered = subjects if subjects is not None else discover_raw_subjects(root)
    folds = lopo_folds(list(discovered))
    if folds_filter is not None:
        wanted = {f"{s}" for s in folds_filter}
        folds = [f for f in folds if f.test_subjects[0] in wanted]

    results: list[FoldResult] = []
    for fold in folds:
        results.append(
            run_fold(
                fold,
                root=root,
                output_root=output_root,
                training_config=training_config,
                batch_size=baseline.batch_size,
                seed=seed,
                model_name=model_name,
                model_builder=model_builder,
                fold_datasets_factory=fold_datasets_factory,
                progress=progress,
            )
        )

    completed = sum(r.status == "completed" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = [r for r in results if r.status == "failed"]
    summary = {
        "output_root": str(output_root),
        "protocol": "Protocol B - Engineering Baseline",
        "seed": seed,
        "epochs_per_fold": training_config.epochs,
        "batch_size": baseline.batch_size,
        "model": model_name,
        "folds": [
            {
                "fold_index": r.fold_index,
                "test_subject": r.test_subject,
                "status": r.status,
                **({"error": r.detail} if r.detail else {}),
            }
            for r in results
        ],
        "counts": {"completed": completed, "skipped": skipped, "failed": len(failed)},
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_default_output_root()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default="mpii_facegaze")
    parser.add_argument("--folds", default="", help="comma-separated subset, e.g. p03,p11")
    parser.add_argument("--epochs", type=int, default=None,
                        help="training epochs per fold; default = frozen Protocol B value (1)")
    parser.add_argument("--model", default="resnet50",
                        help="registered model name; default = frozen baseline 'resnet50'")
    args = parser.parse_args(argv)

    folds_filter = [f.strip() for f in args.folds.split(",") if f.strip()] or None
    summary = run_lopo(
        output_root=args.output,
        seed=args.seed,
        device=args.device,
        dataset_name=args.dataset,
        folds_filter=folds_filter,
        epochs=args.epochs,
        model_name=args.model,
    )
    print(json.dumps(summary["counts"], indent=2))
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
