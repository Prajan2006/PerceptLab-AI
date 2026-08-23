# PerceptLab-AI

**Human-Centered Gaze Estimation with Deep Learning**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu126-orange)

---

## Overview

PerceptLab-AI is a research-oriented computer-vision project studying gaze estimation from facial images using deep learning. The project maintains an end-to-end pipeline for the MPIIFaceGaze dataset — annotation parsing, appearance-based preprocessing, PyTorch dataset/loading layers, model training, and subject-independent evaluation — with every experimental variable controlled and recorded.

## Research Question

> **Under fixed preprocessing, architecture, training settings, and subject-independent LOPO
> evaluation, does adding localized eye-region information to a full-face RGB gaze-estimation
> baseline improve mean angular error on MPIIFaceGaze?**

This question drives the current experimental series. The completed training-duration study below establishes the reference configuration against which that comparison will be run.

## Current Status

| State | Item |
|---|---|
| ✅ Completed | MPIIFaceGaze pipeline established (validated: 37,667 image–annotation matches) |
| ✅ Completed | ResNet-50 baseline architecture |
| ✅ Completed | 15-fold subject-independent LOPO evaluation |
| ✅ Completed | Frozen 1-epoch baseline (`engineering_baseline_v1`) |
| ✅ Completed | E1 vs E3 training-duration experiment (`engineering_baseline_v1_E3`) |
| ✅ Completed | E1 vs E3 analysis ([analysis/reports/E1_vs_E3_analysis.md](analysis/reports/E1_vs_E3_analysis.md)) |
| ✅ Completed | Reproducibility / checkpoint verification (15/15 checkpoints load; finite inference) |
| ✅ Completed | GitHub repository |
| 🔄 In progress | Moving from training-duration analysis toward the controlled input-representation experiment |

## Frozen Baseline

The 1-epoch result is kept as a **frozen engineering reference** for all future comparisons.
It is not a state-of-the-art claim.

| Setting | Value |
|---|---|
| Dataset | MPIIFaceGaze |
| Model | ResNet-50 |
| Input | Full-face RGB |
| Evaluation | 15-fold subject-independent LOPO |
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 0 |
| Batch size | 8 |
| Seed | 0 |
| Baseline epochs | 1 |

**Result:** mean angular error **11.2045°**, population std dev **±2.2633°**,
best fold **7.087°**, worst fold **15.628°** — **15/15 folds completed, 0 failed, 0 skipped**.

## Training-Duration Experiment

E3 repeated the frozen recipe with a single changed variable: training duration
(1 epoch → 3 epochs). Everything else — preprocessing, architecture, optimizer,
batch size, seed, splits, evaluator — was held fixed.

| Metric | E1 (1 epoch) | E3 (3 epochs) | Δ |
|---|---|---|---|
| Mean angular error | 11.2045° | 9.7642° | **−1.4403° (−12.9%)** |
| Improved folds | — | — | **13 of 15** |

Caveats recorded with the result:

- 2 folds worsened; **p03 was a notable degradation** (+25.2%, with a rising validation-loss trajectory).
- Three epochs is **not** claimed to be universally optimal; intermediate durations were not tested.
- The finding informs the next controlled experiment rather than closing it.

Full fold-level table and statistics:
[analysis/reports/E1_vs_E3_analysis.md](analysis/reports/E1_vs_E3_analysis.md)

## Evaluation Protocol

Evaluation uses leave-one-person-out (LOPO): each fold holds out one participant,
trains on the remaining participants, evaluates on the held-out participant, and
computes gaze angular error between predicted and ground-truth 3-D directions.
Results are aggregated across all 15 folds.

## Project Structure

```text
PerceptLab-AI/
├── analysis/
│   ├── evaluation/          # angular-error metric implementation
│   └── reports/             # read-only experiment analyses
├── backend/                 # FastAPI services (API, camera gateway, websockets)
├── camera/                  # capture interfaces and device abstraction
├── config/
│   ├── experiments/         # experiment configuration examples
│   ├── training/            # frozen Protocol B baseline configuration
│   └── datasets.json        # dataset registry
├── data/
│   ├── mpiifacegaze/        # raw-layout parser, dataset layer, synthetic fixtures
│   ├── preprocessing/       # GazeHub-style face/eye preprocessing
│   ├── splits.py            # deterministic LOPO fold generation
│   └── gaze3d.py            # 3-D gaze conventions and metrics helpers
├── models/
│   ├── registry.py          # declared input/output contracts
│   └── resnet50.py          # ResNet-50 gaze architecture
├── scripts/
│   └── train_lopo.py        # LOPO training runner
├── tests/
│   ├── unit/                # component tests (synthetic fixtures)
│   ├── integration/         # real-data gated tests
│   └── backend/             # service tests
├── training/
│   ├── trainer.py           # minimal epoch trainer
│   └── baseline.py          # frozen baseline configuration loader
├── frontend/                # TypeScript dashboard
└── requirements.txt
```

Datasets, checkpoints, and generated experiment outputs live outside version control.

## Reproducibility

Every run records its configuration, seed, evaluation protocol, per-fold metrics,
training history, and timing; analyses are written as read-only reports. The frozen
baseline configuration is validated by tests so undocumented settings cannot enter a run.
Datasets, checkpoints, generated experiment outputs, caches, environments, and secrets
are excluded from version control via `.gitignore`.

## Environment

| Component | Version |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.13.0+cu126 |
| Torchvision | 0.28.0+cu126 |
| GPU | NVIDIA GeForce RTX 3050 (4 GB VRAM) |

## Controlled Experimental Approach

Experiments change one major variable at a time while keeping the evaluation
protocol and all relevant settings fixed:

```text
Frozen baseline (1 epoch)
      ↓
Training-duration analysis (3 epochs)      ← completed
      ↓
Input-representation experiment            ← next
      ↓
Subject-level analysis
      ↓
Research synthesis
```

## Technical Preparation

Practical experience in Python, PyTorch and basic machine learning, computer vision,
and evaluation/experimental analysis — with each of these areas being strengthened
through this project's controlled studies.

## Related Project

- [Human Activity Recognition](https://github.com/Prajan2006/Human-activity-recognition)

## Repository

[github.com/Prajan2006/PerceptLab-AI](https://github.com/Prajan2006/PerceptLab-AI)

## Current Research Roadmap

1. Finish/interpret the training-duration experiment
2. Establish the next reference configuration
3. Run the localized eye-region vs full-face experiment
4. Analyse subject-level behaviour
5. Synthesize findings and prepare the research handoff

## Author

**Prajan**
