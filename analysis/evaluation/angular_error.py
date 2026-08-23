"""Mean 3D angular error — the locked evaluation metric.

Accumulates predictions/targets grouped by subject so leave-one-person-out
runs report both the overall mean and the per-subject breakdown required
by the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.gaze3d import angular_error_deg

METRIC_NAME = "mean_angular_error_deg"


@dataclass(frozen=True)
class AngularErrorSummary:
    metric: str
    mean_deg: float
    num_samples: int
    per_subject_deg: dict[str, float]
    per_subject_counts: dict[str, int]


@dataclass
class _Accumulator:
    errors: list[float] = field(default_factory=list)
    per_subject: dict[str, list[float]] = field(default_factory=dict)


class MeanAngularErrorEvaluator:
    """Streaming evaluator: add batches, then request a summary."""

    def __init__(self) -> None:
        self._acc = _Accumulator()

    def reset(self) -> None:
        self._acc = _Accumulator()

    def add(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        subject_ids: list[str] | None = None,
    ) -> None:
        predictions = np.atleast_2d(np.asarray(predictions, dtype=np.float64))
        targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
        if predictions.shape != targets.shape or predictions.shape[1] != 3:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}; "
                "expected (N, 3)."
            )
        if subject_ids is not None and len(subject_ids) != predictions.shape[0]:
            raise ValueError("subject_ids length must match batch size.")

        for index in range(predictions.shape[0]):
            error = angular_error_deg(predictions[index], targets[index])
            self._acc.errors.append(error)
            subject = subject_ids[index] if subject_ids is not None else "__all__"
            self._acc.per_subject.setdefault(subject, []).append(error)

    def summary(self) -> AngularErrorSummary:
        if not self._acc.errors:
            raise ValueError("No samples evaluated yet.")
        per_subject_deg = {
            subject: float(np.mean(errors))
            for subject, errors in sorted(self._acc.per_subject.items())
        }
        return AngularErrorSummary(
            metric=METRIC_NAME,
            mean_deg=float(np.mean(self._acc.errors)),
            num_samples=len(self._acc.errors),
            per_subject_deg=per_subject_deg,
            per_subject_counts={
                subject: len(errors) for subject, errors in sorted(self._acc.per_subject.items())
            },
        )
