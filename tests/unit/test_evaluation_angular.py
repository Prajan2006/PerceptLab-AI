"""Unit tests for the mean 3D angular error evaluator."""

import numpy as np
import pytest

from analysis.evaluation import MeanAngularErrorEvaluator


class TestMeanAngularErrorEvaluator:
    def test_perfect_predictions_zero_error(self):
        evaluator = MeanAngularErrorEvaluator()
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        evaluator.add(targets.copy(), targets, subject_ids=["p00", "p01"])
        summary = evaluator.summary()
        assert summary.metric == "mean_angular_error_deg"
        assert summary.mean_deg == pytest.approx(0.0, abs=1e-9)
        assert summary.num_samples == 2

    def test_orthogonal_predictions_ninety(self):
        evaluator = MeanAngularErrorEvaluator()
        targets = np.array([[1.0, 0.0, 0.0]])
        predictions = np.array([[0.0, 1.0, 0.0]])
        evaluator.add(predictions, targets, subject_ids=["p00"])
        assert evaluator.summary().mean_deg == pytest.approx(90.0, abs=1e-9)

    def test_overall_mean_is_sample_weighted(self):
        evaluator = MeanAngularErrorEvaluator()
        z = np.array([0.0, 0.0, 1.0])
        diag = np.array([1.0, 0.0, 1.0])  # 45° from z
        side = np.array([1.0, 0.0, 0.0])  # 90° from z

        # p00 contributes two 45° samples; p01 one 90° sample.
        evaluator.add(np.stack([diag, diag]), np.stack([z, z]), subject_ids=["p00", "p00"])
        evaluator.add(side[None, :], z[None, :], subject_ids=["p01"])

        summary = evaluator.summary()
        assert summary.per_subject_deg["p00"] == pytest.approx(45.0, abs=1e-9)
        assert summary.per_subject_deg["p01"] == pytest.approx(90.0, abs=1e-9)
        expected_overall = (45.0 + 45.0 + 90.0) / 3.0
        assert summary.mean_deg == pytest.approx(expected_overall, abs=1e-9)

    def test_unnormalized_inputs_ok(self):
        evaluator = MeanAngularErrorEvaluator()
        evaluator.add([[5.0, 0, 0]], [[7.0, 0, 0]], subject_ids=["p00"])
        assert evaluator.summary().mean_deg == pytest.approx(0.0, abs=1e-9)

    def test_shape_mismatch_rejected(self):
        evaluator = MeanAngularErrorEvaluator()
        with pytest.raises(ValueError):
            evaluator.add(np.zeros((2, 3)), np.zeros((3, 3)))
        with pytest.raises(ValueError):
            evaluator.add(np.zeros((2, 4)), np.zeros((2, 4)))

    def test_subject_length_mismatch_rejected(self):
        evaluator = MeanAngularErrorEvaluator()
        with pytest.raises(ValueError):
            evaluator.add(
                np.zeros((2, 3)), np.zeros((2, 3)), subject_ids=["only-one"]
            )

    def test_summary_before_any_data_raises(self):
        with pytest.raises(ValueError):
            MeanAngularErrorEvaluator().summary()

    def test_reset_clears_state(self):
        evaluator = MeanAngularErrorEvaluator()
        evaluator.add([[1, 0, 0]], [[0, 0, 1]], subject_ids=["p00"])
        evaluator.reset()
        with pytest.raises(ValueError):
            evaluator.summary()
