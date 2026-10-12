"""
Unit Tests for Custom Evaluation Metrics
Validates MAE, RMSE, VAF, Specificity, Confusion Matrix, and ROC/AUC logic.
"""

import numpy as np
import pytest
from src.metrics import (
    variance_accounted_for,
    evaluate_regression,
    calculate_specificity_per_class,
    evaluate_classification
)


def test_variance_accounted_for_perfect_prediction():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([100.0, 200.0, 300.0, 400.0])
    vaf = variance_accounted_for(y_true, y_pred)
    assert np.isclose(vaf, 100.0), f"Expected VAF = 100%, got {vaf}"


def test_evaluate_regression_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    res = evaluate_regression(y_true, y_pred, model_name="TestModel")

    assert res["MAE"] == pytest.approx(6.67, 0.01)
    assert res["RMSE"] > 0
    assert "VAF (%)" in res
    assert "R2 Score" in res


def test_calculate_specificity():
    # 2x2 confusion matrix: [[TP, FN], [FP, TN]]
    # Or for multi-class [3x3]:
    cm = np.array([
        [10, 2, 0],
        [1, 20, 1],
        [0, 2, 15]
    ])
    specs = calculate_specificity_per_class(cm)
    assert len(specs) == 3
    for s in specs:
        assert 0.0 <= s <= 1.0


def test_evaluate_classification_metrics():
    y_true = ["Budget", "Mid-Range", "Luxury", "Budget"]
    y_pred = ["Budget", "Mid-Range", "Luxury", "Mid-Range"]
    eval_res = evaluate_classification(y_true, y_pred, class_names=["Budget", "Luxury", "Mid-Range"])

    summary = eval_res["summary"]
    assert "Accuracy" in summary
    assert "Recall / Sensitivity (Macro)" in summary
    assert "Specificity (Macro)" in summary
    assert "F1-Score (Macro)" in summary
    assert summary["Accuracy"] == 0.75
