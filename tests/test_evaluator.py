"""Tests for features/model/evaluator.py — ModelEvaluator class."""

import pytest
import pandas as pd
import numpy as np
from features.model.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    """Evaluator with no real model (for metrics-only tests)."""
    return ModelEvaluator(model=None, preprocessor=None)


class TestConfusionMatrix:
    """Verify confusion matrix computation."""

    def test_confusion_matrix_shape(self, evaluator):
        y_true = ["Excelente", "Bueno", "Regular", "Critico", "Excelente"]
        y_pred = ["Excelente", "Regular", "Regular", "Critico", "Bueno"]
        cm = evaluator.confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)

    def test_confusion_matrix_is_dataframe(self, evaluator):
        y_true = ["Excelente", "Bueno"]
        y_pred = ["Excelente", "Bueno"]
        cm = evaluator.confusion_matrix(y_true, y_pred)
        assert isinstance(cm, pd.DataFrame)

    def test_perfect_predictions_diagonal(self, evaluator):
        labels = ["Excelente", "Bueno", "Regular", "Critico"]
        y_true = labels * 5
        y_pred = labels * 5
        cm = evaluator.confusion_matrix(y_true, y_pred)
        # All off-diagonal elements should be 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert cm.iloc[i, j] == 0


class TestClassificationReport:
    """Verify classification report generation."""

    def test_report_is_dict(self, evaluator):
        y_true = ["Excelente", "Bueno", "Regular"]
        y_pred = ["Excelente", "Bueno", "Regular"]
        report = evaluator.classification_report(y_true, y_pred)
        assert isinstance(report, dict)

    def test_report_has_accuracy(self, evaluator):
        y_true = ["Excelente", "Bueno"]
        y_pred = ["Excelente", "Bueno"]
        report = evaluator.classification_report(y_true, y_pred)
        assert "accuracy" in report


class TestAccuracyScore:
    """Verify accuracy score computation."""

    def test_perfect_accuracy(self, evaluator):
        y_true = ["Excelente", "Bueno", "Regular"]
        y_pred = ["Excelente", "Bueno", "Regular"]
        assert evaluator.accuracy_score(y_true, y_pred) == 1.0

    def test_zero_accuracy(self, evaluator):
        y_true = ["Excelente", "Bueno"]
        y_pred = ["Bueno", "Excelente"]
        assert evaluator.accuracy_score(y_true, y_pred) == 0.0

    def test_partial_accuracy(self, evaluator):
        y_true = ["Excelente", "Bueno", "Regular", "Critico"]
        y_pred = ["Excelente", "Bueno", "Bueno", "Bueno"]
        score = evaluator.accuracy_score(y_true, y_pred)
        assert score == 0.5
