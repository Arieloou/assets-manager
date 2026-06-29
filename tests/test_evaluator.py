"""Tests for features/model/evaluator.py — ModelEvaluator class.

Confusion-matrix rendering is owned by
``features.monitoring.confusion_matrix.ConfusionMatrixMonitor`` (it computes its
own sklearn confusion matrix), so ``ModelEvaluator`` only exposes the
classification report / accuracy / summary metrics used for the Métricas tab.
"""

import pytest

from features.model.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator(model=None, preprocessor=None)


class TestClassificationReport:
    def test_report_is_dict(self, evaluator):
        report = evaluator.classification_report(["Bajo", "Alto"], ["Bajo", "Alto"])
        assert isinstance(report, dict)

    def test_report_has_accuracy(self, evaluator):
        report = evaluator.classification_report(["Bajo", "Alto"], ["Bajo", "Alto"])
        assert "accuracy" in report


class TestAccuracyScore:
    def test_perfect_accuracy(self, evaluator):
        assert evaluator.accuracy_score(["Bajo", "Alto"], ["Bajo", "Alto"]) == 1.0

    def test_zero_accuracy(self, evaluator):
        assert evaluator.accuracy_score(["Bajo", "Alto"], ["Alto", "Bajo"]) == 0.0


class TestSummaryMetrics:
    def test_summary_has_expected_keys(self, evaluator):
        y_true = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
        summary = evaluator.summary_metrics(y_true, y_true)
        for key in [
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ]:
            assert key in summary

    def test_perfect_summary_is_one(self, evaluator):
        y_true = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
        summary = evaluator.summary_metrics(y_true, y_true)
        assert summary["accuracy"] == 1.0
        assert summary["f1_macro"] == 1.0
