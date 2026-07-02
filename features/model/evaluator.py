# -*- coding: utf-8 -*-
"""Model evaluator for confusion matrix and classification metrics."""

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from features.config import get_risk_levels


class ModelEvaluator:
    """Evaluates risk-level predictions with standard classification metrics."""

    def __init__(self, model=None, preprocessor=None):
        """Initialize with an optional trained model and preprocessor.

        Args:
            model: Trained model (optional, only needed for live predictions).
            preprocessor: Preprocessor instance (optional).
        """
        self.model = model
        self.preprocessor = preprocessor
        self.target_classes = get_risk_levels()

    def classification_report(self, y_true, y_pred):
        """Classification report (precision / recall / F1) as a dict."""
        return classification_report(
            y_true, y_pred, labels=self.target_classes, output_dict=True, zero_division=0
        )

    def accuracy_score(self, y_true, y_pred):
        """Overall accuracy in [0, 1]."""
        return float(accuracy_score(y_true, y_pred))

    def summary_metrics(self, y_true, y_pred):
        """Headline metrics for the UI cards.

        Returns:
            Dict with accuracy plus macro/weighted precision, recall and F1.
        """
        report = self.classification_report(y_true, y_pred)
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        return {
            "accuracy": self.accuracy_score(y_true, y_pred),
            "precision_macro": macro.get("precision", 0.0),
            "recall_macro": macro.get("recall", 0.0),
            "f1_macro": macro.get("f1-score", 0.0),
        }
