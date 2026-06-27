# -*- coding: utf-8 -*-
"""Model evaluator for confusion matrix and classification metrics."""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class ModelEvaluator:
    """Evaluates model performance using standard classification metrics."""

    TARGET_CLASSES = ['Excelente', 'Bueno', 'Regular', 'Critico']

    def __init__(self, model, preprocessor):
        """Initialize with a trained model and preprocessor.

        Args:
            model: Trained sklearn model (can be None for metrics-only usage)
            preprocessor: Preprocessor instance
        """
        self.model = model
        self.preprocessor = preprocessor

    def confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix as a labeled DataFrame.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            DataFrame with rows=actual, columns=predicted
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.TARGET_CLASSES)
        return pd.DataFrame(cm, index=self.TARGET_CLASSES, columns=self.TARGET_CLASSES)

    def classification_report(self, y_true, y_pred):
        """Generate classification report as a dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with per-class and overall metrics
        """
        report = classification_report(
            y_true, y_pred, labels=self.TARGET_CLASSES, output_dict=True
        )
        return report

    def accuracy_score(self, y_true, y_pred):
        """Compute overall accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Float accuracy score between 0 and 1
        """
        return float(accuracy_score(y_true, y_pred))

    def get_predictions_with_actuals(self, df):
        """Generate predictions and return alongside actual values.

        Args:
            df: DataFrame with a 'target' column and feature columns

        Returns:
            DataFrame with 'actual' and 'predicted' columns
        """
        X = df.drop(columns=['target'])
        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        return pd.DataFrame({
            'actual': df['target'],
            'predicted': predictions
        })

    def analyze_false_positives(self):
        """Placeholder for false positive analysis."""
        pass
