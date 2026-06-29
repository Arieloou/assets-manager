"""Tests for features/alerts/ — EarlyWarningSystem and FeatureImportanceViewer."""

from unittest.mock import patch, MagicMock

import pytest

from features.alerts.early_warnings import EarlyWarningSystem, HIGH_VALUE_TYPES


class TestEarlyWarningSystem:
    @patch("features.alerts.early_warnings.save_alert")
    def test_muy_alto_creates_alert(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        alerts = ews.check_prediction("EQ-001", "Muy Alto", {"device_type": "Laptop"})
        assert len(alerts) == 1
        mock_save.assert_called()

    @patch("features.alerts.early_warnings.save_alert")
    def test_low_risk_no_alert(self, mock_save):
        ews = EarlyWarningSystem()
        alerts = ews.check_prediction("EQ-002", "Bajo", {"device_type": "Laptop"})
        assert len(alerts) == 0
        mock_save.assert_not_called()

    @patch("features.alerts.early_warnings.save_alert")
    def test_muy_alto_is_alta_priority(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        ews.check_prediction("EQ-003", "Muy Alto", {"device_type": "Impresora"})
        call_args = mock_save.call_args_list[0][0][0]
        assert call_args["priority_level"] == "ALTA"

    @patch("features.alerts.early_warnings.save_alert")
    def test_alto_low_value_is_media_priority(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        ews.check_prediction("EQ-004", "Alto", {"device_type": "Impresora"})
        call_args = mock_save.call_args_list[0][0][0]
        assert call_args["priority_level"] == "MEDIA"

    @patch("features.alerts.early_warnings.save_alert")
    def test_alto_high_value_is_alta_priority(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        ews.check_prediction("EQ-005", "Alto", {"device_type": HIGH_VALUE_TYPES[0]})
        call_args = mock_save.call_args_list[0][0][0]
        assert call_args["priority_level"] == "ALTA"


class TestFeatureImportanceViewer:
    def test_init_stores_trainer(self, trained_model_and_preprocessor):
        from features.alerts.feature_importance import FeatureImportanceViewer
        trainer, _ = trained_model_and_preprocessor
        viewer = FeatureImportanceViewer(trainer)
        assert viewer.trainer is trainer

    def test_trainer_importance_accessible(self, trained_model_and_preprocessor):
        from features.alerts.feature_importance import FeatureImportanceViewer
        trainer, _ = trained_model_and_preprocessor
        viewer = FeatureImportanceViewer(trainer)
        importance = viewer.trainer.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
