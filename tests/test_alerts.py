"""Tests for features/alerts/ — EarlyWarningSystem and FeatureImportanceViewer."""

import pytest
from unittest.mock import patch, MagicMock
from features.alerts.early_warnings import EarlyWarningSystem, HIGH_VALUE_THRESHOLD


class TestEarlyWarningSystem:
    """Verify alert generation logic (DB calls are mocked)."""

    @patch("features.alerts.early_warnings.save_alert")
    def test_critico_estado_creates_alert(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        alerts = ews.check_prediction("EQ-001", "Crítico", "Crítico", {"costo_mto": 100})
        # Should create at least one alert for critical estado
        assert len(alerts) >= 1
        mock_save.assert_called()

    @patch("features.alerts.early_warnings.save_alert")
    def test_non_critico_no_alert(self, mock_save):
        ews = EarlyWarningSystem()
        alerts = ews.check_prediction("EQ-002", "Bueno", "Bajo", {"costo_mto": 50})
        assert len(alerts) == 0
        mock_save.assert_not_called()

    @patch("features.alerts.early_warnings.save_alert")
    def test_high_value_equipment_gets_alta_priority(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        # costo_mto > HIGH_VALUE_THRESHOLD should trigger ALTA priority
        ews.check_prediction("EQ-003", "Crítico", "Crítico", {"costo_mto": HIGH_VALUE_THRESHOLD + 100})
        # Check the first call's alert data has ALTA prioridad
        call_args = mock_save.call_args_list[0][0][0]
        assert call_args["prioridad"] == "ALTA"

    @patch("features.alerts.early_warnings.save_alert")
    def test_low_value_critico_gets_media_priority(self, mock_save):
        mock_save.return_value = MagicMock()
        ews = EarlyWarningSystem()
        ews.check_prediction("EQ-004", "Crítico", "Bajo", {"costo_mto": 50})
        call_args = mock_save.call_args_list[0][0][0]
        assert call_args["prioridad"] == "MEDIA"

    def test_high_value_threshold_is_positive(self):
        assert HIGH_VALUE_THRESHOLD > 0


class TestFeatureImportanceViewer:
    """Verify FeatureImportanceViewer initialization."""

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
