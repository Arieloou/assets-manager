"""Tests for features/model/trainer.py — ModelTrainer class."""

import pytest
import numpy as np
from features.model.trainer import ModelTrainer


class TestModelTrainerInit:
    """Verify model initialization from config params."""

    def test_creates_model(self):
        trainer = ModelTrainer()
        assert trainer.model is not None
        assert trainer._feature_importances is None


class TestTrain:
    """Verify model training."""

    def test_train_sets_feature_importances(self, trained_model_and_preprocessor):
        trainer, _, _ = trained_model_and_preprocessor
        importances = trainer._feature_importances
        assert importances is not None
        assert len(importances) == 6  # 6 features

    def test_train_returns_self(self, sample_pascal_df):
        from features.data import DataLoader, Preprocessor

        preprocessor = Preprocessor()
        features = DataLoader.get_features(sample_pascal_df)
        targets = DataLoader.get_targets(sample_pascal_df)

        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(targets.copy()[["estado_integridad_hardware"]], fit=True)

        feature_cols = [
            "vida_util_consumida", "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado", "costo_mto_reactivo_acumulado",
            "ubicacion_activo_encoded", "tipo_equipo_encoded",
        ]
        trainer = ModelTrainer()
        result = trainer.train(X[feature_cols], y["estado_integridad_hardware_encoded"])
        assert result is trainer


class TestCrossValidate:
    """Verify cross-validation returns expected structure."""

    def test_cv_returns_dict_with_keys(self, sample_pascal_df):
        from features.data import DataLoader, Preprocessor

        preprocessor = Preprocessor()
        features = DataLoader.get_features(sample_pascal_df)
        targets = DataLoader.get_targets(sample_pascal_df)

        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(targets.copy()[["estado_integridad_hardware"]], fit=True)

        feature_cols = [
            "vida_util_consumida", "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado", "costo_mto_reactivo_acumulado",
            "ubicacion_activo_encoded", "tipo_equipo_encoded",
        ]
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(
            X[feature_cols], y["estado_integridad_hardware_encoded"], k=5
        )

        assert "mean_accuracy" in cv_results
        assert "std_accuracy" in cv_results
        assert "fold_scores" in cv_results
        assert len(cv_results["fold_scores"]) == 5

    def test_cv_accuracy_in_range(self, sample_pascal_df):
        from features.data import DataLoader, Preprocessor

        preprocessor = Preprocessor()
        features = DataLoader.get_features(sample_pascal_df)
        targets = DataLoader.get_targets(sample_pascal_df)

        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(targets.copy()[["estado_integridad_hardware"]], fit=True)

        feature_cols = [
            "vida_util_consumida", "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado", "costo_mto_reactivo_acumulado",
            "ubicacion_activo_encoded", "tipo_equipo_encoded",
        ]
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(
            X[feature_cols], y["estado_integridad_hardware_encoded"], k=5
        )
        assert 0.0 <= cv_results["mean_accuracy"] <= 1.0


class TestGetFeatureImportance:
    """Verify feature importance retrieval after training."""

    def test_importance_dict_has_correct_keys(self, trained_model_and_preprocessor):
        trainer, _, _ = trained_model_and_preprocessor
        importance = trainer.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == 6
        for name in ModelTrainer.FEATURE_NAMES:
            assert name in importance

    def test_importance_values_sum_to_one(self, trained_model_and_preprocessor):
        trainer, _, _ = trained_model_and_preprocessor
        importance = trainer.get_feature_importance()
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01


class TestSaveLoadModel:
    """Verify model persistence with joblib."""

    def test_save_and_load_roundtrip(self, trained_model_and_preprocessor, tmp_path):
        trainer, _, _ = trained_model_and_preprocessor
        model_path = tmp_path / "test_model.joblib"
        trainer.save_model(str(model_path))

        new_trainer = ModelTrainer()
        new_trainer.load_model(str(model_path))

        # Loaded model should produce same feature importances
        original_imp = trainer.get_feature_importance()
        loaded_imp = new_trainer.get_feature_importance()
        for key in original_imp:
            assert abs(original_imp[key] - loaded_imp[key]) < 1e-6
