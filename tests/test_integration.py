"""End-to-end integration test: load CSV → preprocess → train → predict → evaluate."""

import pytest
import pandas as pd
import numpy as np
from features.data import DataLoader, Preprocessor
from features.model import ModelTrainer, ModelPredictor, ModelEvaluator
from features.monitoring.data_drift import DataDriftDetector


class TestEndToEndPipeline:
    """Full pipeline integration test without database."""

    def test_full_pipeline(self, sample_pascal_df):
        # 1. Load and validate
        loader = DataLoader()
        assert loader.validate_schema(sample_pascal_df)

        # 2. Extract features and targets
        features = loader.get_features(sample_pascal_df)
        targets = loader.get_targets(sample_pascal_df)
        assert len(features) == len(sample_pascal_df)
        assert "estado_integridad_hardware" in targets.columns

        # 3. Preprocess
        preprocessor = Preprocessor()
        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(
            targets.copy()[["estado_integridad_hardware"]], fit=True
        )
        y_riesgo = preprocessor.encode_risk_target(
            targets.copy()[["nivel_riesgo_operativo"]], fit=True
        )

        feature_cols = [
            "vida_util_consumida", "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado", "costo_mto_reactivo_acumulado",
            "ubicacion_activo_encoded", "tipo_equipo_encoded",
        ]
        X_ready = X[feature_cols]
        y_ready = y["estado_integridad_hardware_encoded"]
        y_riesgo_ready = y_riesgo["nivel_riesgo_operativo_encoded"]

        # 4. Cross-validate
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(X_ready, y_ready, k=5)
        assert 0.0 <= cv_results["mean_accuracy"] <= 1.0
        assert len(cv_results["fold_scores"]) == 5

        # 5. Train
        trainer.train(X_ready, y_ready)
        assert trainer._feature_importances is not None

        # Train risk model
        trainer_risk = ModelTrainer()
        trainer_risk.train_risk(X_ready, y_riesgo_ready)

        # 6. Feature importance
        importance = trainer.get_feature_importance()
        assert len(importance) == 6
        assert abs(sum(importance.values()) - 1.0) < 0.01

        # 7. Predict single sample
        predictor = ModelPredictor(trainer.model, preprocessor, trainer_risk.model)
        estado, riesgo = predictor.predict({
            "vida_util_consumida": 60.0,
            "tasa_incidencias_tecnicas": 5,
            "tiempo_inactividad_acumulado": 200.0,
            "costo_mto_reactivo_acumulado": 250.0,
            "ubicacion_activo": "GRANADOS",
            "tipo_equipo": "Servidor",
        })
        assert estado in ["Excelente", "Bueno", "Regular", "Critico"]
        assert riesgo in ["Bajo", "Medio", "Alto", "Critico"]

        # 8. Predict batch
        batch_result = predictor.predict_batch(features.copy())
        assert len(batch_result) == len(features)
        assert "estado_integridad_hardware" in batch_result.columns
        assert "nivel_riesgo_operativo" in batch_result.columns

        # 9. Evaluate
        evaluator = ModelEvaluator(trainer.model, preprocessor)
        y_true = targets["estado_integridad_hardware"].tolist()
        y_pred = batch_result["estado_integridad_hardware"].tolist()
        cm = evaluator.confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)
        acc = evaluator.accuracy_score(y_true, y_pred)
        assert 0.0 <= acc <= 1.0

        # 10. Data drift (baseline vs same data — no drift expected)
        baseline_features = features[["vida_util_consumida", "costo_mto_reactivo_acumulado"]]
        detector = DataDriftDetector(baseline_features)
        drift_results = detector.check_drift(baseline_features)
        for key, result in drift_results.items():
            assert result["drift_detected"] == False

    def test_save_load_predict_consistency(self, sample_pascal_df, tmp_path):
        """Train, save, load, and verify predictions match."""
        loader = DataLoader()
        preprocessor = Preprocessor()
        features = loader.get_features(sample_pascal_df)
        targets = loader.get_targets(sample_pascal_df)

        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(
            targets.copy()[["estado_integridad_hardware"]], fit=True
        )
        y_riesgo = preprocessor.encode_risk_target(
            targets.copy()[["nivel_riesgo_operativo"]], fit=True
        )
        feature_cols = [
            "vida_util_consumida", "tasa_incidencias_tecnicas",
            "tiempo_inactividad_acumulado", "costo_mto_reactivo_acumulado",
            "ubicacion_activo_encoded", "tipo_equipo_encoded",
        ]

        # Train and predict
        trainer = ModelTrainer()
        trainer.train(X[feature_cols], y["estado_integridad_hardware_encoded"])
        
        trainer_risk = ModelTrainer()
        trainer_risk.train_risk(X[feature_cols], y_riesgo["nivel_riesgo_operativo_encoded"])
        
        predictor = ModelPredictor(trainer.model, preprocessor, trainer_risk.model)

        sample_input = {
            "vida_util_consumida": 30.0,
            "tasa_incidencias_tecnicas": 1,
            "tiempo_inactividad_acumulado": 50.0,
            "costo_mto_reactivo_acumulado": 80.0,
            "ubicacion_activo": "COLON",
            "tipo_equipo": "Router",
        }
        original_pred = predictor.predict(sample_input)

        # Save and reload
        model_path = tmp_path / "model.joblib"
        trainer.save_model(str(model_path))
        new_trainer = ModelTrainer()
        new_trainer.load_model(str(model_path))
        new_predictor = ModelPredictor(new_trainer.model, preprocessor, trainer_risk.model)

        loaded_pred = new_predictor.predict(sample_input)

        # Predictions should be identical
        assert original_pred == loaded_pred
