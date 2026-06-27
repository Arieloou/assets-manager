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
        assert "Estado_Integridad_Hardware" in targets.columns

        # 3. Preprocess
        preprocessor = Preprocessor()
        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(
            targets.copy()[["Estado_Integridad_Hardware"]], fit=True
        )

        feature_cols = [
            "Vida_Util_Consumida", "Tasa_Incidencias_Tecnicas",
            "Tiempo_Inactividad_Acumulado", "Costo_Mto_Reactivo_Acumulado",
            "Ubicacion_Activo_encoded", "Tipo_Equipo_encoded",
        ]
        X_ready = X[feature_cols]
        y_ready = y["Estado_Integridad_Hardware_encoded"]

        # 4. Cross-validate
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(X_ready, y_ready, k=5)
        assert 0.0 <= cv_results["mean_accuracy"] <= 1.0
        assert len(cv_results["fold_scores"]) == 5

        # 5. Train
        trainer.train(X_ready, y_ready)
        assert trainer._feature_importances is not None

        # 6. Feature importance
        importance = trainer.get_feature_importance()
        assert len(importance) == 6
        assert abs(sum(importance.values()) - 1.0) < 0.01

        # 7. Predict single sample
        predictor = ModelPredictor(trainer.model, preprocessor)
        estado, riesgo = predictor.predict({
            "Vida_Util_Consumida": 60.0,
            "Tasa_Incidencias_Tecnicas": 5,
            "Tiempo_Inactividad_Acumulado": 200.0,
            "Costo_Mto_Reactivo_Acumulado": 250.0,
            "Ubicacion_Activo": "GRANADOS",
            "Tipo_Equipo": "Servidor",
        })
        assert estado in ["Excelente", "Bueno", "Regular", "Crítico"]
        assert riesgo in ["Bajo", "Medio", "Alto", "Critico", "Crítico"]

        # 8. Predict batch
        batch_result = predictor.predict_batch(features.copy())
        assert len(batch_result) == len(features)
        assert "Estado_Integridad_Hardware" in batch_result.columns
        assert "Nivel_Riesgo_Operativo" in batch_result.columns

        # 9. Evaluate
        evaluator = ModelEvaluator(trainer.model, preprocessor)
        y_true = targets["Estado_Integridad_Hardware"].tolist()
        y_pred = batch_result["Estado_Integridad_Hardware"].tolist()
        cm = evaluator.confusion_matrix(y_true, y_pred)
        assert cm.shape == (4, 4)
        acc = evaluator.accuracy_score(y_true, y_pred)
        assert 0.0 <= acc <= 1.0

        # 10. Data drift (baseline vs same data — no drift expected)
        baseline_features = features[["Vida_Util_Consumida", "Costo_Mto_Reactivo_Acumulado"]]
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
            targets.copy()[["Estado_Integridad_Hardware"]], fit=True
        )
        feature_cols = [
            "Vida_Util_Consumida", "Tasa_Incidencias_Tecnicas",
            "Tiempo_Inactividad_Acumulado", "Costo_Mto_Reactivo_Acumulado",
            "Ubicacion_Activo_encoded", "Tipo_Equipo_encoded",
        ]

        # Train and predict
        trainer = ModelTrainer()
        trainer.train(X[feature_cols], y["Estado_Integridad_Hardware_encoded"])
        predictor = ModelPredictor(trainer.model, preprocessor)

        sample_input = {
            "Vida_Util_Consumida": 30.0,
            "Tasa_Incidencias_Tecnicas": 1,
            "Tiempo_Inactividad_Acumulado": 50.0,
            "Costo_Mto_Reactivo_Acumulado": 80.0,
            "Ubicacion_Activo": "COLON",
            "Tipo_Equipo": "Router",
        }
        original_pred = predictor.predict(sample_input)

        # Save and reload
        model_path = tmp_path / "model.joblib"
        trainer.save_model(str(model_path))
        new_trainer = ModelTrainer()
        new_trainer.load_model(str(model_path))
        new_predictor = ModelPredictor(new_trainer.model, preprocessor)

        loaded_pred = new_predictor.predict(sample_input)

        # Predictions should be identical
        assert original_pred == loaded_pred
