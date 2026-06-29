"""End-to-end integration test: load → preprocess → train → predict → evaluate."""

import pytest

from features.config import get_risk_levels
from features.data import DataLoader, Preprocessor
from features.model import ModelTrainer, ModelPredictor, ModelEvaluator
from features.monitoring.data_drift import DataDriftDetector


class TestEndToEndPipeline:
    def test_full_pipeline(self, sample_equipos_df):
        # 1. Load and validate
        loader = DataLoader()
        assert loader.validate_schema(sample_equipos_df)

        # 2. Extract features and target
        features = loader.get_features(sample_equipos_df)
        targets = loader.get_targets(sample_equipos_df)
        assert len(features) == len(sample_equipos_df)
        assert "operational_risk_level" in targets.columns

        # 3. Preprocess (date features + encoders)
        preprocessor = Preprocessor()
        X = preprocessor.build_features(features, fit=True)
        y = preprocessor.encode_target(targets, fit=True)
        assert not X.isna().any().any()

        # 4. Cross-validate
        trainer = ModelTrainer()
        cv = trainer.cross_validate(X, y, k=5)
        assert 0.0 <= cv["mean_accuracy"] <= 1.0
        assert len(cv["fold_scores"]) == 5

        # 5. Train
        trainer.train(X, y)
        assert trainer._feature_importances is not None

        # 6. Feature importance
        importance = trainer.get_feature_importance()
        assert set(importance.keys()) == set(preprocessor.get_feature_names())
        assert abs(sum(importance.values()) - 1.0) < 0.01

        # 7. Predict single sample + soft output
        predictor = ModelPredictor(trainer, preprocessor)
        sample = {
            "device_brand": "Epson",
            "device_type": "Proyector",
            "hardware_integrity_status": "Malo",
            "headquarters_location": "Granados",
            "acquisition_date": "01/06/2020",
            "technical_incident_rate": 8,
            "last_reactive_maintenance_date": None,
            "last_preventive_maintenance_date": "01/01/2024",
        }
        risk = predictor.predict(sample)
        assert risk in get_risk_levels()
        proba = predictor.predict_proba(sample)
        assert abs(sum(proba.values()) - 1.0) < 0.01

        # 8. Predict batch
        batch_result = predictor.predict_batch(features.copy())
        assert len(batch_result) == len(features)
        assert "operational_risk_level" in batch_result.columns

        # 9. Evaluate (classification report + accuracy)
        evaluator = ModelEvaluator(trainer, preprocessor)
        y_true = targets["operational_risk_level"].tolist()
        y_pred = batch_result["operational_risk_level"].tolist()
        report = evaluator.classification_report(y_true, y_pred)
        assert "accuracy" in report
        assert 0.0 <= evaluator.accuracy_score(y_true, y_pred) <= 1.0

        # 10. Data drift (same data -> no drift)
        detector = DataDriftDetector(sample_equipos_df)
        for result in detector.check_drift(sample_equipos_df).values():
            assert result["drift_detected"] is False

    def test_save_load_predict_consistency(self, sample_equipos_df, tmp_path):
        loader = DataLoader()
        preprocessor = Preprocessor()
        features = loader.get_features(sample_equipos_df)
        targets = loader.get_targets(sample_equipos_df)

        X = preprocessor.build_features(features, fit=True)
        y = preprocessor.encode_target(targets, fit=True)

        trainer = ModelTrainer()
        trainer.train(X, y)
        predictor = ModelPredictor(trainer, preprocessor)

        sample = {
            "device_brand": "HP",
            "device_type": "Computadora de Escritorio",
            "hardware_integrity_status": "Bueno",
            "headquarters_location": "Colon",
            "acquisition_date": "01/01/2021",
            "technical_incident_rate": 1,
            "last_reactive_maintenance_date": "01/01/2025",
            "last_preventive_maintenance_date": "01/06/2024",
        }
        original_pred = predictor.predict(sample)

        model_path = tmp_path / "model.joblib"
        trainer.save_model(str(model_path))
        new_trainer = ModelTrainer()
        new_trainer.load_model(str(model_path))
        loaded_pred = ModelPredictor(new_trainer, preprocessor).predict(sample)

        assert original_pred == loaded_pred
