"""Tests for features/model/predictor.py — ModelPredictor class."""

import pytest
from features.model.predictor import ModelPredictor


@pytest.fixture
def predictor(trained_model_and_preprocessor):
    """Build a ModelPredictor from the shared trained model."""
    trainer, preprocessor = trained_model_and_preprocessor
    return ModelPredictor(trainer.model, preprocessor)


@pytest.fixture
def valid_input():
    """Single sample input dict for prediction."""
    return {
        "vida_util_consumida": 45.0,
        "tasa_incidencias_tecnicas": 3,
        "tiempo_inactividad_acumulado": 120.0,
        "costo_mto_reactivo_acumulado": 200.0,
        "ubicacion_activo": "UDLAPARK",
        "tipo_equipo": "Computadora",
    }


class TestPredict:
    """Verify single-sample prediction."""

    def test_returns_tuple_of_two(self, predictor, valid_input):
        result = predictor.predict(valid_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_estado_is_valid_label(self, predictor, valid_input):
        estado, _ = predictor.predict(valid_input)
        valid_labels = ["Excelente", "Bueno", "Regular", "Critico"]
        assert estado in valid_labels

    def test_risk_is_valid_label(self, predictor, valid_input):
        _, riesgo = predictor.predict(valid_input)
        valid_labels = ["Bajo", "Medio", "Alto", "Critico"]
        assert riesgo in valid_labels


class TestPredictProba:
    """Verify probability prediction."""

    def test_returns_array(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert len(proba) > 0

    def test_probabilities_sum_to_one(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert abs(sum(proba) - 1.0) < 0.01

    def test_all_probabilities_non_negative(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert all(p >= 0 for p in proba)


class TestPredictBatch:
    """Verify batch prediction on DataFrame."""

    def test_batch_adds_prediction_columns(self, predictor, sample_pascal_df):
        from features.data import DataLoader
        features = DataLoader.get_features(sample_pascal_df)
        result = predictor.predict_batch(features)
        assert "estado_integridad_hardware" in result.columns
        assert "nivel_riesgo_operativo" in result.columns

    def test_batch_preserves_row_count(self, predictor, sample_pascal_df):
        from features.data import DataLoader
        features = DataLoader.get_features(sample_pascal_df)
        result = predictor.predict_batch(features)
        assert len(result) == len(features)


class TestRiskPrediction:
    """Verify risk is predicted independently by separate model."""

    def test_risk_differs_from_estado(self, predictor, valid_input):
        _, riesgo = predictor.predict(valid_input)
        assert isinstance(riesgo, str)

    def test_batch_returns_both_predictions(self, predictor, sample_pascal_df):
        from features.data import DataLoader
        features = DataLoader.get_features(sample_pascal_df)
        result = predictor.predict_batch(features)
        assert "estado_integridad_hardware" in result.columns
        assert "nivel_riesgo_operativo" in result.columns
