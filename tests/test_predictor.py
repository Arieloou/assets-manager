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
        "Vida_Util_Consumida": 45.0,
        "Tasa_Incidencias_Tecnicas": 3,
        "Tiempo_Inactividad_Acumulado": 120.0,
        "Costo_Mto_Reactivo_Acumulado": 200.0,
        "Ubicacion_Activo": "UDLAPARK",
        "Tipo_Equipo": "Computadora",
    }


class TestPredict:
    """Verify single-sample prediction."""

    def test_returns_tuple_of_two(self, predictor, valid_input):
        result = predictor.predict(valid_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_estado_is_valid_label(self, predictor, valid_input):
        estado, _ = predictor.predict(valid_input)
        valid_labels = ["Excelente", "Bueno", "Regular", "Crítico"]
        assert estado in valid_labels

    def test_risk_is_derived_from_estado(self, predictor, valid_input):
        estado, riesgo = predictor.predict(valid_input)
        expected_risk = ModelPredictor.RISK_MAPPING.get(estado, estado)
        assert riesgo == expected_risk


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
        assert "Estado_Integridad_Hardware" in result.columns
        assert "Nivel_Riesgo_Operativo" in result.columns

    def test_batch_preserves_row_count(self, predictor, sample_pascal_df):
        from features.data import DataLoader
        features = DataLoader.get_features(sample_pascal_df)
        result = predictor.predict_batch(features)
        assert len(result) == len(features)


class TestDeriveRiskLevel:
    """Verify risk-level derivation mapping."""

    def test_all_mappings(self, predictor):
        for estado, expected_risk in ModelPredictor.RISK_MAPPING.items():
            assert predictor._derive_risk_level(estado) == expected_risk

    def test_unknown_returns_same(self, predictor):
        assert predictor._derive_risk_level("Desconocido") == "Desconocido"
