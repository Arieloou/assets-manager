"""Tests for features/model/predictor.py — ModelPredictor class."""

import pytest

from features.config import get_risk_levels
from features.data import DataLoader
from features.model.predictor import ModelPredictor


@pytest.fixture
def predictor(trained_model_and_preprocessor):
    trainer, preprocessor = trained_model_and_preprocessor
    return ModelPredictor(trainer, preprocessor)


@pytest.fixture
def valid_input():
    return {
        "device_brand": "HP",
        "device_type": "Laptop",
        "hardware_integrity_status": "Desgastado",
        "headquarters_location": "Park",
        "acquisition_date": "11/10/2019",
        "technical_incident_rate": 5,
        "last_reactive_maintenance_date": None,
        "last_preventive_maintenance_date": "26/11/2024",
    }


class TestPredict:
    def test_returns_string_label(self, predictor, valid_input):
        result = predictor.predict(valid_input)
        assert isinstance(result, str)

    def test_label_is_valid_risk_level(self, predictor, valid_input):
        assert predictor.predict(valid_input) in get_risk_levels()


class TestPredictProba:
    """Soft output: vote proportion per risk level."""

    def test_returns_dict_for_all_levels(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert set(proba.keys()) == set(get_risk_levels())

    def test_probabilities_sum_to_one(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert abs(sum(proba.values()) - 1.0) < 0.01

    def test_all_probabilities_non_negative(self, predictor, valid_input):
        proba = predictor.predict_proba(valid_input)
        assert all(p >= 0 for p in proba.values())


class TestPredictBatch:
    def test_batch_adds_prediction_column(self, predictor, sample_equipos_df):
        features = DataLoader.get_features(sample_equipos_df)
        result = predictor.predict_batch(features)
        assert "operational_risk_level" in result.columns

    def test_batch_preserves_row_count(self, predictor, sample_equipos_df):
        features = DataLoader.get_features(sample_equipos_df)
        result = predictor.predict_batch(features)
        assert len(result) == len(features)

    def test_batch_labels_are_valid(self, predictor, sample_equipos_df):
        features = DataLoader.get_features(sample_equipos_df)
        result = predictor.predict_batch(features)
        assert set(result["operational_risk_level"]).issubset(set(get_risk_levels()))
