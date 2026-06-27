"""Tests for features/data/preprocessor.py — Preprocessor class."""

import pytest
import pandas as pd
import numpy as np
from features.data.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor()


@pytest.fixture
def feature_df():
    """Minimal feature DataFrame for encoding tests."""
    return pd.DataFrame({
        "Vida_Util_Consumida": [10.0, 50.0, 90.0],
        "Tasa_Incidencias_Tecnicas": [0, 3, 7],
        "Tiempo_Inactividad_Acumulado": [10.0, 100.0, 400.0],
        "Costo_Mto_Reactivo_Acumulado": [20.0, 150.0, 350.0],
        "Ubicacion_Activo": ["UDLAPARK", "GRANADOS", "COLON"],
        "Tipo_Equipo": ["Computadora", "Servidor", "Router"],
    })


@pytest.fixture
def target_df():
    """Minimal target DataFrame for encoding tests."""
    return pd.DataFrame({
        "Estado_Integridad_Hardware": ["Excelente", "Regular", "Crítico"],
    })


class TestEncodeCategorical:
    """Verify categorical encoding for Ubicacion_Activo and Tipo_Equipo."""

    def test_creates_encoded_columns(self, preprocessor, feature_df):
        result = preprocessor.encode_categorical(feature_df, fit=True)
        assert "Ubicacion_Activo_encoded" in result.columns
        assert "Tipo_Equipo_encoded" in result.columns

    def test_encoded_values_are_numeric(self, preprocessor, feature_df):
        result = preprocessor.encode_categorical(feature_df, fit=True)
        assert result["Ubicacion_Activo_encoded"].dtype in [np.int32, np.int64, np.intp]
        assert result["Tipo_Equipo_encoded"].dtype in [np.int32, np.int64, np.intp]

    def test_transform_without_fit_raises(self, preprocessor, feature_df):
        with pytest.raises(KeyError):
            preprocessor.encode_categorical(feature_df, fit=False)

    def test_transform_after_fit_consistent(self, preprocessor, feature_df):
        preprocessor.encode_categorical(feature_df, fit=True)
        result2 = preprocessor.encode_categorical(feature_df, fit=False)
        assert "Ubicacion_Activo_encoded" in result2.columns

    def test_does_not_modify_original(self, preprocessor, feature_df):
        original_cols = list(feature_df.columns)
        preprocessor.encode_categorical(feature_df, fit=True)
        assert list(feature_df.columns) == original_cols


class TestEncodeTarget:
    """Verify target encoding for Estado_Integridad_Hardware."""

    def test_creates_encoded_target(self, preprocessor, target_df):
        result = preprocessor.encode_target(target_df, fit=True)
        assert "Estado_Integridad_Hardware_encoded" in result.columns

    def test_encoded_target_is_numeric(self, preprocessor, target_df):
        result = preprocessor.encode_target(target_df, fit=True)
        assert result["Estado_Integridad_Hardware_encoded"].dtype in [np.int32, np.int64, np.intp]

    def test_inverse_transform_recovers_labels(self, preprocessor, target_df):
        result = preprocessor.encode_target(target_df, fit=True)
        encoded = result["Estado_Integridad_Hardware_encoded"].values
        decoded = preprocessor.target_encoder.inverse_transform(encoded)
        assert list(decoded) == list(target_df["Estado_Integridad_Hardware"])


class TestEncodeRiskLevel:
    """Verify manual risk-level encoding."""

    def test_maps_known_levels(self, preprocessor):
        series = pd.Series(["Bajo", "Medio", "Alto", "Critico"])
        result = preprocessor.encode_risk_level(series)
        assert list(result) == [0, 1, 2, 3]

    def test_unknown_level_returns_nan(self, preprocessor):
        series = pd.Series(["Desconocido"])
        result = preprocessor.encode_risk_level(series)
        assert pd.isna(result.iloc[0])


class TestGetFeatureNames:
    def test_returns_list(self, preprocessor):
        names = preprocessor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 6
