"""Tests for features/data/preprocessor.py — Preprocessor class."""

import numpy as np
import pandas as pd
import pytest

from features.data.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor(reference_date=pd.Timestamp("2026-06-28"))


@pytest.fixture
def feature_df():
    """Minimal raw feature DataFrame (dates as day-first strings)."""
    return pd.DataFrame({
        "device_brand": ["HP", "Dell", "Canon"],
        "device_type": ["Laptop", "Proyector", "Impresora"],
        "hardware_integrity_status": ["Excelente", "Desgastado", "Crítico"],
        "headquarters_location": ["Park", "Granados", "Colon"],
        "acquisition_date": ["01/01/2020", "01/01/2022", "01/01/2024"],
        "technical_incident_rate": [0, 3, 7],
        "last_reactive_maintenance_date": ["01/01/2025", None, "01/06/2025"],
        "last_preventive_maintenance_date": ["01/01/2023", "01/01/2023", "01/01/2025"],
    })


@pytest.fixture
def target_df():
    return pd.DataFrame({"operational_risk_level": ["Muy Bajo", "Medio", "Muy Alto"]})


class TestEngineerFeatures:
    """Verify date-derived feature engineering."""

    def test_creates_derived_columns(self, preprocessor, feature_df):
        out = preprocessor.engineer_features(feature_df)
        for col in [
            "useful_life_consumed_days",
            "days_since_last_corrective_maintenance",
            "days_since_last_preventive_maintenance",
        ]:
            assert col in out.columns

    def test_useful_life_is_positive_days(self, preprocessor, feature_df):
        out = preprocessor.engineer_features(feature_df)
        # 2020-01-01 -> 2026-06-28 is ~2370 days
        assert out["useful_life_consumed_days"].iloc[0] > 2000

    def test_null_corrective_falls_back_to_useful_life(self, preprocessor, feature_df):
        out = preprocessor.engineer_features(feature_df)
        # Row index 1 has a null last_reactive_maintenance_date
        assert (
            out["days_since_last_corrective_maintenance"].iloc[1]
            == out["useful_life_consumed_days"].iloc[1]
        )

    def test_no_nan_in_corrective(self, preprocessor, feature_df):
        out = preprocessor.engineer_features(feature_df)
        assert not out["days_since_last_corrective_maintenance"].isna().any()


class TestBuildFeatures:
    """Verify the assembled feature matrix."""

    def test_returns_dataframe_without_nan(self, preprocessor, feature_df):
        X = preprocessor.build_features(feature_df, fit=True)
        assert isinstance(X, pd.DataFrame)
        assert not X.isna().any().any()

    def test_has_numeric_ordinal_and_onehot_columns(self, preprocessor, feature_df):
        X = preprocessor.build_features(feature_df, fit=True)
        assert "useful_life_consumed_days" in X.columns
        assert "technical_incident_rate" in X.columns
        assert "hardware_integrity_status_ord" in X.columns
        # One-hot expands location + type + brand
        assert any(c.startswith("headquarters_location_") for c in X.columns)
        assert any(c.startswith("device_type_") for c in X.columns)
        assert any(c.startswith("device_brand_") for c in X.columns)

    def test_quantitative_features_are_scaled(self, preprocessor, feature_df):
        X = preprocessor.build_features(feature_df, fit=True)
        # StandardScaler -> mean ~0 across the 4 quantitative columns
        quant = X[[
            "useful_life_consumed_days",
            "technical_incident_rate",
            "days_since_last_corrective_maintenance",
            "days_since_last_preventive_maintenance",
        ]]
        assert np.allclose(quant.mean().values, 0.0, atol=1e-6)
        # technical_incident_rate is now standardized (no longer raw 0/3/7)
        assert set(X["technical_incident_rate"].tolist()) != {0.0, 3.0, 7.0}

    def test_ordinal_respects_order(self, preprocessor, feature_df):
        X = preprocessor.build_features(feature_df, fit=True)
        # Excelente (0) < Desgastado (2) < Crítico (4)
        ords = X["hardware_integrity_status_ord"].tolist()
        assert ords == [0.0, 2.0, 4.0]

    def test_transform_after_fit_consistent(self, preprocessor, feature_df):
        preprocessor.build_features(feature_df, fit=True)
        X2 = preprocessor.build_features(feature_df, fit=False)
        assert list(X2.columns) == preprocessor.get_feature_names()


class TestEncodeTarget:
    """Verify ordinal encoding of operational_risk_level."""

    def test_encodes_to_ordered_integers(self, preprocessor, target_df):
        y = preprocessor.encode_target(target_df, fit=True)
        # Muy Bajo=0, Medio=2, Muy Alto=4
        assert list(y) == [0, 2, 4]

    def test_decode_recovers_labels(self, preprocessor, target_df):
        y = preprocessor.encode_target(target_df, fit=True)
        decoded = preprocessor.decode_target(y)
        assert list(decoded) == list(target_df["operational_risk_level"])

    def test_accepts_series(self, preprocessor):
        s = pd.Series(["Bajo", "Alto"])
        y = preprocessor.encode_target(s, fit=True)
        assert list(y) == [1, 3]


class TestGetFeatureNames:
    def test_returns_list_after_fit(self, preprocessor, feature_df):
        preprocessor.build_features(feature_df, fit=True)
        names = preprocessor.get_feature_names()
        assert isinstance(names, list)
        # 3 quant + 1 numeric + 1 ordinal + 3 location + 4 type + 5 brand = 17
        assert len(names) == 17
