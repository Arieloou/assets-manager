"""Tests for features/data/loader.py — DataLoader class."""

import pandas as pd
import pytest

from features.data.loader import DataLoader


class TestLoadCsv:
    def test_load_csv_returns_dataframe(self, sample_csv_file):
        df = DataLoader.load_csv(str(sample_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_csv_has_expected_columns(self, sample_csv_file):
        df = DataLoader.load_csv(str(sample_csv_file))
        assert "device_id" in df.columns
        assert "operational_risk_level" in df.columns


class TestValidateSchema:
    def test_valid_schema_passes(self, sample_equipos_df):
        assert DataLoader.validate_schema(sample_equipos_df) is True

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            DataLoader.validate_schema(df)


class TestGetFeatures:
    def test_get_features_returns_feature_columns(self, sample_equipos_df):
        features = DataLoader.get_features(sample_equipos_df)
        for col in DataLoader.FEATURE_COLUMNS:
            assert col in features.columns

    def test_get_features_excludes_target(self, sample_equipos_df):
        features = DataLoader.get_features(sample_equipos_df)
        assert "operational_risk_level" not in features.columns

    def test_get_features_on_empty_raises(self):
        with pytest.raises((ValueError, KeyError)):
            DataLoader.get_features(pd.DataFrame())


class TestGetTargets:
    def test_get_targets_returns_target_column(self, sample_equipos_df):
        targets = DataLoader.get_targets(sample_equipos_df)
        assert "operational_risk_level" in targets.columns

    def test_get_targets_on_missing_returns_empty(self):
        targets = DataLoader.get_targets(pd.DataFrame({"foo": [1, 2]}))
        assert targets.empty or len(targets) == 0
