"""Tests for features/data/loader.py — DataLoader class."""

import pytest
import pandas as pd
from features.data.loader import DataLoader


class TestLoadCsv:
    """Verify CSV loading and column normalization."""

    def test_load_csv_returns_dataframe(self, sample_csv_file):
        df = DataLoader.load_csv(str(sample_csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_csv_normalizes_columns(self, sample_csv_file):
        df = DataLoader.load_csv(str(sample_csv_file))
        # After normalization, PascalCase columns should exist
        assert "ID_Equipo" in df.columns or "id_equipo" in df.columns


class TestValidateSchema:
    """Verify schema validation logic."""

    def test_valid_schema_passes(self, sample_pascal_df):
        assert DataLoader.validate_schema(sample_pascal_df) is True

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            DataLoader.validate_schema(df)


class TestGetFeatures:
    """Verify feature column extraction."""

    def test_get_features_returns_correct_columns(self, sample_pascal_df):
        features = DataLoader.get_features(sample_pascal_df)
        expected = DataLoader.FEATURE_COLUMNS
        for col in expected:
            assert col in features.columns

    def test_get_features_excludes_targets(self, sample_pascal_df):
        features = DataLoader.get_features(sample_pascal_df)
        assert "estado_integridad_hardware" not in features.columns
        assert "nivel_riesgo_operativo" not in features.columns

    def test_get_features_on_empty_raises(self):
        df = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            DataLoader.get_features(df)


class TestGetTargets:
    """Verify target column extraction."""

    def test_get_targets_returns_target_columns(self, sample_pascal_df):
        targets = DataLoader.get_targets(sample_pascal_df)
        assert "estado_integridad_hardware" in targets.columns

    def test_get_targets_on_missing_returns_empty(self):
        df = pd.DataFrame({"foo": [1, 2]})
        targets = DataLoader.get_targets(df)
        assert targets.empty or len(targets.columns) == 0 or len(targets) == 0
