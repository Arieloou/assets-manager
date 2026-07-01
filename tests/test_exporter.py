"""Tests for features/data/exporter.py — import/persistence functions."""

from unittest.mock import patch

import pandas as pd
import pytest

from features.data.exporter import (
    validate_schema,
    import_csv,
    save_to_database,
    get_historical_data,
)


class TestValidateSchema:
    def test_valid_schema(self, sample_equipos_df):
        assert validate_schema(sample_equipos_df) is True

    def test_invalid_schema_missing_cols(self):
        assert validate_schema(pd.DataFrame({"foo": [1]})) is False


class TestImportCsv:
    def test_import_adds_timestamp(self, sample_csv_file):
        with open(sample_csv_file, "rb") as f:
            df = import_csv(f)
        assert "registered_at" in df.columns
        # Target already present in the dataset
        assert "operational_risk_level" in df.columns

    def test_import_preserves_row_count(self, sample_csv_file):
        original = pd.read_csv(sample_csv_file, sep=";")
        with open(sample_csv_file, "rb") as f:
            df = import_csv(f)
        assert len(df) == len(original)

    def test_import_invalid_schema_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(bad_csv, index=False)
        with open(bad_csv, "rb") as f:
            with pytest.raises(ValueError, match="Invalid CSV schema"):
                import_csv(f)


class TestSaveToDatabase:
    @patch("features.data.exporter.save_device")
    def test_saves_correct_count(self, mock_save, sample_equipos_df):
        count = save_to_database(sample_equipos_df)
        assert count == len(sample_equipos_df)
        assert mock_save.call_count == len(sample_equipos_df)

    @patch("features.data.exporter.save_device")
    def test_progress_callback_invoked_per_row(self, mock_save, sample_equipos_df):
        calls = []
        save_to_database(sample_equipos_df, progress_callback=lambda done, total: calls.append((done, total)))
        # Called once per row, ending at (n, n)
        assert len(calls) == len(sample_equipos_df)
        assert calls[-1] == (len(sample_equipos_df), len(sample_equipos_df))

    @patch("features.data.exporter.save_device")
    def test_null_reactive_parsed_as_none(self, mock_save, sample_equipos_df):
        save_to_database(sample_equipos_df)
        # Row 0 has a null last_reactive_maintenance_date in the fixture
        first_call = mock_save.call_args_list[0][0][0]
        assert first_call["last_reactive_maintenance_date"] is None
        assert first_call["acquisition_date"] is not None


class TestGetHistoricalData:
    @patch("features.data.exporter.get_all_devices")
    def test_delegates_to_db(self, mock_get):
        mock_get.return_value = pd.DataFrame({"id": [1]})
        result = get_historical_data()
        mock_get.assert_called_once()
        assert len(result) == 1
