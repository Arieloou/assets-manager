"""Tests for features/data/exporter.py — import/export functions."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from io import BytesIO
from features.data.exporter import (
    get_risk_level,
    validate_schema,
    import_csv,
    export_to_csv,
    save_to_database,
    get_historical_data,
)


class TestGetRiskLevel:
    """Verify risk-level derivation from hardware integrity state."""

    def test_excelente_maps_to_bajo(self):
        assert get_risk_level("Excelente") == "Bajo"

    def test_bueno_maps_to_medio(self):
        assert get_risk_level("Bueno") == "Medio"

    def test_regular_maps_to_alto(self):
        assert get_risk_level("Regular") == "Alto"

    def test_critico_maps_to_critico(self):
        assert get_risk_level("Crítico") == "Crítico"

    def test_unknown_defaults_to_medio(self):
        assert get_risk_level("Desconocido") == "Medio"


class TestValidateSchema:
    """Verify schema validation for CSV import."""

    def test_valid_schema(self):
        df = pd.DataFrame({
            "id_equipo": ["EQ-001"],
            "vida_util_consumida": [50.0],
            "tasa_incidencias_tecnicas": [2],
            "tiempo_inactividad_acumulado": [100.0],
            "costo_mto_reactivo_acumulado": [150.0],
            "ubicacion_activo": ["UDLAPARK"],
            "estado_integridad_hardware": ["Bueno"],
            "tipo_equipo": ["Computadora"],
        })
        assert validate_schema(df) is True

    def test_invalid_schema_missing_cols(self):
        df = pd.DataFrame({"foo": [1]})
        assert validate_schema(df) is False


class TestImportCsv:
    """Verify CSV import adds timestamp and risk level columns."""

    def test_import_adds_columns(self, sample_csv_file):
        with open(sample_csv_file, "rb") as f:
            df = import_csv(f)
        assert "timestamp_registro" in df.columns
        assert "nivel_riesgo_operativo" in df.columns

    def test_import_preserves_row_count(self, sample_csv_file):
        original = pd.read_csv(sample_csv_file)
        with open(sample_csv_file, "rb") as f:
            df = import_csv(f)
        assert len(df) == len(original)

    def test_import_invalid_schema_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(bad_csv, index=False)
        with open(bad_csv, "rb") as f:
            with pytest.raises(ValueError, match="Invalid CSV schema"):
                import_csv(f)


class TestExportToCsv:
    """Verify CSV export returns bytes."""

    def test_export_returns_bytes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = export_to_csv(df)
        assert isinstance(result, bytes)
        assert b"a,b" in result

    def test_export_roundtrip(self, sample_csv_file):
        original = pd.read_csv(sample_csv_file)
        csv_bytes = export_to_csv(original)
        recovered = pd.read_csv(BytesIO(csv_bytes))
        assert list(recovered.columns) == list(original.columns)
        assert len(recovered) == len(original)


class TestSaveToDatabase:
    """Verify save_to_database iterates rows and calls save_equipo."""

    @patch("features.data.exporter.save_equipo")
    def test_saves_correct_count(self, mock_save):
        df = pd.DataFrame({
            "id_equipo": ["EQ-001", "EQ-002"],
            "vida_util_consumida": [10.0, 20.0],
            "tasa_incidencias_tecnicas": [1, 2],
            "tiempo_inactividad_acumulado": [50.0, 100.0],
            "costo_mto_reactivo_acumulado": [30.0, 60.0],
            "ubicacion_activo": ["UDLAPARK", "COLON"],
            "estado_integridad_hardware": ["Bueno", "Crítico"],
            "tipo_equipo": ["Computadora", "Servidor"],
            "nivel_riesgo_operativo": ["Medio", "Crítico"],
            "timestamp_registro": ["2025-01-01", "2025-01-02"],
        })
        count = save_to_database(df)
        assert count == 2
        assert mock_save.call_count == 2


class TestGetHistoricalData:
    """Verify get_historical_data delegates to get_all_equipos."""

    @patch("features.data.exporter.get_all_equipos")
    def test_delegates_to_db(self, mock_get):
        mock_get.return_value = pd.DataFrame({"id": [1]})
        result = get_historical_data()
        mock_get.assert_called_once()
        assert len(result) == 1
