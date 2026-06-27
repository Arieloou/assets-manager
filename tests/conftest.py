"""Shared fixtures for all test modules."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- Synthetic DataFrame fixtures ---

@pytest.fixture
def sample_equipos_df():
    """Synthetic DataFrame simulating database 'equipos' records (snake_case cols)."""
    np.random.seed(42)
    n = 50
    ubicaciones = ["UDLAPARK", "GRANADOS", "COLON"]
    tipos = ["Computadora", "Impresora", "Servidor", "Router", "Switch"]
    estados = ["Excelente", "Bueno", "Regular", "Crítico"]
    riesgos = ["Bajo", "Medio", "Alto", "Crítico"]

    return pd.DataFrame({
        "id": [f"uuid-{i}" for i in range(n)],
        "id_equipo": [f"EQ-{i:04d}" for i in range(n)],
        "vida_util_consumida": np.random.uniform(5, 95, n).round(2),
        "tasa_incidencias_tecnicas": np.random.randint(0, 10, n),
        "tiempo_inactividad_acumulado": np.random.uniform(0, 500, n).round(2),
        "costo_mto_reactivo_acumulado": np.random.uniform(0, 400, n).round(2),
        "ubicacion_activo": np.random.choice(ubicaciones, n),
        "estado_integridad_hardware": np.random.choice(estados, n),
        "tipo_equipo": np.random.choice(tipos, n),
        "nivel_riesgo_operativo": np.random.choice(riesgos, n),
        "timestamp_registro": pd.date_range("2025-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def sample_pascal_df(sample_equipos_df):
    """Same data but with PascalCase column names (spec format)."""
    mapping = {
        "id_equipo": "ID_Equipo",
        "vida_util_consumida": "Vida_Util_Consumida",
        "tasa_incidencias_tecnicas": "Tasa_Incidencias_Tecnicas",
        "tiempo_inactividad_acumulado": "Tiempo_Inactividad_Acumulado",
        "costo_mto_reactivo_acumulado": "Costo_Mto_Reactivo_Acumulado",
        "ubicacion_activo": "Ubicacion_Activo",
        "estado_integridad_hardware": "Estado_Integridad_Hardware",
        "tipo_equipo": "Tipo_Equipo",
        "nivel_riesgo_operativo": "Nivel_Riesgo_Operativo",
    }
    df = sample_equipos_df.rename(columns=mapping)
    return df


@pytest.fixture
def sample_csv_file(tmp_path, sample_equipos_df):
    """Write a temp CSV file with snake_case columns for import tests."""
    csv_path = tmp_path / "test_data.csv"
    # Drop the 'id' column (UUID), keep the rest
    df = sample_equipos_df.drop(columns=["id", "timestamp_registro", "nivel_riesgo_operativo"])
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def trained_model_and_preprocessor(sample_pascal_df):
    """Return a fitted (trainer, preprocessor) pair for prediction tests."""
    from features.data import DataLoader, Preprocessor
    from features.model import ModelTrainer

    df = sample_pascal_df
    loader = DataLoader()
    preprocessor = Preprocessor()

    features = loader.get_features(df)
    targets = loader.get_targets(df)

    X = preprocessor.encode_categorical(features.copy(), fit=True)
    y = preprocessor.encode_target(targets.copy()[["Estado_Integridad_Hardware"]], fit=True)

    # Select only the model-ready columns
    feature_cols = [
        "Vida_Util_Consumida",
        "Tasa_Incidencias_Tecnicas",
        "Tiempo_Inactividad_Acumulado",
        "Costo_Mto_Reactivo_Acumulado",
        "Ubicacion_Activo_encoded",
        "Tipo_Equipo_encoded",
    ]
    X_ready = X[feature_cols]
    y_ready = y["Estado_Integridad_Hardware_encoded"]

    trainer = ModelTrainer()
    trainer.train(X_ready, y_ready)

    return trainer, preprocessor
