"""Shared fixtures for all test modules (English dataset schema)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


LOCATIONS = ["Park", "Granados", "Colon"]
DEVICE_TYPES = ["Computadora de Escritorio", "Laptop", "Proyector", "Impresora"]
BRANDS = ["HP", "Dell", "Epson", "NEC", "Canon"]
STATUSES = ["Excelente", "Bueno", "Desgastado", "Malo", "Crítico"]
RISK_LEVELS = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]

# Rough status -> risk mapping so the synthetic data carries a learnable signal.
STATUS_TO_RISK = dict(zip(STATUSES, RISK_LEVELS))


@pytest.fixture
def sample_equipos_df():
    """Synthetic DataFrame matching the dataset schema (English columns)."""
    rng = np.random.RandomState(42)
    n = 75  # 15 per risk class
    rows = []
    for i in range(n):
        status = STATUSES[i % len(STATUSES)]
        risk = STATUS_TO_RISK[status]
        # Acquisition between ~1 and ~7 years ago
        acq = pd.Timestamp("2026-01-01") - pd.Timedelta(days=int(rng.randint(365, 2555)))
        prev = acq + pd.Timedelta(days=int(rng.randint(30, 900)))
        # Every 3rd device has no corrective maintenance recorded
        if i % 3 == 0:
            reactive = None
        else:
            reactive = acq + pd.Timedelta(days=int(rng.randint(30, 1200)))

        rows.append({
            "device_id": f"EQ-{i:04d}",
            "device_brand": BRANDS[i % len(BRANDS)],
            "device_type": DEVICE_TYPES[i % len(DEVICE_TYPES)],
            "acquisition_date": acq.strftime("%d/%m/%Y"),
            "technical_incident_rate": int(rng.randint(0, 20)),
            "last_reactive_maintenance_date": reactive.strftime("%d/%m/%Y") if reactive is not None else None,
            "last_preventive_maintenance_date": prev.strftime("%d/%m/%Y"),
            "headquarters_location": LOCATIONS[i % len(LOCATIONS)],
            "hardware_integrity_status": status,
            "operational_risk_level": risk,
            "registered_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_pascal_df(sample_equipos_df):
    """Backwards-compatible alias."""
    return sample_equipos_df


@pytest.fixture
def sample_csv_file(tmp_path, sample_equipos_df):
    """Write a temp semicolon-separated CSV with the raw dataset columns."""
    csv_path = tmp_path / "test_data.csv"
    df = sample_equipos_df.drop(columns=["registered_at"])
    df.to_csv(csv_path, index=False, sep=";")
    return csv_path


@pytest.fixture
def trained_model_and_preprocessor(sample_equipos_df):
    """Return a fitted (trainer, preprocessor) tuple for the risk model."""
    from features.data import DataLoader, Preprocessor
    from features.model import ModelTrainer

    loader = DataLoader()
    preprocessor = Preprocessor()

    features = loader.get_features(sample_equipos_df)
    targets = loader.get_targets(sample_equipos_df)

    X = preprocessor.build_features(features, fit=True)
    y = preprocessor.encode_target(targets, fit=True)

    trainer = ModelTrainer()
    trainer.train(X, y)

    return trainer, preprocessor
