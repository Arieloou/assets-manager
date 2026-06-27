"""Tests for features/monitoring/data_drift.py — DataDriftDetector."""

import pytest
import pandas as pd
import numpy as np
from features.monitoring.data_drift import DataDriftDetector, DRIFT_THRESHOLD_P_VALUE


@pytest.fixture
def baseline_df():
    """Baseline DataFrame with known distributions."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Vida_Util_Consumida": np.random.normal(50, 10, n),
        "Costo_Mto_Reactivo_Acumulado": np.random.normal(100, 30, n),
    })


@pytest.fixture
def similar_df(baseline_df):
    """New data from the same distribution — should NOT trigger drift."""
    np.random.seed(99)
    n = 100
    return pd.DataFrame({
        "Vida_Util_Consumida": np.random.normal(50, 10, n),
        "Costo_Mto_Reactivo_Acumulado": np.random.normal(100, 30, n),
    })


@pytest.fixture
def drifted_df():
    """New data from a very different distribution — SHOULD trigger drift."""
    np.random.seed(7)
    n = 100
    return pd.DataFrame({
        "Vida_Util_Consumida": np.random.normal(90, 3, n),
        "Costo_Mto_Reactivo_Acumulado": np.random.normal(500, 10, n),
    })


class TestDataDriftDetector:
    """Verify KS-test-based data drift detection."""

    def test_no_drift_on_similar_data(self, baseline_df, similar_df):
        detector = DataDriftDetector(baseline_df)
        results = detector.check_drift(similar_df)

        for key, result in results.items():
            assert "ks_statistic" in result
            assert "p_value" in result
            assert "drift_detected" in result
            # Similar data should NOT trigger drift
            assert result["drift_detected"] == False

    def test_drift_on_shifted_data(self, baseline_df, drifted_df):
        detector = DataDriftDetector(baseline_df)
        results = detector.check_drift(drifted_df)

        # Both features should show drift
        for key, result in results.items():
            assert result["drift_detected"] == True
            assert result["p_value"] < DRIFT_THRESHOLD_P_VALUE

    def test_baseline_stats_computed(self, baseline_df):
        detector = DataDriftDetector(baseline_df)
        assert "vida_util" in detector.baseline_stats
        assert "costo_mto" in detector.baseline_stats
        assert "mean" in detector.baseline_stats["vida_util"]
        assert "std" in detector.baseline_stats["vida_util"]

    def test_empty_new_df_returns_empty(self, baseline_df):
        detector = DataDriftDetector(baseline_df)
        empty_df = pd.DataFrame()
        results = detector.check_drift(empty_df)
        assert len(results) == 0

    def test_partial_columns(self, baseline_df):
        """New data with only one of the monitored columns."""
        detector = DataDriftDetector(baseline_df)
        partial_df = pd.DataFrame({
            "Vida_Util_Consumida": np.random.normal(50, 10, 50),
        })
        results = detector.check_drift(partial_df)
        assert "vida_util" in results
        assert "costo_mto" not in results
