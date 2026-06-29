"""Tests for features/monitoring/data_drift.py — DataDriftDetector."""

import numpy as np
import pandas as pd
import pytest

from features.monitoring.data_drift import DataDriftDetector, DRIFT_THRESHOLD_P_VALUE


@pytest.fixture
def baseline_df():
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "useful_life_consumed_days": rng.normal(1500, 300, n),
        "technical_incident_rate": rng.normal(5, 2, n),
    })


@pytest.fixture
def similar_df():
    rng = np.random.RandomState(99)
    n = 100
    return pd.DataFrame({
        "useful_life_consumed_days": rng.normal(1500, 300, n),
        "technical_incident_rate": rng.normal(5, 2, n),
    })


@pytest.fixture
def drifted_df():
    rng = np.random.RandomState(7)
    n = 100
    return pd.DataFrame({
        "useful_life_consumed_days": rng.normal(4000, 100, n),
        "technical_incident_rate": rng.normal(18, 1, n),
    })


class TestDataDriftDetector:
    def test_no_drift_on_similar_data(self, baseline_df, similar_df):
        detector = DataDriftDetector(baseline_df)
        results = detector.check_drift(similar_df)
        assert results
        for result in results.values():
            assert {"ks_statistic", "p_value", "drift_detected"} <= set(result)
            assert result["drift_detected"] is False

    def test_drift_on_shifted_data(self, baseline_df, drifted_df):
        detector = DataDriftDetector(baseline_df)
        results = detector.check_drift(drifted_df)
        for result in results.values():
            assert result["drift_detected"] is True
            assert result["p_value"] < DRIFT_THRESHOLD_P_VALUE

    def test_baseline_stats_computed(self, baseline_df):
        detector = DataDriftDetector(baseline_df)
        assert "useful_life_consumed_days" in detector.baseline_stats
        assert "technical_incident_rate" in detector.baseline_stats
        assert "mean" in detector.baseline_stats["useful_life_consumed_days"]

    def test_empty_new_df_returns_empty(self, baseline_df):
        detector = DataDriftDetector(baseline_df)
        assert detector.check_drift(pd.DataFrame()) == {}

    def test_engineers_features_from_raw_dates(self, sample_equipos_df):
        """A raw dataset (with date columns) is engineered before comparison."""
        detector = DataDriftDetector(sample_equipos_df)
        assert "useful_life_consumed_days" in detector.baseline_stats
        results = detector.check_drift(sample_equipos_df)
        for result in results.values():
            assert result["drift_detected"] is False
