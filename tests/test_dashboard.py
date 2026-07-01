"""Tests for features/dashboard/ — FilterManager and correlation logic (no Streamlit UI)."""

import pytest

from features.dashboard.charts import build_correlation_frame
from features.dashboard.filters import FilterManager


class TestCorrelationFrame:
    def test_includes_studied_variables_and_target(self, sample_equipos_df):
        corr = build_correlation_frame(sample_equipos_df)
        for col in [
            "useful_life_consumed_days",
            "technical_incident_rate",
            "days_since_last_corrective_maintenance",
            "days_since_last_preventive_maintenance",
            "hardware_integrity_status",
            "device_brand",
            "device_type",
            "headquarters_location",
            "operational_risk_level",
        ]:
            assert col in corr.columns

    def test_is_square_with_unit_diagonal(self, sample_equipos_df):
        corr = build_correlation_frame(sample_equipos_df)
        assert corr.shape[0] == corr.shape[1]
        # Diagonal of a correlation matrix is 1.0
        assert all(abs(corr.iloc[i, i] - 1.0) < 1e-9 for i in range(corr.shape[0]))


class TestFilterManager:
    def test_init_stores_original(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        assert len(fm.original_df) == len(sample_equipos_df)

    def test_no_filters_returns_full_df(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        result = fm.apply_filters()
        assert len(result) == len(sample_equipos_df)

    def test_location_filter(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        fm._active_filters["location"] = ["Park"]
        result = fm.apply_filters()
        assert set(result["headquarters_location"].unique()) == {"Park"}

    def test_risk_level_filter(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        fm._active_filters["risk_level"] = ["Muy Alto"]
        result = fm.apply_filters()
        assert set(result["operational_risk_level"].unique()) == {"Muy Alto"}

    def test_brand_filter(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        fm._active_filters["brand"] = ["HP"]
        result = fm.apply_filters()
        assert set(result["device_brand"].unique()) == {"HP"}

    def test_useful_life_filter_derives_column_and_filters(self, sample_equipos_df):
        # FilterManager receives the raw df (only acquisition_date); it must
        # derive useful_life_consumed_days so the VIDA ÚTIL filter works.
        from features.config import get_useful_life_params

        fm = FilterManager(sample_equipos_df)
        assert "useful_life_consumed_days" in fm.original_df.columns

        low, high = get_useful_life_params()["OPERATIVO"]
        fm._active_filters["useful_life"] = "OPERATIVO"
        result = fm.apply_filters()

        # Filter must take effect (subset) and respect the configured range
        assert len(result) < len(sample_equipos_df)
        vals = result["useful_life_consumed_days"]
        assert ((vals >= low) & (vals < high)).all()

    def test_empty_filter_no_crash(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        fm._active_filters = {}
        result = fm.apply_filters()
        assert len(result) == len(sample_equipos_df)

    def test_original_df_is_copy(self, sample_equipos_df):
        fm = FilterManager(sample_equipos_df)
        fm.original_df.iloc[0, 0] = "MODIFIED"
        assert sample_equipos_df.iloc[0, 0] != "MODIFIED"
