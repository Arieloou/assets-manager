import streamlit as st
import pandas as pd
from typing import Dict, Any

from features.config import (
    get_useful_life_params,
    get_locations,
    get_brands,
    get_hardware_states,
    get_device_types,
    get_risk_levels,
)
from features.data.preprocessor import Preprocessor


class FilterManager:
    """Manages dynamic filtering for the dashboard."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        # Derive day-based features (useful_life_consumed_days, etc.) so the
        # "VIDA ÚTIL" filter has a column to operate on. The raw DataFrame only
        # carries acquisition_date, not the engineered useful_life_consumed_days.
        if not df.empty and "acquisition_date" in df.columns and "useful_life_consumed_days" not in df.columns:
            try:
                df = Preprocessor().engineer_features(df)
            except Exception:
                pass
        self.original_df = df
        self._filtered_df = df.copy()
        self._active_filters: Dict[str, Any] = {}

    def apply_filters(self) -> pd.DataFrame:
        """Apply all active filters and return the filtered DataFrame."""
        self._filtered_df = self.original_df.copy()

        # USEFUL LIFE range (days)
        if "useful_life" in self._active_filters and "useful_life_consumed_days" in self._filtered_df.columns:
            key = self._active_filters["useful_life"]
            ranges = get_useful_life_params()
            if key in ranges:
                low, high = ranges[key]
                vals = pd.to_numeric(self._filtered_df["useful_life_consumed_days"], errors="coerce")
                self._filtered_df = self._filtered_df[(vals >= low) & (vals < high)]

        # Categorical multi-selects
        for filter_key, column in [
            ("location", "headquarters_location"),
            ("brand", "device_brand"),
            ("status", "hardware_integrity_status"),
            ("device_type", "device_type"),
            ("risk_level", "operational_risk_level"),
        ]:
            if self._active_filters.get(filter_key) and column in self._filtered_df.columns:
                self._filtered_df = self._filtered_df[
                    self._filtered_df[column].isin(self._active_filters[filter_key])
                ]

        return self._filtered_df

    def _multiselect(self, col, label, options, filter_key):
        with col:
            selected = st.multiselect(label, options=options, default=[], key=f"filter_{filter_key}")
            if selected:
                self._active_filters[filter_key] = selected
            elif filter_key in self._active_filters:
                del self._active_filters[filter_key]

    def render_ui(self) -> pd.DataFrame:
        """Render Streamlit filter widgets and return the filtered DataFrame."""
        st.subheader("Filtros")

        col1, col2, col3 = st.columns(3)

        with col1:
            useful_life_options = list(get_useful_life_params().keys())
            useful_life_selected = st.selectbox(
                "VIDA ÚTIL", options=["TODOS"] + useful_life_options, index=0, key="filter_useful_life"
            )
            if useful_life_selected and useful_life_selected != "TODOS":
                self._active_filters["useful_life"] = useful_life_selected
            elif "useful_life" in self._active_filters:
                del self._active_filters["useful_life"]

        self._multiselect(col2, "UBICACIÓN", get_locations(), "location")
        self._multiselect(col3, "MARCA", get_brands(), "brand")

        col4, col5, col6 = st.columns(3)
        self._multiselect(col4, "ESTADO INTEGRIDAD", get_hardware_states(), "status")
        self._multiselect(col5, "TIPO EQUIPO", get_device_types(), "device_type")
        self._multiselect(col6, "NIVEL DE RIESGO", get_risk_levels(), "risk_level")

        return self.apply_filters()
