import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from features.config import (
    get_vida_util_params,
    get_incidencias_lambda,
    get_costo_params,
    get_locations,
    get_hardware_states,
    get_equipment_types,
)


class FilterManager:
    """Manages dynamic filtering for the dashboard."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with the original DataFrame.
        
        Args:
            df: Original DataFrame to filter
        """
        self.original_df = df.copy()
        self._filtered_df = df.copy()
        self._active_filters: Dict[str, Any] = {}

    def apply_filters(self) -> pd.DataFrame:
        """Apply all active filters and return the filtered DataFrame.
        
        Returns:
            Filtered DataFrame
        """
        self._filtered_df = self.original_df.copy()

        # Filter by VIDA ÚTIL
        if "vida_util" in self._active_filters:
            vida_util_key = self._active_filters["vida_util"]
            vida_params = get_vida_util_params()
            if vida_util_key in vida_params:
                mean, std, clip_low, clip_high = vida_params[vida_util_key]
                self._filtered_df = self._filtered_df[
                    (self._filtered_df["vida_util_consumida"] >= clip_low) &
                    (self._filtered_df["vida_util_consumida"] <= clip_high)
                ]

        # Filter by TASA INCIDENCIAS
        if "tasa_incidencias" in self._active_filters:
            tasa_key = self._active_filters["tasa_incidencias"]
            lambda_val = get_incidencias_lambda().get(tasa_key, 0)
            self._filtered_df = self._filtered_df[
                self._filtered_df["tasa_incidencias_tecnicas"] == lambda_val
            ]

        # Filter by COSTO REPARACIÓN
        if "costo_reparacion" in self._active_filters:
            costo_key = self._active_filters["costo_reparacion"]
            costo_ranges = get_costo_params()
            if costo_key in costo_ranges:
                min_costo, max_costo = costo_ranges[costo_key]
                self._filtered_df = self._filtered_df[
                    (self._filtered_df["costo_mto_reactivo_acumulado"] >= min_costo) &
                    (self._filtered_df["costo_mto_reactivo_acumulado"] <= max_costo)
                ]

        # Filter by UBICACIÓN
        if "ubicacion" in self._active_filters and self._active_filters["ubicacion"]:
            ubicaciones = self._active_filters["ubicacion"]
            self._filtered_df = self._filtered_df[
                self._filtered_df["ubicacion_activo"].isin(ubicaciones)
            ]

        # Filter by ESTADO INTEGRIDAD
        if "estado_integridad" in self._active_filters and self._active_filters["estado_integridad"]:
            estados = self._active_filters["estado_integridad"]
            self._filtered_df = self._filtered_df[
                self._filtered_df["estado_integridad_hardware"].isin(estados)
            ]

        # Filter by TIPO EQUIPO
        if "tipo_equipo" in self._active_filters and self._active_filters["tipo_equipo"]:
            tipos = self._active_filters["tipo_equipo"]
            self._filtered_df = self._filtered_df[
                self._filtered_df["tipo_equipo"].isin(tipos)
            ]

        return self._filtered_df

    def render_ui(self) -> pd.DataFrame:
        """Render Streamlit filter widgets and return filtered DataFrame.
        
        Returns:
            Filtered DataFrame after applying user-selected filters
        """
        st.subheader("Filtros")

        col1, col2, col3 = st.columns(3)

        # VIDA ÚTIL filter (single select)
        with col1:
            vida_util_options = list(get_vida_util_params().keys())
            vida_util_selected = st.selectbox(
                "VIDA ÚTIL",
                options=["TODOS"] + vida_util_options,
                index=0,
                key="filter_vida_util"
            )
            if vida_util_selected and vida_util_selected != "TODOS":
                self._active_filters["vida_util"] = vida_util_selected
            elif "vida_util" in self._active_filters:
                del self._active_filters["vida_util"]

        # TASA INCIDENCIAS filter (single select)
        with col2:
            tasa_options = list(get_incidencias_lambda().keys())
            tasa_selected = st.selectbox(
                "TASA INCIDENCIAS",
                options=["TODOS"] + tasa_options,
                index=0,
                key="filter_tasa_incidencias"
            )
            if tasa_selected and tasa_selected != "TODOS":
                self._active_filters["tasa_incidencias"] = tasa_selected
            elif "tasa_incidencias" in self._active_filters:
                del self._active_filters["tasa_incidencias"]

        # COSTO REPARACIÓN filter (single select)
        with col3:
            costo_options = list(get_costo_params().keys())
            costo_selected = st.selectbox(
                "COSTO REPARACIÓN",
                options=["TODOS"] + costo_options,
                index=0,
                key="filter_costo_reparacion"
            )
            if costo_selected and costo_selected != "TODOS":
                self._active_filters["costo_reparacion"] = costo_selected
            elif "costo_reparacion" in self._active_filters:
                del self._active_filters["costo_reparacion"]

        col4, col5, col6 = st.columns(3)

        # UBICACIÓN filter (multi-select)
        with col4:
            ubicaciones = get_locations()
            ubicacion_selected = st.multiselect(
                "UBICACIÓN",
                options=ubicaciones,
                default=[],
                key="filter_ubicacion"
            )
            if ubicacion_selected:
                self._active_filters["ubicacion"] = ubicacion_selected
            elif "ubicacion" in self._active_filters:
                del self._active_filters["ubicacion"]

        # ESTADO INTEGRIDAD filter (multi-select)
        with col5:
            estados = get_hardware_states()
            estado_selected = st.multiselect(
                "ESTADO INTEGRIDAD",
                options=estados,
                default=[],
                key="filter_estado_integridad"
            )
            if estado_selected:
                self._active_filters["estado_integridad"] = estado_selected
            elif "estado_integridad" in self._active_filters:
                del self._active_filters["estado_integridad"]

        # TIPO EQUIPO filter (multi-select)
        with col6:
            tipos = get_equipment_types()
            tipo_selected = st.multiselect(
                "TIPO EQUIPO",
                options=tipos,
                default=[],
                key="filter_tipo_equipo"
            )
            if tipo_selected:
                self._active_filters["tipo_equipo"] = tipo_selected
            elif "tipo_equipo" in self._active_filters:
                del self._active_filters["tipo_equipo"]

        return self.apply_filters()
