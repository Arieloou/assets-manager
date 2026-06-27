import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from features.config import get_vida_util_params, get_costo_params

DRIFT_THRESHOLD_P_VALUE = 0.05

class DataDriftDetector:
    def __init__(self, baseline_df):
        self.baseline_df = baseline_df
        self.baseline_stats = {}
        self._compute_baseline_stats()

    def _compute_baseline_stats(self):
        if "Vida_Util_Consumida" in self.baseline_df.columns:
            self.baseline_stats["vida_util"] = {
                "mean": self.baseline_df["Vida_Util_Consumida"].mean(),
                "std": self.baseline_df["Vida_Util_Consumida"].std(),
                "values": self.baseline_df["Vida_Util_Consumida"].values
            }
        if "Costo_Mto_Reactivo_Acumulado" in self.baseline_df.columns:
            self.baseline_stats["costo_mto"] = {
                "mean": self.baseline_df["Costo_Mto_Reactivo_Acumulado"].mean(),
                "std": self.baseline_df["Costo_Mto_Reactivo_Acumulado"].std(),
                "values": self.baseline_df["Costo_Mto_Reactivo_Acumulado"].values
            }

    def check_drift(self, new_df):
        results = {}
        for key, stats_dict in self.baseline_stats.items():
            if key == "vida_util" and "Vida_Util_Consumida" in new_df.columns:
                new_values = new_df["Vida_Util_Consumida"].values
                statistic, p_value = stats.ks_2samp(stats_dict["values"], new_values)
                results[key] = {
                    "ks_statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": p_value < DRIFT_THRESHOLD_P_VALUE
                }
            elif key == "costo_mto" and "Costo_Mto_Reactivo_Acumulado" in new_df.columns:
                new_values = new_df["Costo_Mto_Reactivo_Acumulado"].values
                statistic, p_value = stats.ks_2samp(stats_dict["values"], new_values)
                results[key] = {
                    "ks_statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": p_value < DRIFT_THRESHOLD_P_VALUE
                }
        return results

    def render_ui(self, new_df):
        st.subheader("Monitoreo de Data Drift")
        results = self.check_drift(new_df)

        for key, result in results.items():
            label = "Vida Útil Consumida" if key == "vida_util" else "Costo Mantenimiento"
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{label} - KS Statistic", f"{result['ks_statistic']:.4f}")
            with col2:
                st.metric(f"{label} - P-Value", f"{result['p_value']:.4f}")

            if result["drift_detected"]:
                st.error(f"⚠️ ALERTA: Drift detectado en {label}. El modelo puede no ser confiable para nuevos datos.")
            else:
                st.success(f"✓ Distribución de {label} estable")

        return results
