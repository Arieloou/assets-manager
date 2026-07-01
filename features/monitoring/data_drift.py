import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

from features.data.preprocessor import Preprocessor

DRIFT_THRESHOLD_P_VALUE = 0.05

# Derived/numeric features monitored for distribution drift.
MONITORED_FEATURES = {
    "useful_life_consumed_days": "Vida Útil Consumida",
    "technical_incident_rate": "Tasa de Incidencias",
    "days_since_last_corrective_maintenance": "Último Mto. Correctivo",
    "days_since_last_preventive_maintenance": "Último Mto. Preventivo",
}


class DataDriftDetector:
    """Detects distribution drift between a baseline and new data (KS test)."""

    def __init__(self, baseline_df):
        self.baseline_df = self._engineer(baseline_df)
        self.baseline_stats = {}
        self._compute_baseline_stats()

    @staticmethod
    def _engineer(df):
        """Add derived features when the raw date columns are present."""
        if df is None or df.empty:
            return pd.DataFrame()
        if "acquisition_date" in df.columns:
            try:
                return Preprocessor().engineer_features(df)
            except Exception:
                return df.copy()
        return df.copy()

    def _compute_baseline_stats(self):
        for key in MONITORED_FEATURES:
            if key in self.baseline_df.columns:
                values = pd.to_numeric(self.baseline_df[key], errors="coerce").dropna().values
                if len(values):
                    self.baseline_stats[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values,
                    }

    def check_drift(self, new_df):
        new_df = self._engineer(new_df)
        results = {}
        for key, stats_dict in self.baseline_stats.items():
            if key in new_df.columns:
                new_values = pd.to_numeric(new_df[key], errors="coerce").dropna().values
                if len(new_values) == 0:
                    continue
                statistic, p_value = stats.ks_2samp(stats_dict["values"], new_values)
                results[key] = {
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": bool(p_value < DRIFT_THRESHOLD_P_VALUE),
                }
        return results

    def render_ui(self, new_df):
        st.subheader("Monitoreo de Data Drift")
        st.caption(
            "Prueba de Kolmogórov–Smirnov entre los datos de referencia (baseline) y los "
            f"datos nuevos. Se considera *drift* si el p-valor < {DRIFT_THRESHOLD_P_VALUE}."
        )
        results = self.check_drift(new_df)

        if not results:
            st.info("No hay variables comparables entre el baseline y los nuevos datos.", icon=":material/info:")
            return results

        n_drift = sum(1 for r in results.values() if r["drift_detected"])
        c1, c2 = st.columns(2)
        c1.metric("Variables monitoreadas", len(results))
        c2.metric(
            "Variables con drift",
            n_drift,
            delta=None if n_drift == 0 else f"{n_drift} alerta(s)",
            delta_color="inverse",
            help="Variables cuya distribución cambió de forma significativa frente al baseline.",
        )

        if n_drift == 0:
            st.success("Todas las distribuciones se mantienen estables frente al baseline.", icon=":material/check_circle:")
        else:
            st.error(
                f"Se detectó drift en {n_drift} variable(s). El modelo puede no ser confiable "
                "para los nuevos datos; considere reentrenar.",
                icon=":material/crisis_alert:",
            )

        st.markdown("---")
        for key, result in results.items():
            label = MONITORED_FEATURES.get(key, key)
            drift = result["drift_detected"]
            icon = ":material/warning:" if drift else ":material/check_circle:"
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"{icon} **{label}**")
                st.caption("Drift detectado" if drift else "Distribución estable")
            col2.metric(
                "KS", f"{result['ks_statistic']:.3f}",
                help="Estadístico de Kolmogórov–Smirnov: distancia máxima entre las dos distribuciones.",
            )
            col3.metric(
                "p-valor", f"{result['p_value']:.3f}",
                help=f"Drift si p < {DRIFT_THRESHOLD_P_VALUE}.",
            )

        return results
