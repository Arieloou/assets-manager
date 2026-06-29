import streamlit as st
import pandas as pd

from features.config import get_risk_levels

HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]


def display_total_devices(df):
    if df.empty:
        return None
    return st.metric("Total Equipos Registrados", len(df))


def display_high_risk_pct(df):
    if df.empty or "operational_risk_level" not in df.columns:
        return None
    high = df["operational_risk_level"].isin(HIGH_RISK_LEVELS).sum()
    pct = (high / len(df) * 100) if len(df) else 0
    return st.metric("Equipos en Riesgo Alto/Muy Alto", f"{pct:.1f}%")


def display_average_useful_life(df):
    if df.empty or "useful_life_consumed_days" not in df.columns:
        return None
    mean_days = pd.to_numeric(df["useful_life_consumed_days"], errors="coerce").mean()
    if pd.isna(mean_days):
        return None
    return st.metric("Vida Útil Promedio", f"{mean_days / 365.25:.1f} años")


def display_all_kpis(df):
    st.subheader("Indicadores Clave de Rendimiento")

    col1, col2, col3 = st.columns(3)
    with col1:
        display_total_devices(df)
    with col2:
        display_high_risk_pct(df)
    with col3:
        display_average_useful_life(df)

    if "operational_risk_level" in df.columns or "hardware_integrity_status" in df.columns:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if "operational_risk_level" in df.columns:
                st.write("**Distribución por Nivel de Riesgo Operativo**")
                counts = df["operational_risk_level"].value_counts()
                for level in get_risk_levels():
                    if level in counts:
                        st.write(f"- {level}: {counts[level]}")
        with col2:
            if "hardware_integrity_status" in df.columns:
                st.write("**Distribución por Estado de Integridad**")
                status_counts = df["hardware_integrity_status"].value_counts()
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")
