import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
import pandas as pd

from features.config import get_hardware_states, get_risk_levels
from features.data.preprocessor import Preprocessor

# Studied variables for the correlation matrix (engineered + raw numeric).
STUDIED_NUMERIC = [
    "useful_life_consumed_days",
    "technical_incident_rate",
    "days_since_last_corrective_maintenance",
    "days_since_last_preventive_maintenance",
]
NOMINAL_CATEGORICALS = ["device_brand", "device_type", "headquarters_location"]

ESTADO_COLOR_MAP = {
    "Excelente": "#2ecc71",
    "Bueno": "#3498db",
    "Desgastado": "#f39c12",
    "Malo": "#e67e22",
    "Crítico": "#e74c3c",
}

RIESGO_COLOR_MAP = {
    "Muy Bajo": "#2ecc71",
    "Bajo": "#27ae60",
    "Medio": "#f39c12",
    "Alto": "#e67e22",
    "Muy Alto": "#e74c3c",
}


def donut_chart_by_location(df):
    if df.empty or "headquarters_location" not in df.columns:
        st.info("No hay datos de ubicación disponibles")
        return

    counts = df["headquarters_location"].value_counts().reset_index()
    counts.columns = ["Ubicacion", "Cantidad"]

    fig = px.pie(
        counts,
        values="Cantidad",
        names="Ubicacion",
        hole=0.5,
        title="Distribución de Equipos por Ubicación",
        color="Ubicacion",
        color_discrete_map={
            "Park": "#1f77b4",
            "Granados": "#ff7f0e",
            "Colon": "#2ca02c",
        },
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width='stretch')


def bar_chart_device_status(df):
    if df.empty or "hardware_integrity_status" not in df.columns:
        st.info("No hay datos de estado disponibles")
        return

    counts = df["hardware_integrity_status"].value_counts().reset_index()
    counts.columns = ["Estado", "Cantidad"]

    fig = px.bar(
        counts,
        x="Estado",
        y="Cantidad",
        title="Cantidad de Equipos por Estado de Integridad",
        color="Estado",
        color_discrete_map=ESTADO_COLOR_MAP,
        category_orders={"Estado": get_hardware_states()},
        text="Cantidad",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


def bar_chart_risk_level(df):
    if df.empty or "operational_risk_level" not in df.columns:
        st.info("No hay datos de riesgo disponibles")
        return

    counts = df["operational_risk_level"].value_counts().reset_index()
    counts.columns = ["Riesgo", "Cantidad"]

    fig = px.bar(
        counts,
        x="Riesgo",
        y="Cantidad",
        title="Cantidad de Equipos por Nivel de Riesgo Operativo",
        color="Riesgo",
        color_discrete_map=RIESGO_COLOR_MAP,
        category_orders={"Riesgo": get_risk_levels()},
        text="Cantidad",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


def build_correlation_frame(df):
    """Build a numeric DataFrame of the studied variables for correlation.

    Quantitative day-based features and ``technical_incident_rate`` are used
    directly; ``hardware_integrity_status`` and the target ``operational_risk_level``
    are ordinal-encoded by their configured order; the nominal categoricals
    (brand, type, location) are label-encoded purely for the correlation view.
    """
    pre = Preprocessor()
    engineered = pre.engineer_features(df)

    corr_data = {}
    for col in STUDIED_NUMERIC:
        if col in engineered.columns:
            corr_data[col] = pd.to_numeric(engineered[col], errors="coerce")

    status_order = {value: i for i, value in enumerate(get_hardware_states())}
    if "hardware_integrity_status" in engineered.columns:
        corr_data["hardware_integrity_status"] = engineered["hardware_integrity_status"].map(status_order)

    for col in NOMINAL_CATEGORICALS:
        if col in engineered.columns:
            corr_data[col] = engineered[col].astype("category").cat.codes

    risk_order = {value: i for i, value in enumerate(get_risk_levels())}
    if "operational_risk_level" in engineered.columns:
        corr_data["operational_risk_level"] = engineered["operational_risk_level"].map(risk_order)

    return pd.DataFrame(corr_data).corr()


def render_correlation_matrix(df):
    """Render the correlation matrix of the studied variables as a heatmap."""
    if df.empty or "acquisition_date" not in df.columns:
        st.info("No hay datos suficientes para calcular la matriz de correlación")
        return

    corr = build_correlation_frame(df)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Matriz de Correlación de las Variables Estudiadas",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Las variables categóricas nominales (marca, tipo, ubicación) se codificaron "
        "numéricamente solo para el cálculo de correlación; su magnitud no implica orden."
    )


def render_all_charts(df):
    st.subheader("Visualizaciones")

    tab1, tab2, tab3 = st.tabs(["Ubicación", "Estado", "Riesgo"])

    with tab1:
        donut_chart_by_location(df)
    with tab2:
        bar_chart_device_status(df)
    with tab3:
        bar_chart_risk_level(df)
