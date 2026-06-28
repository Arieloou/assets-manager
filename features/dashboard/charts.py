import streamlit as st
import plotly.express as px
import pandas as pd

def donut_chart_ubicacion(df):
    if df.empty or "ubicacion_activo" not in df.columns:
        st.info("No hay datos de ubicación disponibles")
        return

    counts = df["ubicacion_activo"].value_counts().reset_index()
    counts.columns = ["Ubicacion", "Cantidad"]

    fig = px.pie(
        counts,
        values="Cantidad",
        names="Ubicacion",
        hole=0.5,
        title="Distribución de Equipos por Ubicación",
        color="Ubicacion",
        color_discrete_map={
            "UDLAPARK": "#1f77b4",
            "GRANADOS": "#ff7f0e",
            "COLON": "#2ca02c"
        }
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width='stretch')

def bar_chart_estado(df):
    if df.empty or "estado_integridad_hardware" not in df.columns:
        st.info("No hay datos de estado disponibles")
        return

    counts = df["estado_integridad_hardware"].value_counts().reset_index()
    counts.columns = ["Estado", "Cantidad"]

    color_map = {
        "Excelente": "#2ecc71",
        "Bueno": "#3498db",
        "Regular": "#f39c12",
        "Crítico": "#e74c3c"
    }

    fig = px.bar(
        counts,
        x="Estado",
        y="Cantidad",
        title="Cantidad de Equipos por Estado de Integridad",
        color="Estado",
        color_discrete_map=color_map,
        text="Cantidad"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

def line_chart_tiempo(df):
    if df.empty or "timestamp_registro" not in df.columns:
        return

    df["timestamp_registro"] = pd.to_datetime(df["timestamp_registro"])
    temporal = df.groupby(df["timestamp_registro"].dt.date).size().reset_index(name="Equipos_Registrados")
    temporal.columns = ["Fecha", "Equipos Registrados"]

    fig = px.line(
        temporal,
        x="Fecha",
        y="Equipos Registrados",
        title="Registro de Equipos en el Tiempo"
    )
    st.plotly_chart(fig, width='stretch')

def scatter_plot_vida_costo(df):
    if df.empty or "vida_util_consumida" not in df.columns or "costo_mto_reactivo_acumulado" not in df.columns:
        return

    color_map = {
        "Excelente": "#2ecc71",
        "Bueno": "#3498db",
        "Regular": "#f39c12",
        "Crítico": "#e74c3c"
    }

    fig = px.scatter(
        df,
        x="vida_util_consumida",
        y="costo_mto_reactivo_acumulado",
        color="estado_integridad_hardware" if "estado_integridad_hardware" in df.columns else None,
        title="Vida Útil Consumida vs Costo de Mantenimiento",
        color_discrete_map=color_map,
        hover_data=["id_equipo"] if "id_equipo" in df.columns else None
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

def render_all_charts(df):
    st.subheader("Visualizaciones")

    tab1, tab2, tab3 = st.tabs(["Ubicación", "Estado", "Análisis"])

    with tab1:
        donut_chart_ubicacion(df)

    with tab2:
        bar_chart_estado(df)

    with tab3:
        scatter_plot_vida_costo(df)
        line_chart_tiempo(df)
