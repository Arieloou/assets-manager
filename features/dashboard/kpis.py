import streamlit as st
import pandas as pd

def display_tiempo_inactividad_promedio(df):
    if df.empty or "tiempo_inactividad_acumulado" not in df.columns:
        return None
    mean_val = df["tiempo_inactividad_acumulado"].mean()
    return st.metric("Tiempo Inactividad Promedio", f"{mean_val:.2f} hrs")

def display_costo_mantenimiento_promedio(df):
    if df.empty or "costo_mto_reactivo_acumulado" not in df.columns:
        return None
    mean_val = df["costo_mto_reactivo_acumulado"].mean()
    return st.metric("Costo Mantenimiento Promedio", f"${mean_val:.2f}")

def display_total_equipos(df):
    if df.empty:
        return None
    return st.metric("Total Equipos Registrados", len(df))

def display_estado_distribution(df):
    if df.empty or "estado_integridad_hardware" not in df.columns:
        return None
    counts = df["estado_integridad_hardware"].value_counts()
    return st.metric("Equipos por Estado", counts.to_dict())

def display_ubicacion_distribution(df):
    if df.empty or "ubicacion_activo" not in df.columns:
        return None
    counts = df["ubicacion_activo"].value_counts()
    return st.metric("Equipos por Ubicación", counts.to_dict())

def display_all_kpis(df):
    st.subheader("Indicadores Clave de Rendimiento")

    col1, col2, col3 = st.columns(3)

    with col1:
        display_total_equipos(df)
    with col2:
        display_tiempo_inactividad_promedio(df)
    with col3:
        display_costo_mantenimiento_promedio(df)

    if "estado_integridad_hardware" in df.columns:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Distribución por Estado de Integridad**")
            estado_counts = df["estado_integridad_hardware"].value_counts()
            for estado, count in estado_counts.items():
                st.write(f"- {estado}: {count}")
        with col2:
            if "ubicacion_activo" in df.columns:
                st.write("**Distribución por Ubicación**")
                ubicacion_counts = df["ubicacion_activo"].value_counts()
                for ubicacion, count in ubicacion_counts.items():
                    st.write(f"- {ubicacion}: {count}")
