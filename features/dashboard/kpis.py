import streamlit as st
import pandas as pd

def display_tiempo_inactividad_promedio(df):
    if df.empty or "Tiempo_Inactividad_Acumulado" not in df.columns:
        return None
    mean_val = df["Tiempo_Inactividad_Acumulado"].mean()
    return st.metric("Tiempo Inactividad Promedio", f"{mean_val:.2f} hrs")

def display_costo_mantenimiento_promedio(df):
    if df.empty or "Costo_Mto_Reactivo_Acumulado" not in df.columns:
        return None
    mean_val = df["Costo_Mto_Reactivo_Acumulado"].mean()
    return st.metric("Costo Mantenimiento Promedio", f"${mean_val:.2f}")

def display_total_equipos(df):
    if df.empty:
        return None
    return st.metric("Total Equipos Registrados", len(df))

def display_estado_distribution(df):
    if df.empty or "Estado_Integridad_Hardware" not in df.columns:
        return None
    counts = df["Estado_Integridad_Hardware"].value_counts()
    return st.metric("Equipos por Estado", counts.to_dict())

def display_ubicacion_distribution(df):
    if df.empty or "Ubicacion_Activo" not in df.columns:
        return None
    counts = df["Ubicacion_Activo"].value_counts()
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

    if "Estado_Integridad_Hardware" in df.columns:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Distribución por Estado de Integridad**")
            estado_counts = df["Estado_Integridad_Hardware"].value_counts()
            for estado, count in estado_counts.items():
                st.write(f"- {estado}: {count}")
        with col2:
            if "Ubicacion_Activo" in df.columns:
                st.write("**Distribución por Ubicación**")
                ubicacion_counts = df["Ubicacion_Activo"].value_counts()
                for ubicacion, count in ubicacion_counts.items():
                    st.write(f"- {ubicacion}: {count}")
