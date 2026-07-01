import streamlit as st
import pandas as pd

HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]


def display_all_kpis(df):
    st.subheader("Indicadores Clave de Rendimiento")

    if df.empty:
        st.info("Carga datos para ver los indicadores clave.", icon=":material/info:")
        return

    total = len(df)
    risk_col = "operational_risk_level" in df.columns
    status_col = "hardware_integrity_status" in df.columns

    # --- Fila 1: visión general ------------------------------------------------
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total de equipos",
        f"{total:,}".replace(",", "."),
        help="Número de equipos registrados en el conjunto filtrado.",
    )

    if risk_col:
        high = int(df["operational_risk_level"].isin(HIGH_RISK_LEVELS).sum())
        pct = (high / total * 100) if total else 0.0
        c2.metric(
            "Riesgo Alto / Muy Alto",
            f"{pct:.1f}%",
            delta=f"{high} equipos",
            delta_color="inverse",  # más equipos en riesgo = peor (rojo)
            help="Porcentaje de equipos clasificados como Alto o Muy Alto. "
                 "Estos son los que requieren atención prioritaria.",
        )
    else:
        c2.metric("Riesgo Alto / Muy Alto", "—")

    if "useful_life_consumed_days" in df.columns:
        mean_days = pd.to_numeric(df["useful_life_consumed_days"], errors="coerce").mean()
        value = f"{mean_days / 365.25:.1f} años" if pd.notna(mean_days) else "—"
        c3.metric(
            "Vida útil promedio",
            value,
            help="Promedio del tiempo transcurrido desde la adquisición de los equipos.",
        )
    else:
        c3.metric("Vida útil promedio", "—")

    # --- Fila 2: focos de atención --------------------------------------------
    c4, c5, c6 = st.columns(3)

    if risk_col:
        muy_alto = int((df["operational_risk_level"] == "Muy Alto").sum())
        c4.metric(
            "Equipos en Riesgo Muy Alto",
            f"{muy_alto}",
            help="Equipos en el nivel de riesgo más crítico. No fallar en detectarlos es la prioridad del modelo.",
        )

    if status_col:
        critico = int((df["hardware_integrity_status"] == "Crítico").sum())
        c5.metric(
            "Equipos en Estado Crítico",
            f"{critico}",
            help="Equipos cuyo estado de integridad de hardware es 'Crítico'.",
        )

    if "technical_incident_rate" in df.columns:
        mean_rate = pd.to_numeric(df["technical_incident_rate"], errors="coerce").mean()
        value = f"{mean_rate:.1f}" if pd.notna(mean_rate) else "—"
        c6.metric(
            "Tasa de incidencias promedio",
            value,
            help="Promedio de incidencias técnicas reportadas por equipo.",
        )
