"""Paleta y utilidades de estilo compartidas por toda la aplicación.

Este módulo es la **única fuente de verdad** para los colores. Los módulos de
dashboard, monitoreo y alertas importan estas constantes para que el nivel de
riesgo se codifique con el MISMO color en todas partes (KPIs, gráficos,
resultado de predicción y matrices de confusión).

Principio de accesibilidad: el color nunca va solo. Cada nivel de riesgo tiene
también un ícono y una etiqueta de estado (`success`/`warning`/`error`) para no
depender únicamente del color.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Identidad / paleta base
# -----------------------------------------------------------------------------
PRIMARY = "#2563EB"          # Azul corporativo (coincide con .streamlit/config.toml)
NEUTRAL = "#64748B"          # slate-500 para textos/series neutras

# Paleta categórica para series sin semántica de riesgo (ubicación, marca, ...).
CATEGORICAL = [
    "#2563EB", "#0EA5E9", "#14B8A6", "#8B5CF6",
    "#F59E0B", "#EC4899", "#10B981", "#6366F1",
]

# -----------------------------------------------------------------------------
# Colores semánticos de RIESGO (5 clases, en orden de severidad ascendente).
# El orden coincide con config.yaml -> risk_levels.
# -----------------------------------------------------------------------------
RISK_COLORS = {
    "Muy Bajo": "#15803D",   # verde profundo
    "Bajo":     "#65A30D",   # verde lima
    "Medio":    "#F59E0B",   # ámbar
    "Alto":     "#EA580C",   # naranja
    "Muy Alto": "#DC2626",   # rojo
}

# Ícono por nivel (no depender solo del color) + etiqueta de estado de Streamlit.
# Iconos de Material Symbols (sintaxis ":material/...:"), severidad ascendente.
RISK_ICONS = {
    "Muy Bajo": ":material/verified:",
    "Bajo":     ":material/check_circle:",
    "Medio":    ":material/info:",
    "Alto":     ":material/warning:",
    "Muy Alto": ":material/dangerous:",
}

RISK_STATUS = {           # mapea a st.success / st.warning / st.error
    "Muy Bajo": "success",
    "Bajo":     "success",
    "Medio":    "warning",
    "Alto":     "error",
    "Muy Alto": "error",
}

# -----------------------------------------------------------------------------
# Colores semánticos del ESTADO DE INTEGRIDAD del hardware (misma rampa).
# Orden ordinal de mejor a peor (config.yaml -> hardware_states).
# -----------------------------------------------------------------------------
STATUS_COLORS = {
    "Excelente":   "#15803D",
    "Bueno":       "#65A30D",
    "Desgastado":  "#F59E0B",
    "Malo":        "#EA580C",
    "Crítico":     "#DC2626",
}

# Colores por ubicación (categórico, sin orden semántico).
LOCATION_COLORS = {
    "Park":     "#2563EB",
    "Granados": "#0EA5E9",
    "Colon":    "#14B8A6",
}

# -----------------------------------------------------------------------------
# Escalas continuas reutilizables
# -----------------------------------------------------------------------------
# Divergente para correlaciones (-1 azul ... +1 rojo), seguro para daltonismo.
CORRELATION_SCALE = "RdBu_r"
# Secuencial para matrices de confusión / conteos (intensidad creciente).
COUNT_SCALE = "Blues"


def risk_color(level: str) -> str:
    """Color semántico de un nivel de riesgo (gris neutro si es desconocido)."""
    return RISK_COLORS.get(level, NEUTRAL)


def risk_label(level: str) -> str:
    """Etiqueta con ícono Material para no depender solo del color.

    Devuelve algo como ':material/dangerous: Muy Alto' (renderiza en Markdown).
    """
    return f"{RISK_ICONS.get(level, ':material/help:')} {level}"


def style_fig(
    fig: "go.Figure",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
    height: int | None = None,
    show_legend: bool | None = None,
) -> "go.Figure":
    """Aplica el estilo común a una figura Plotly.

    Deja que el tema de Streamlit (``theme="streamlit"`` en ``st.plotly_chart``)
    controle los colores de fondo/tipografía para que el gráfico se adapte solo
    a modo claro/oscuro: por eso fijamos fondos transparentes y NO forzamos el
    color de la fuente. Solo unificamos márgenes, título, ejes y leyenda.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=56 if title else 24, b=12),
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=18)) if title else None,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0.0, title_text=legend_title or "",
        ),
        hoverlabel=dict(font_size=12),
    )
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)
    if height is not None:
        fig.update_layout(height=height)
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    return fig
