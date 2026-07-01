import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from features.config import get_risk_levels
from features.theme import RISK_COLORS, style_fig

HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]
LOW_RISK_LEVELS = ["Muy Bajo", "Bajo"]


class ConfusionMatrixMonitor:
    """Renders confusion matrices for the operational-risk model."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def _heatmap(self, cm, labels):
        """Annotated heatmap: color = % por fila (recall), texto = conteo + %.

        El color se normaliza por fila (cada fila = casos reales de una clase),
        de modo que la diagonal intensa significa buen acierto por clase.
        """
        cm = np.asarray(cm, dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        # Texto por celda: "conteo\n(porcentaje de la fila)".
        text = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text[i, j] = f"{int(cm[i, j])}<br>{row_pct[i, j] * 100:.0f}%"

        fig = px.imshow(
            row_pct,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
            aspect="auto",
            labels=dict(x="Predicho", y="Real", color="% por clase real"),
        )
        fig.update_traces(
            text=text,
            texttemplate="%{text}",
            hovertemplate="Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>"
                          "%{customdata} casos<extra></extra>",
            customdata=cm.astype(int),
        )
        fig.update_coloraxes(colorbar_title="% por<br>clase real")

        # Resaltar las filas de alto riesgo (no detectarlas es lo más costoso).
        n = len(labels)
        for level in HIGH_RISK_LEVELS:
            if level in labels:
                r = labels.index(level)
                fig.add_shape(
                    type="rect",
                    x0=-0.5, x1=n - 0.5, y0=r - 0.5, y1=r + 0.5,
                    line=dict(color=RISK_COLORS.get(level, "#DC2626"), width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                )
        style_fig(
            fig,
            xaxis_title="Nivel de riesgo PREDICHO",
            yaxis_title="Nivel de riesgo REAL",
            height=520,
        )
        return fig

    def render(self, predictions_df, actuals_col="operational_risk_level", pred_col="predicted_risk_level"):
        st.subheader("Matriz de Confusión (datos actuales)")

        if actuals_col not in predictions_df.columns or pred_col not in predictions_df.columns:
            st.info(
                "No hay datos suficientes para mostrar la matriz de confusión. "
                "Se requieren predicciones con su valor real para comparar.",
                icon=":material/info:",
            )
            return

        labels = get_risk_levels()
        y_true = predictions_df[actuals_col]
        y_pred = predictions_df[pred_col]

        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

        st.caption(
            "Color = porcentaje por **clase real** (fila). La diagonal marcada en rojo "
            "destaca las clases de alto riesgo: no detectarlas es el error más costoso."
        )
        st.plotly_chart(self._heatmap(cm, labels), width="stretch")

        # Métricas de seguridad orientadas al alto riesgo.
        missed = self._missed_high_risk(predictions_df, actuals_col, pred_col)
        false_alarms = self._false_alarms(predictions_df, actuals_col, pred_col)
        c1, c2 = st.columns(2)
        c1.metric(
            "Alto riesgo NO detectado",
            missed,
            help="Equipos realmente de Alto/Muy Alto que el modelo predijo como Bajo/Muy Bajo. "
                 "Es el error más grave (falsos negativos).",
        )
        c2.metric(
            "Falsas alarmas",
            false_alarms,
            help="Equipos de bajo riesgo real predichos como Alto/Muy Alto (falsos positivos).",
        )
        if missed > 0:
            st.error(
                f"{missed} equipo(s) de alto riesgo no fueron detectados. "
                "Revise la sensibilidad (recall) del modelo en las clases críticas.",
                icon=":material/crisis_alert:",
            )

        with st.expander("Ver matriz de conteos"):
            st.dataframe(
                pd.DataFrame(cm, index=labels, columns=labels),
                width="stretch",
            )

    def _missed_high_risk(self, df, actuals_col, pred_col):
        """Real alto riesgo predicho como bajo riesgo (falsos negativos críticos)."""
        if pred_col in df.columns and actuals_col in df.columns:
            return int(len(df[
                df[actuals_col].isin(HIGH_RISK_LEVELS) & df[pred_col].isin(LOW_RISK_LEVELS)
            ]))
        return 0

    def _false_alarms(self, df, actuals_col, pred_col):
        """Real bajo riesgo predicho como alto riesgo (falsos positivos)."""
        if pred_col in df.columns and actuals_col in df.columns:
            return int(len(df[
                df[pred_col].isin(HIGH_RISK_LEVELS) & df[actuals_col].isin(LOW_RISK_LEVELS)
            ]))
        return 0

    def render_historical(self, history_df):
        st.subheader("Rendimiento Histórico del Modelo")

        if history_df.empty:
            st.info("No hay historial de predicciones disponible.", icon=":material/info:")
            return

        history_df = history_df.copy()
        history_df["date"] = pd.to_datetime(history_df["prediction_at"]).dt.date
        daily_stats = history_df.groupby("date").size().reset_index(name="predictions")

        fig = px.line(daily_stats, x="date", y="predictions", markers=True)
        fig.update_traces(line_color="#2563EB", hovertemplate="%{x}<br>%{y} predicciones<extra></extra>")
        style_fig(
            fig,
            title="Predicciones por Día",
            xaxis_title="Fecha",
            yaxis_title="N.º de predicciones",
            height=360,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Ver últimas predicciones registradas"):
            st.dataframe(history_df.tail(20), width="stretch")
