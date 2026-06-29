import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from features.config import get_risk_levels

HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]
LOW_RISK_LEVELS = ["Muy Bajo", "Bajo"]


class ConfusionMatrixMonitor:
    """Renders confusion matrices for the operational-risk model."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def render(self, predictions_df, actuals_col="operational_risk_level", pred_col="predicted_risk_level"):
        st.subheader("Matriz de Confusión")

        if actuals_col not in predictions_df.columns or pred_col not in predictions_df.columns:
            st.info(
                "No hay datos suficientes para mostrar la matriz de confusión. "
                "Se requieren predicciones con su valor real para comparar."
            )
            return

        labels = get_risk_levels()
        y_true = predictions_df[actuals_col]
        y_pred = predictions_df[pred_col]

        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)

        fig = px.imshow(
            cm_df,
            labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=labels,
            y=labels,
            color_continuous_scale="Reds",
            text_auto=True,
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        false_positives = self._analyze_false_positives(predictions_df, actuals_col, pred_col)
        if false_positives > 0:
            st.warning(
                f"Se detectaron {false_positives} falsos positivos "
                f"(predicho Alto/Muy Alto, real Muy Bajo/Bajo)"
            )

    def _analyze_false_positives(self, df, actuals_col, pred_col):
        if pred_col in df.columns and actuals_col in df.columns:
            fp = df[
                df[pred_col].isin(HIGH_RISK_LEVELS) & df[actuals_col].isin(LOW_RISK_LEVELS)
            ]
            return len(fp)
        return 0

    def render_historical(self, history_df):
        st.subheader("Rendimiento Histórico del Modelo")

        if history_df.empty:
            st.info("No hay historial de predicciones disponible.")
            return

        history_df = history_df.copy()
        history_df["date"] = pd.to_datetime(history_df["prediction_at"]).dt.date
        daily_stats = history_df.groupby("date").size().reset_index(name="predictions")

        fig = px.line(daily_stats, x="date", y="predictions", title="Predicciones por Día")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(history_df.tail(20), use_container_width=True)
