import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
import pandas as pd

from features.theme import style_fig

TOP_N = 15


class FeatureImportanceViewer:
    def __init__(self, trainer):
        self.trainer = trainer

    def render(self):
        st.subheader("Análisis de Importancia de Variables")

        importance_dict = self.trainer.get_feature_importance()

        if not importance_dict:
            st.info("El modelo aún no ha sido entrenado.", icon=":material/info:")
            return

        importance_df = pd.DataFrame({
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values()),
        }).sort_values("Importance", ascending=False)

        st.caption(
            f"Top {min(TOP_N, len(importance_df))} variables con mayor peso en las "
            "predicciones del Random Forest."
        )

        top = importance_df.head(TOP_N).sort_values("Importance", ascending=True)
        fig = px.bar(
            top,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            text=top["Importance"].map(lambda v: f"{v:.3f}"),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_coloraxes(showscale=False)
        style_fig(
            fig,
            xaxis_title="Importancia relativa",
            yaxis_title=None,
            height=max(360, 26 * len(top) + 80),
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Ver tabla completa de importancia"):
            st.dataframe(
                importance_df.reset_index(drop=True),
                width="stretch",
                hide_index=True,
            )

    def render_comparison(self, old_importance, new_importance):
        st.subheader("Comparación de Importancia de Variables (Pre vs Post Reentrenamiento)")

        features = list(set(list(old_importance.keys()) + list(new_importance.keys())))
        comparison_df = pd.DataFrame({
            "Feature": features,
            "Antes": [old_importance.get(f, 0) for f in features],
            "Después": [new_importance.get(f, 0) for f in features]
        })

        comparison_df["Cambio"] = comparison_df["Después"] - comparison_df["Antes"]
        comparison_df = comparison_df.sort_values("Cambio", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Antes del Reentrenamiento**")
            st.dataframe(pd.DataFrame({"Feature": list(old_importance.keys()), "Importance": list(old_importance.values())}))
        with col2:
            st.write("**Después del Reentrenamiento**")
            st.dataframe(pd.DataFrame({"Feature": list(new_importance.keys()), "Importance": list(new_importance.values())}))

        st.write("**Cambios (Después - Antes)**")
        st.dataframe(comparison_df)
