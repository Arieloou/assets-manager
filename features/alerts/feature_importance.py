import streamlit as st
import plotly.express as px
import pandas as pd

class FeatureImportanceViewer:
    def __init__(self, trainer):
        self.trainer = trainer

    def render(self):
        st.subheader("Análisis de Importancia de Variables")

        importance_dict = self.trainer.get_feature_importance()

        if not importance_dict:
            st.info("El modelo aún no ha sido entrenado.")
            return

        importance_df = pd.DataFrame({
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values())
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance - Random Forest",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df.sort_values("Importance", ascending=False), use_container_width=True)

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
