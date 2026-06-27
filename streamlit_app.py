import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Sistema Predicción de Fallos", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "trainer" not in st.session_state:
    st.session_state["trainer"] = None
if "preprocessor" not in st.session_state:
    st.session_state["preprocessor"] = None
if "predictor" not in st.session_state:
    st.session_state["predictor"] = None
if "baseline_df" not in st.session_state:
    st.session_state["baseline_df"] = None

from features.auth.login import authenticate_user, require_auth
from features.config import load_config, get_locations, get_hardware_states, get_equipment_types
from features.database import init_db, get_all_equipos, save_prediction, get_predictions_history, get_active_model, save_trained_model
from features.data import DataLoader, Preprocessor, import_csv, export_to_csv, save_to_database, get_historical_data
from features.model import ModelTrainer, ModelPredictor, ModelEvaluator
from features.dashboard import display_all_kpis, render_all_charts, FilterManager
from features.monitoring import DataDriftDetector, ConfusionMatrixMonitor
from features.alerts import EarlyWarningSystem, FeatureImportanceViewer

def init_session():
    try:
        init_db()
    except Exception as e:
        st.warning(f"Base de datos no disponible: {e}")

def load_initial_data():
    try:
        df = get_all_equipos()
        if not df.empty:
            st.session_state["baseline_df"] = df
        return df
    except Exception as e:
        st.warning(f"No se pudieron cargar datos: {e}")
        return pd.DataFrame()

def train_model(df):
    with st.spinner("Entrenando modelo..."):
        loader = DataLoader()
        preprocessor = Preprocessor()

        features = loader.get_features(df)
        targets = loader.get_targets(df)

        X = preprocessor.encode_categorical(features.copy(), fit=True)
        y = preprocessor.encode_target(targets.copy()[["Estado_Integridad_Hardware"]], fit=True)

        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(X, y, k=5)

        trainer.train(X, y)

        model_data = {
            "model_name": f"RandomForest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "model_path": f"models/trained_models/rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            "features": ",".join(preprocessor.get_feature_names()),
            "metrics_json": str(cv_results),
            "feature_importance_json": str(trainer.get_feature_importance())
        }
        save_trained_model(model_data)

        predictor = ModelPredictor(trainer.model, preprocessor)

        st.session_state["trainer"] = trainer
        st.session_state["preprocessor"] = preprocessor
        st.session_state["predictor"] = predictor
        st.session_state["model_trained"] = True

        return trainer, cv_results

def render_login_page():
    st.title("🔐 Iniciar Sesión")
    st.markdown("---")

    authenticator = authenticate_user()
    authenticator.login(location='main')

    if st.session_state.get("authentication_status"):
        st.session_state["authenticated"] = True
        st.session_state["username"] = st.session_state.get("name")
        st.rerun()
    elif st.session_state.get("authentication_status") is False:
        st.error("Credenciales incorrectas")
    elif st.session_state.get("authentication_status") is None:
        pass

def render_prediction_form():
    st.subheader("Predicción de Estado de Equipo")

    col1, col2 = st.columns(2)

    with col1:
        vida_util = st.number_input("Vida Útil Consumida (%)", min_value=0.0, max_value=100.0, value=50.0)
        tasa_incidencias = st.number_input("Tasa Incidencias Técnicas", min_value=0, max_value=100, value=2)
        tiempo_inactividad = st.number_input("Tiempo Inactividad Acumulado (hrs)", min_value=0.0, max_value=1000.0, value=100.0)

    with col2:
        costo_mto = st.number_input("Costo Mantenimiento Reactivo ($)", min_value=0.0, max_value=10000.0, value=150.0)
        ubicacion = st.selectbox("Ubicación", get_locations())
        tipo_equipo = st.selectbox("Tipo de Equipo", get_equipment_types())

    if st.button("Predecir Estado", type="primary"):
        if st.session_state["predictor"] is None:
            st.error("El modelo aún no ha sido entrenado")
            return

        input_dict = {
            "Vida_Util_Consumida": vida_util,
            "Tasa_Incidencias_Tecnicas": tasa_incidencias,
            "Tiempo_Inactividad_Acumulado": tiempo_inactividad,
            "Costo_Mto_Reactivo_Acumulado": costo_mto,
            "Ubicacion_Activo": ubicacion,
            "Tipo_Equipo": tipo_equipo
        }

        estado, riesgo = st.session_state["predictor"].predict(input_dict)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Estado de Integridad:** {estado}")
        with col2:
            st.warning(f"**Nivel de Riesgo:** {riesgo}")

        alert_system = EarlyWarningSystem()
        equipo_id = f"PRED_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        alerts = alert_system.check_prediction(equipo_id, estado, riesgo, {"costo_mto": costo_mto})

        if alerts:
            st.warning("Se han generado alertas para este equipo")

        prediction_data = {
            "equipo_id": equipo_id,
            "vida_util_consumida": vida_util,
            "tasa_incidencias_tecnicas": tasa_incidencias,
            "tiempo_inactividad_acumulado": tiempo_inactividad,
            "costo_mto_reactivo_acumulado": costo_mto,
            "ubicacion_activo": ubicacion,
            "tipo_equipo": tipo_equipo,
            "estado_integridad_predicted": estado,
            "nivel_riesgo_predicted": riesgo
        }
        save_prediction(prediction_data)

        return estado, riesgo

def render_import_export():
    st.subheader("Importar/Exportar Datos")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Importar CSV**")
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = import_csv(uploaded_file)
                st.success(f"CSV cargado: {len(df)} registros")

                if st.button("💾 Guardar en Base de Datos"):
                    count = save_to_database(df)
                    st.success(f"{count} registros guardados exitosamente")
                    st.rerun()
            except Exception as e:
                st.error(f"Error al importar: {e}")

    with col2:
        st.write("**Exportar Histórico**")
        if st.button("Descargar CSV"):
            try:
                df = get_historical_data()
                csv_bytes = export_to_csv(df)
                st.download_button(
                    label="Descargar datos completos",
                    data=csv_bytes,
                    file_name=f"historico_equipos_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error al exportar: {e}")

def render_dashboard(df):
    display_all_kpis(df)
    render_all_charts(df)

def render_model_section(df):
    st.subheader("Gestión del Modelo")

    tab1, tab2, tab3 = st.tabs(["Entrenar", "Feature Importance", "Métricas"])

    with tab1:
        if st.button("Entrenar Modelo Random Forest", type="primary"):
            train_model(df)

        if st.session_state["model_trained"]:
            st.success("✓ Modelo entrenado y listo")

    with tab2:
        if st.session_state["trainer"] is not None:
            viewer = FeatureImportanceViewer(st.session_state["trainer"])
            viewer.render()
        else:
            st.info("Entrena el modelo para ver la importancia de variables")

    with tab3:
        if st.session_state["model_trained"]:
            st.json({"status": "Modelo activo", "entrenado": True})
        else:
            st.info("No hay métricas disponibles")

def render_monitoring_section(df):
    st.subheader("Monitoreo")

    tab1, tab2 = st.tabs(["Data Drift", "Matriz de Confusión"])

    with tab1:
        if st.session_state["baseline_df"] is not None and not df.empty:
            detector = DataDriftDetector(st.session_state["baseline_df"])
            detector.render_ui(df)
        else:
            st.info("No hay datos de referencia para comparar")

    with tab2:
        history = get_predictions_history(limit=100)
        if not history.empty:
            eval = ModelEvaluator(None, st.session_state.get("preprocessor"))
            cm_monitor = ConfusionMatrixMonitor(eval)
            cm_monitor.render_historical(history)
        else:
            st.info("No hay historial de predicciones")

def render_alerts_section():
    st.subheader("Alertas Tempranas")
    alert_system = EarlyWarningSystem()
    alert_system.render_alerts_panel()

def main_app():
    st.title("Sistema de Predicción de Fallos en Equipos Electrónicos")
    st.markdown("---")

    init_session()
    df = load_initial_data()

    if not df.empty and st.session_state["baseline_df"] is None:
        st.session_state["baseline_df"] = df

    menu = ["Dashboard", "Predicción", "Importar/Exportar", "Modelo", "Monitoreo", "Alertas"]

    if st.session_state.get("username"):
        menu.append(f"👤 {st.session_state['username']}")

    page = st.sidebar.selectbox("Navegación", menu)

    if page == "Dashboard":
        if df.empty:
            st.warning("No hay datos registrados. Importa un CSV para comenzar.")
        else:
            filter_mgr = FilterManager(df)
            filtered_df = filter_mgr.render_ui()
            render_dashboard(filtered_df)

    elif page == "Predicción":
        render_prediction_form()

    elif page == "Importar/Exportar":
        render_import_export()

    elif page == "Modelo":
        if df.empty:
            st.warning("No hay datos para entrenar el modelo")
        else:
            render_model_section(df)

    elif page == "Monitoreo":
        render_monitoring_section(df)

    elif page == "Alertas":
        render_alerts_section()

    elif "👤" in page:
        st.write(f"Bienvenido, {st.session_state['username']}")
        if st.button("Cerrar Sesión"):
            st.session_state["authenticated"] = False
            st.rerun()

def main():
    init_session()

    if not st.session_state["authenticated"]:
        render_login_page()
    else:
        authenticator = authenticate_user()
        authenticator.logout(location='sidebar', button_name='Cerrar Sesión')
        main_app()

if __name__ == "__main__":
    main()
