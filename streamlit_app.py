import streamlit as st
import pandas as pd
import numpy as np
import time

# pyrefly: ignore [missing-import]
import joblib
from pathlib import Path
import streamlit_antd_components as sac

st.set_page_config(page_title="Sistema Predicción de Fallos", layout="centered")

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

        # Encode categorical features and extract numeric columns for model input
        X_encoded = preprocessor.encode_categorical(features.copy(), fit=True)
        X = X_encoded[[
            'vida_util_consumida',
            'tasa_incidencias_tecnicas',
            'tiempo_inactividad_acumulado',
            'costo_mto_reactivo_acumulado',
            'ubicacion_activo_encoded',
            'tipo_equipo_encoded'
        ]]

        # Encode target and extract only the encoded integer column
        y_encoded = preprocessor.encode_target(targets.copy()[["estado_integridad_hardware"]], fit=True)
        y_estado = y_encoded['estado_integridad_hardware_encoded'].values

        # Encode risk target
        y_riesgo_encoded = preprocessor.encode_risk_target(targets.copy()[["nivel_riesgo_operativo"]], fit=True)
        y_riesgo = y_riesgo_encoded['nivel_riesgo_operativo_encoded'].values

        # Combine targets using np.column_stack for multi-output training
        y_combined = np.column_stack((y_estado, y_riesgo))

        # Import train_test_split and metrics locally
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix

        # Split features and combined target labels using 80-20 partition with random state 0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_combined, test_size=0.20, random_state=0
        )

        trainer = ModelTrainer()
        # Train multi-output model on the training partition
        trainer.train_multioutput(X_train, y_train)

        # Predict predictions on the test partition (both estado and riesgo)
        y_pred = trainer.model.predict(X_test)

        # Calculate accuracy for estado (first column)
        accuracy = accuracy_score(y_test[:, 0], y_pred[:, 0])
        print(f"Precisión del modelo: {accuracy * 100:.2f}%")

        # Cálculo de la matriz y la precisión
        cm = confusion_matrix(y_test[:, 0], y_pred[:, 0])
        print(cm)

        # Save test metrics in Streamlit session state to display in UI
        st.session_state["test_accuracy"] = accuracy
        st.session_state["test_confusion_matrix"] = cm

        # Save model data including test evaluation results
        model_data = {
            "model_name": f"RandomForest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "model_path": f"models/trained_models/rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            "features": ",".join(preprocessor.get_feature_names()),
            "metrics_json": str({
                "test_accuracy": float(accuracy),
                "confusion_matrix": cm.tolist()
            }),
            "feature_importance_json": str(trainer.get_feature_importance())
        }
        save_trained_model(model_data)

        predictor = ModelPredictor(trainer.model, preprocessor)

        st.session_state["trainer"] = trainer
        st.session_state["preprocessor"] = preprocessor
        st.session_state["predictor"] = predictor
        st.session_state["model_trained"] = True

        return trainer

def render_login_page():
    authenticator = authenticate_user()
    authenticator.login(location='main')

    if st.session_state.get("authentication_status"):
        st.session_state["authenticated"] = True
        st.session_state["username"] = st.session_state.get("name")
        st.rerun()
    elif st.session_state.get("authentication_status") is False:
        sac.alert(label="Credenciales incorrectas. Verifique su usuario y contraseña.", color="error", icon=True)
    elif st.session_state.get("authentication_status") is None:
        pass

def render_prediction_form():
    st.subheader("Predicción de Estado de Equipo")

    vida_util = st.number_input("Vida Útil Consumida (%)", min_value=0.0, max_value=100.0, value=50.0)
    tasa_incidencias = st.number_input("Tasa Incidencias Técnicas", min_value=0, max_value=100, value=2)
    tiempo_inactividad = st.number_input("Tiempo Inactividad Acumulado (hrs)", min_value=0.0, max_value=1000.0, value=100.0)

    costo_mto = st.number_input("Costo Mantenimiento Reactivo ($)", min_value=0.0, max_value=10000.0, value=150.0)
    ubicacion = st.selectbox("Ubicación", get_locations())
    tipo_equipo = st.selectbox("Tipo de Equipo", get_equipment_types())

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_clicked = st.button("Predecir Estado", type="primary")

    if predict_clicked:
        if st.session_state["predictor"] is None:
            sac.alert(label="El modelo aún no ha sido entrenado", color="warning", icon=True)
            return

        input_dict = {
            "vida_util_consumida": vida_util,
            "tasa_incidencias_tecnicas": tasa_incidencias,
            "tiempo_inactividad_acumulado": tiempo_inactividad,
            "costo_mto_reactivo_acumulado": costo_mto,
            "ubicacion_activo": ubicacion,
            "tipo_equipo": tipo_equipo
        }

        estado, riesgo = st.session_state["predictor"].predict(input_dict)

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            sac.result(
                label="Estado de Integridad",
                description=estado,
                status="success" if estado == "Excelente" else "warning"
            )
        with result_col2:
            sac.result(
                label="Nivel de Riesgo",
                description=riesgo,
                status="error" if riesgo == "Alto" or riesgo == "Critico" else "warning"
            )

        alert_system = EarlyWarningSystem()
        equipo_id = f"PRED_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        alerts = alert_system.check_prediction(equipo_id, estado, riesgo, {"costo_mto": costo_mto})

        if alerts:
            sac.alert(label="Se han generado alertas para este equipo", color="warning", icon=True)

        prediction_data = {
            "equipo_id": equipo_id,
            "vida_util_consumida": vida_util,
            "tasa_incidencias_tecnicas": tasa_incidencias,
            "tiempo_inactividad_acumulado": tiempo_inactividad,
            "costo_mto_reactivo_acumulado": costo_mto,
            "ubicacion_activo": ubicacion,
            "tipo_equipo": tipo_equipo,
            "estado_integridad_predicted": estado,
            "nivel_riesgo_predicted": riesgo,
        }
        save_prediction(prediction_data)

        return estado

def render_import_export():
    st.subheader("Importar/Exportar Datos")

    tabs = sac.tabs(items=["Importar CSV", "Exportar Histórico"], align="left")

    if tabs == "Importar CSV":
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = import_csv(uploaded_file)
                st.success(f"CSV cargado: {len(df)} registros")

                if st.button("Guardar en Base de Datos", type="primary"):
                    count = save_to_database(df)
                    sac.alert(label=f"{count} registros guardados exitosamente", color="success", icon=True)
                    st.rerun()
            except Exception as e:
                sac.alert(label=f"Error al importar: {e}", color="error", icon=True)
    else:
        if st.button("Descargar CSV", type="primary"):
            try:
                df = get_historical_data()
                csv_bytes = export_to_csv(df)
                st.download_button(
                    label="Descargar datos completos",
                    data=csv_bytes,
                    file_name=f"historico_equipos_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                sac.alert(label="Descarga lista", color="success", icon=True)
            except Exception as e:
                sac.alert(label=f"Error al exportar: {e}", color="error", icon=True)

def render_dashboard(df):
    display_all_kpis(df)
    render_all_charts(df)

def render_model_section(df):
    st.subheader("Gestión del Modelo")

    tabs = sac.tabs(items=["Entrenar", "Feature Importance", "Métricas"], align="left")

    if tabs == "Entrenar":
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Entrenar Modelo Random Forest", type="primary"):
                train_model(df)
                st.rerun()

        if st.session_state["model_trained"]:
            sac.result(
                label="Modelo Entrenado",
                description="Random Forest listo para predicciones",
                status="success"
            )
        else:
            sac.result(
                label="Modelo No Entrenado",
                description="Entrene el modelo para comenzar",
                status="info"
            )

    elif tabs == "Feature Importance":
        if st.session_state["trainer"] is not None:
            viewer = FeatureImportanceViewer(st.session_state["trainer"])
            viewer.render()
        else:
            sac.alert(label="Entrena el modelo para ver la importancia de variables", color="info", closable=False)

    elif tabs == "Métricas":
        if st.session_state["model_trained"]:
            # Retrieve test set accuracy and confusion matrix from session state
            accuracy = st.session_state.get("test_accuracy")
            cm = st.session_state.get("test_confusion_matrix")
            
            metrics_data = {
                "status": "Modelo activo",
                "entrenado": True,
            }
            if accuracy is not None:
                metrics_data["Precisión del modelo (Test)"] = f"{accuracy * 100:.2f}%"
            st.json(metrics_data)
            
            # If the confusion matrix exists, display it as a labeled DataFrame
            if cm is not None:
                st.write("**Matriz de Confusión (Test)**")
                preprocessor = st.session_state.get("preprocessor")
                if preprocessor and hasattr(preprocessor, "target_encoder") and preprocessor.target_encoder:
                    classes = preprocessor.target_encoder.categories_[0]
                    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
                    st.dataframe(cm_df)
                else:
                    st.dataframe(pd.DataFrame(cm))
        else:
            sac.alert(label="No hay métricas disponibles", color="info", closable=False)

def render_monitoring_section(df):
    st.subheader("Monitoreo")

    tabs = sac.tabs(items=["Data Drift", "Matriz de Confusión"], align="left")

    if tabs == "Data Drift":
        if st.session_state["baseline_df"] is not None and not df.empty:
            detector = DataDriftDetector(st.session_state["baseline_df"])
            detector.render_ui(df)
        else:
            sac.alert(label="No hay datos de referencia para comparar", color="info", closable=False)

    elif tabs == "Matriz de Confusión":
        history = get_predictions_history(limit=100)
        if not history.empty:
            eval = ModelEvaluator(None, st.session_state.get("preprocessor"))
            cm_monitor = ConfusionMatrixMonitor(eval)
            cm_monitor.render_historical(history)
        else:
            sac.alert(label="No hay historial de predicciones", color="info", closable=False)

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

    page = st.sidebar.selectbox("Navegación", menu)

    if page == "Dashboard":
        if df.empty:
            sac.alert(label="No hay datos registrados. Importa un CSV para comenzar.", color="warning", closable=False)
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
            sac.alert(label="No hay datos para entrenar el modelo", color="warning", closable=False)
        else:
            render_model_section(df)

    elif page == "Monitoreo":
        render_monitoring_section(df)

    elif page == "Alertas":
        render_alerts_section()

def main():
    init_session()
    # Initialize authenticator in session state to persist across reruns
    if "authenticator" not in st.session_state:
        st.session_state["authenticator"] = authenticate_user()
    
    authenticator = st.session_state["authenticator"]
    # Render login widget if the user is not authenticated
    if not st.session_state.get("authentication_status"):
        try:
            authenticator.login(location='main')
        except Exception as e:
            st.error(f"Error al iniciar sesión: {e}")
        
        # Check authentication status after login attempt or cookie check
        if st.session_state.get("authentication_status"):
            time.sleep(0.5)
            st.rerun()
        elif st.session_state.get("authentication_status") is False:
            sac.alert(label="Credenciales incorrectas. Verifique su usuario y contraseña.", color="error", icon=True)
    else:
        # User is authenticated
        authenticator.logout(location='sidebar', button_name='Cerrar Sesión')
        main_app()

if __name__ == "__main__":
    main()
