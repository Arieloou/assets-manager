import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.express as px
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
from features.config import load_config, get_locations, get_hardware_states, get_device_types, get_brands, get_risk_levels
from features.database import init_db, get_all_devices, save_prediction, get_predictions_history, save_trained_model
from features.data import DataLoader, Preprocessor, import_csv, export_to_csv, save_to_database, get_historical_data
from features.model import ModelTrainer, ModelPredictor, ModelEvaluator
from features.dashboard import display_all_kpis, render_all_charts, render_correlation_matrix, FilterManager
from features.monitoring import DataDriftDetector, ConfusionMatrixMonitor
from features.alerts import EarlyWarningSystem, FeatureImportanceViewer

def init_session():
    try:
        init_db()
    except Exception as e:
        st.warning(f"Base de datos no disponible: {e}")

def load_initial_data():
    try:
        df = get_all_devices()
        if not df.empty:
            st.session_state["baseline_df"] = df
        return df
    except Exception as e:
        st.warning(f"No se pudieron cargar datos: {e}")
        return pd.DataFrame()

def floating_progress(placeholder, pct, message):
    """Render a floating progress notification (fixed position) with a percentage."""
    pct = max(0, min(100, int(round(pct))))
    placeholder.markdown(
        f"""
        <div style="position: fixed; top: 4.5rem; right: 1.5rem; z-index: 1000000;
                    background: rgba(17,24,39,0.96); color: #f9fafb; padding: 14px 18px;
                    border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
                    min-width: 270px; font-family: 'Source Sans Pro', sans-serif;">
            <div style="display:flex; align-items:center; gap:8px; font-size:0.9rem; margin-bottom:10px;">
                <span>{message}</span>
            </div>
            <div style="background:#374151; border-radius:8px; overflow:hidden; height:10px;">
                <div style="width:{pct}%; height:100%; transition: width .25s ease;
                            background:linear-gradient(90deg,#10b981,#34d399);"></div>
            </div>
            <div style="text-align:right; font-size:0.8rem; margin-top:6px; color:#9ca3af;">{pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def train_model(df):
    from sklearn.model_selection import train_test_split

    progress = st.empty()
    try:
        floating_progress(progress, 5, "Cargando datos del dataset...")
        loader = DataLoader()
        preprocessor = Preprocessor()

        features = loader.get_features(df)
        targets = loader.get_targets(df)

        # Split the RAW data first so the scaler/encoders are fit on train only
        # (avoids leaking test statistics into the StandardScaler).
        floating_progress(progress, 20, "Particionando datos (entrenamiento/prueba)...")
        raw_train, raw_test, target_train, target_test = train_test_split(
            features, targets, test_size=0.20, random_state=0,
            stratify=targets[loader.TARGET_COLUMNS[0]],
        )

        # Build scaled/encoded feature matrices and encode the target
        floating_progress(progress, 40, "Procesando features (escalado y codificación)...")
        X_train = preprocessor.build_features(raw_train, fit=True)
        X_test = preprocessor.build_features(raw_test, fit=False)
        y_train = preprocessor.encode_target(target_train, fit=True)
        y_test = preprocessor.encode_target(target_test, fit=False)

        floating_progress(progress, 65, "Optimizando hiperparámetros (RandomizedSearchCV)...")
        trainer = ModelTrainer()
        trainer.tune_hyperparameters(X_train, y_train)

        # Evaluate on the held-out test partition, in original label space
        floating_progress(progress, 85, "Evaluando el modelo...")
        y_pred = trainer.predict(X_test)
        evaluator = ModelEvaluator(trainer, preprocessor)
        y_test_labels = preprocessor.decode_target(y_test)
        y_pred_labels = preprocessor.decode_target(y_pred)

        summary = evaluator.summary_metrics(y_test_labels, y_pred_labels)
        report = evaluator.classification_report(y_test_labels, y_pred_labels)

        print(f"Precisión del modelo: {summary['accuracy'] * 100:.2f}%")
        print(f"Mejores hiperparámetros: {trainer.best_params_}")

        # Persist evaluation results in session state for the UI
        st.session_state["test_metrics"] = summary
        st.session_state["test_report"] = report
        st.session_state["best_params"] = trainer.best_params_
        st.session_state["best_cv_score"] = trainer.best_cv_score_

        floating_progress(progress, 95, "Guardando el modelo...")
        model_data = {
            "model_name": f"RandomForest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "model_path": f"models/trained_models/rf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            "features": ",".join(preprocessor.get_feature_names()),
            "metrics_json": str({
                "test_accuracy": summary["accuracy"],
                "best_params": trainer.best_params_,
                "best_cv_score": trainer.best_cv_score_,
            }),
            "feature_importance_json": str(trainer.get_feature_importance()),
        }
        try:
            save_trained_model(model_data)
        except Exception as e:
            st.warning(f"No se pudo registrar el modelo en la BD: {e}")

        predictor = ModelPredictor(trainer, preprocessor)

        st.session_state["trainer"] = trainer
        st.session_state["preprocessor"] = preprocessor
        st.session_state["predictor"] = predictor
        st.session_state["model_trained"] = True

        floating_progress(progress, 100, "¡Procesamiento completado!")
        time.sleep(0.5)
    finally:
        progress.empty()

    st.toast("Modelo entrenado correctamente")
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
    st.subheader("Predicción de Nivel de Riesgo Operativo")

    from datetime import date
    import json

    col1, col2 = st.columns(2)
    with col1:
        device_brand = st.selectbox("Marca", get_brands())
        device_type = st.selectbox("Tipo de Equipo", get_device_types())
        acquisition_date = st.date_input(
            "Fecha de Adquisición", value=date(2021, 1, 1), max_value=date.today()
        )
        technical_incident_rate = st.number_input(
            "Tasa Incidencias Técnicas", min_value=0, max_value=100, value=2
        )
    with col2:
        headquarters_location = st.selectbox("Ubicación", get_locations())
        hardware_integrity_status = st.selectbox("Estado de Integridad de Hardware", get_hardware_states())
        no_corrective = st.checkbox("Sin mantenimiento correctivo registrado", value=False)
        last_reactive_maintenance_date = None
        if not no_corrective:
            last_reactive_maintenance_date = st.date_input(
                "Fecha Último Mto. Correctivo", value=date(2025, 1, 1), max_value=date.today()
            )
        last_preventive_maintenance_date = st.date_input(
            "Fecha Último Mto. Preventivo", value=date(2024, 1, 1), max_value=date.today()
        )

    predict_clicked = st.button("Predecir Nivel de Riesgo", type="primary")

    if predict_clicked:
        if st.session_state["predictor"] is None:
            sac.alert(label="El modelo aún no ha sido entrenado", color="warning", icon=True)
            return

        input_dict = {
            "device_brand": device_brand,
            "device_type": device_type,
            "hardware_integrity_status": hardware_integrity_status,
            "headquarters_location": headquarters_location,
            "acquisition_date": pd.Timestamp(acquisition_date),
            "technical_incident_rate": technical_incident_rate,
            "last_reactive_maintenance_date": (
                pd.Timestamp(last_reactive_maintenance_date) if last_reactive_maintenance_date else pd.NaT
            ),
            "last_preventive_maintenance_date": pd.Timestamp(last_preventive_maintenance_date),
        }

        predictor = st.session_state["predictor"]
        risk_level = predictor.predict(input_dict)
        confidence = predictor.predict_proba(input_dict)

        status = "error" if risk_level in ("Alto", "Muy Alto") else (
            "warning" if risk_level == "Medio" else "success"
        )
        sac.result(label="Nivel de Riesgo Operativo Predicho", description=risk_level, status=status)

        # Soft output: vote proportion per risk level (confidence)
        st.markdown("**Confianza de la predicción (proporción de votos)**")
        cols = st.columns(len(confidence))
        for col, (level, prop) in zip(cols, confidence.items()):
            with col:
                st.metric(level, f"{prop * 100:.1f}%")

        # Derive day-difference features so they can be stored with the prediction
        engineered = Preprocessor().engineer_features(pd.DataFrame([input_dict]))

        alert_system = EarlyWarningSystem()
        device_id = f"PRED_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            alerts = alert_system.check_prediction(
                device_id, risk_level, {"device_type": device_type, "device_brand": device_brand}
            )
            if alerts:
                sac.alert(label="Se han generado alertas para este equipo", color="warning", icon=True)
        except Exception as e:
            st.warning(f"No se pudieron registrar alertas: {e}")

        prediction_data = {
            "device_id": device_id,
            "device_brand": device_brand,
            "acquisition_date": acquisition_date,
            "technical_incident_rate": technical_incident_rate,
            "days_since_reactive_maintenance": int(engineered["days_since_last_corrective_maintenance"].iloc[0]),
            "days_since_preventive_maintenance": int(engineered["days_since_last_preventive_maintenance"].iloc[0]),
            "headquarters_location": headquarters_location,
            "hardware_integrity_status": hardware_integrity_status,
            "device_type": device_type,
            "predicted_risk_level": risk_level,
            "confidence_json": json.dumps(confidence),
        }
        try:
            save_prediction(prediction_data)
        except Exception as e:
            st.warning(f"No se pudo guardar la predicción: {e}")

        return risk_level

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
                    progress = st.empty()
                    last_pct = {"value": -1}

                    def _save_progress(done, total):
                        pct = int(done / total * 100) if total else 100
                        # Throttle: only re-render the overlay when the percent changes
                        if pct != last_pct["value"]:
                            last_pct["value"] = pct
                            floating_progress(
                                progress, pct, f"Guardando en base de datos... ({done}/{total})"
                            )

                    try:
                        floating_progress(progress, 0, "Guardando en base de datos...")
                        count = save_to_database(df, progress_callback=_save_progress)
                        floating_progress(progress, 100, "¡Guardado completado!")
                        time.sleep(0.4)
                    finally:
                        progress.empty()

                    st.toast(f"{count} registros guardados", icon="✅")
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
    # Derive day-based features (useful_life_consumed_days, etc.) for KPIs/charts
    if not df.empty and "acquisition_date" in df.columns:
        try:
            df = Preprocessor().engineer_features(df)
        except Exception:
            pass
    display_all_kpis(df)
    render_all_charts(df)

def render_model_section(df):
    st.subheader("Gestión del Modelo")

    tabs = sac.tabs(items=["Entrenar", "Feature Importance", "Métricas", "Correlación"], align="left")

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
            summary = st.session_state.get("test_metrics")

            if summary is not None:
                st.markdown("**Métricas globales (partición de prueba)**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{summary['accuracy'] * 100:.2f}%")
                c2.metric("Precision (macro)", f"{summary['precision_macro'] * 100:.2f}%")
                c3.metric("Recall (macro)", f"{summary['recall_macro'] * 100:.2f}%")
                c4.metric("F1-Score (macro)", f"{summary['f1_macro'] * 100:.2f}%")

                c5, c6, c7 = st.columns(3)
                c5.metric("Precision (weighted)", f"{summary['precision_weighted'] * 100:.2f}%")
                c6.metric("Recall (weighted)", f"{summary['recall_weighted'] * 100:.2f}%")
                c7.metric("F1-Score (weighted)", f"{summary['f1_weighted'] * 100:.2f}%")

            best_params = st.session_state.get("best_params")
            best_cv_score = st.session_state.get("best_cv_score")
            if best_params is not None:
                st.markdown("---")
                st.markdown("**Mejores Hiperparámetros (RandomizedSearchCV)**")
                if best_cv_score is not None:
                    st.metric("F1-Score (macro, validación cruzada)", f"{best_cv_score * 100:.2f}%")
                st.json(best_params)

        else:
            sac.alert(label="No hay métricas disponibles", color="info", closable=False)

    elif tabs == "Correlación":
        if df.empty:
            sac.alert(label="No hay datos para calcular la correlación", color="info", closable=False)
        else:
            render_correlation_matrix(df)

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

        # Render the dynamic confusion matrix on the current database data if the model is trained
        if st.session_state.get("model_trained") and st.session_state.get("predictor") is not None:
            if not df.empty:
                try:
                    df_with_pred = df.copy()
                    predictor = st.session_state["predictor"]

                    # Predict risk levels using the trained model
                    X = predictor.preprocessor.build_features(df_with_pred, fit=False)
                    encoded_preds = predictor.model.predict(X)
                    df_with_pred['predicted_risk_level'] = predictor.preprocessor.decode_target(encoded_preds)

                    # Initialize the evaluator and confusion matrix monitor
                    eval_obj = ModelEvaluator(None, st.session_state.get("preprocessor"))
                    cm_monitor = ConfusionMatrixMonitor(eval_obj)
                    cm_monitor.render(df_with_pred)
                except Exception as e:
                    st.error(f"Error al calcular la matriz de confusión dinámica: {e}")
            else:
                sac.alert(label="No hay datos cargados para generar la matriz de confusión dinámica.", color="warning", closable=False)
        else:
            sac.alert(label="El modelo no está entrenado. Por favor, entrene el modelo en la sección 'Modelo' para ver la matriz de confusión dinámica.", color="info", closable=False)

        st.markdown("---")

        # Render historical predictions tracking if data exists
        if not history.empty:
            eval_obj = ModelEvaluator(None, st.session_state.get("preprocessor"))
            cm_monitor = ConfusionMatrixMonitor(eval_obj)
            cm_monitor.render_historical(history)
        else:
            sac.alert(label="No hay historial de predicciones guardado.", color="info", closable=False)

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

    # Render the sidebar navigation menu using Streamlit Ant Design components
    with st.sidebar:
        page = sac.menu(
            items=[
                sac.MenuItem("Dashboard", icon="speedometer2"),
                sac.MenuItem("Predicción", icon="cpu"),
                sac.MenuItem("Importar/Exportar", icon="arrow-left-right"),
                sac.MenuItem("Modelo", icon="gear"),
                sac.MenuItem("Monitoreo", icon="activity"),
                sac.MenuItem("Alertas", icon="bell"),
            ],
            index=0,
            variant='light',
            key="navigation_menu"
        )

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
        # Render main application contents first
        main_app()
        # Add a visual separator in the sidebar
        st.sidebar.divider()
        # Render logout button below the menu navigation
        authenticator.logout(location='sidebar')

if __name__ == "__main__":
    main()
