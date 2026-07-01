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

st.set_page_config(
    page_title="Sistema Predicción de Fallos",
    page_icon=":material/build:",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
from features.config import get_locations, get_hardware_states, get_device_types, get_brands, get_risk_levels
from features.database import init_db, get_all_devices, save_prediction, get_predictions_history, save_trained_model
from features.data import DataLoader, Preprocessor, import_csv, save_to_database
from features.model import ModelTrainer, ModelPredictor, ModelEvaluator
from features.dashboard import display_all_kpis, render_all_charts, render_correlation_matrix, FilterManager
from features.monitoring import DataDriftDetector, ConfusionMatrixMonitor
from features.alerts import EarlyWarningSystem, FeatureImportanceViewer
from features.theme import RISK_ICONS, RISK_COLORS, NEUTRAL, style_fig

def init_session():
    try:
        init_db()
    except Exception as e:
        st.warning(f"Base de datos no disponible: {e}")

@st.cache_data(show_spinner=False)
def fetch_all_devices():
    """Lectura cacheada de equipos desde la BD.

    Evita golpear la base de datos en cada rerun. La caché se invalida
    explícitamente (``fetch_all_devices.clear()``) tras importar un CSV.
    """
    return get_all_devices()

def load_initial_data():
    try:
        df = fetch_all_devices()
        if not df.empty:
            st.session_state["baseline_df"] = df
        return df
    except Exception as e:
        st.warning(f"No se pudieron cargar datos: {e}")
        return pd.DataFrame()

def train_model(df):
    from sklearn.model_selection import train_test_split

    with st.status("Entrenando modelo Random Forest…", expanded=True) as status:
        prog = st.progress(5, text="Cargando datos del dataset…")
        loader = DataLoader()
        preprocessor = Preprocessor()

        features = loader.get_features(df)
        targets = loader.get_targets(df)

        # Split the RAW data first so the scaler/encoders are fit on train only
        # (avoids leaking test statistics into the StandardScaler).
        prog.progress(20, text="Particionando datos (entrenamiento/prueba)…")
        raw_train, raw_test, target_train, target_test = train_test_split(
            features, targets, test_size=0.20, random_state=0,
            stratify=targets[loader.TARGET_COLUMNS[0]],
        )

        # Build scaled/encoded feature matrices and encode the target
        prog.progress(40, text="Procesando features (escalado y codificación)…")
        X_train = preprocessor.build_features(raw_train, fit=True)
        X_test = preprocessor.build_features(raw_test, fit=False)
        y_train = preprocessor.encode_target(target_train, fit=True)
        y_test = preprocessor.encode_target(target_test, fit=False)

        prog.progress(65, text="Optimizando hiperparámetros (RandomizedSearchCV)…")
        trainer = ModelTrainer()
        trainer.tune_hyperparameters(X_train, y_train)

        # Evaluate on the held-out test partition, in original label space
        prog.progress(85, text="Evaluando el modelo…")
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

        prog.progress(95, text="Guardando el modelo…")
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

        prog.progress(100, text="¡Procesamiento completado!")
        status.update(label="Modelo entrenado correctamente", state="complete", expanded=False)

    st.toast("Modelo entrenado correctamente")
    return trainer

def render_prediction_form():
    st.subheader("Predicción de Nivel de Riesgo Operativo")

    from datetime import date
    import json

    # Formulario: solo predice al pulsar "Predecir" (no en cada cambio de campo).
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            device_brand = st.selectbox("Marca", get_brands())
            device_type = st.selectbox("Tipo de Equipo", get_device_types())
            acquisition_date = st.date_input(
                "Fecha de Adquisición", value=date(2021, 1, 1), max_value=date.today(),
                help="Fecha en que el equipo entró en operación. Determina la vida útil consumida.",
            )
            technical_incident_rate = st.number_input(
                "Tasa Incidencias Técnicas", min_value=0, max_value=100, value=2,
                help="Número de incidencias técnicas registradas para el equipo. "
                     "Valores altos elevan el riesgo.",
            )
        with col2:
            headquarters_location = st.selectbox("Ubicación", get_locations())
            hardware_integrity_status = st.selectbox(
                "Estado de Integridad de Hardware", get_hardware_states(),
                help="Estado físico del hardware, de 'Excelente' (mejor) a 'Crítico' (peor).",
            )
            no_corrective = st.checkbox(
                "Sin mantenimiento correctivo registrado", value=False,
                help="Marque si el equipo nunca tuvo mantenimiento correctivo. "
                     "Se ignorará la fecha y el modelo usará la vida útil consumida.",
            )
            last_reactive_maintenance_date = st.date_input(
                "Fecha Último Mto. Correctivo", value=date(2025, 1, 1), max_value=date.today(),
                help="Fecha del último mantenimiento correctivo. Marque la casilla superior si "
                     "el equipo no tiene mantenimientos correctivos registrados.",
            )
            no_preventive = st.checkbox(
                "Sin mantenimiento preventivo registrado", value=False,
                help="Marque si el equipo nunca tuvo mantenimiento preventivo. "
                     "Se ignorará la fecha y el modelo usará la vida útil consumida.",
            )
            last_preventive_maintenance_date = st.date_input(
                "Fecha Último Mto. Preventivo", value=date(2024, 1, 1), max_value=date.today(),
                help="Fecha del último mantenimiento preventivo. Marque la casilla superior si "
                     "el equipo no tiene mantenimientos preventivos registrados.",
            )

        predict_clicked = st.form_submit_button("Predecir Nivel de Riesgo", type="primary")

    if predict_clicked:
        if st.session_state["predictor"] is None:
            st.warning("El modelo aún no ha sido entrenado.", icon=":material/warning:")
            return

        # Dentro de un st.form la casilla no oculta el campo de fecha en vivo;
        # aplicamos su efecto al enviar: si se marca "sin mantenimiento", la
        # fecha se envía como NaT y el preprocesador usa la vida útil consumida.
        if no_corrective:
            last_reactive_maintenance_date = None
        if no_preventive:
            last_preventive_maintenance_date = None

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
            "last_preventive_maintenance_date": (
                pd.Timestamp(last_preventive_maintenance_date) if last_preventive_maintenance_date else pd.NaT
            ),
        }

        predictor = st.session_state["predictor"]
        risk_level = predictor.predict(input_dict)
        confidence = predictor.predict_proba(input_dict)

        # --- Resultado principal (titular destacado) --------------------------
        # El ícono Material (":material/...:") va como Markdown fuera del HTML
        # para que se renderice; el color del nivel se aplica con un <span>.
        color = RISK_COLORS.get(risk_level, NEUTRAL)
        icon = RISK_ICONS.get(risk_level, ":material/help:")
        top_prop = confidence.get(risk_level, max(confidence.values()) if confidence else 0.0)
        with st.container(border=True):
            st.caption("Nivel de Riesgo Operativo Predicho")
            st.markdown(
                f"{icon} <span style='font-size:1.9rem; font-weight:700; color:{color}; "
                f"vertical-align:middle;'>{risk_level}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"El modelo asigna el {top_prop * 100:.0f}% de los votos a este nivel.")

        # --- Confianza: barra horizontal ordenada por proporción de votos -----
        st.markdown("**Confianza de la predicción (proporción de votos por nivel)**")
        conf_df = pd.DataFrame(
            {"Nivel": list(confidence.keys()),
             "Confianza": [v * 100 for v in confidence.values()]}
        ).sort_values("Confianza", ascending=True)
        fig = px.bar(
            conf_df,
            x="Confianza",
            y="Nivel",
            orientation="h",
            color="Nivel",
            color_discrete_map=RISK_COLORS,
            category_orders={"Nivel": conf_df["Nivel"].tolist()},
            text=conf_df["Confianza"].map(lambda v: f"{v:.1f}%"),
        )
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="<b>%{y}</b><br>%{x:.1f}% de los votos<extra></extra>")
        fig.update_xaxes(range=[0, 100])
        style_fig(
            fig,
            xaxis_title="Confianza (%)",
            yaxis_title=None,
            height=300,
            show_legend=False,
        )
        st.plotly_chart(fig, width="stretch")

        # Derive day-difference features so they can be stored with the prediction
        engineered = Preprocessor().engineer_features(pd.DataFrame([input_dict]))

        alert_system = EarlyWarningSystem()
        device_id = f"PRED_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            alerts = alert_system.check_prediction(
                device_id, risk_level, {"device_type": device_type, "device_brand": device_brand}
            )
            if alerts:
                st.warning("Se han generado alertas para este equipo.", icon=":material/notifications:")
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

def render_import():
    st.subheader("Importar Data")

    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = import_csv(uploaded_file)
            st.success(f"CSV cargado: {len(df)} registros", icon=":material/check_circle:")

            if st.button("Guardar en Base de Datos", type="primary"):
                with st.status("Guardando en base de datos…", expanded=True) as status:
                    prog = st.progress(0, text="Guardando en base de datos…")
                    last_pct = {"value": -1}

                    def _save_progress(done, total):
                        pct = int(done / total * 100) if total else 100
                        # Throttle: only re-render the bar when the percent changes
                        if pct != last_pct["value"]:
                            last_pct["value"] = pct
                            prog.progress(
                                pct, text=f"Guardando en base de datos… ({done}/{total})"
                            )

                    count = save_to_database(df, progress_callback=_save_progress)
                    prog.progress(100, text="¡Guardado completado!")
                    status.update(
                        label=f"{count} registros guardados", state="complete", expanded=False
                    )

                # Los datos cambiaron: invalidar las cachés de lectura/exportación.
                fetch_all_devices.clear()
                st.toast(f"{count} registros guardados")
                st.success(f"{count} registros guardados exitosamente", icon=":material/check_circle:")
                st.rerun()
        except Exception as e:
            st.error(f"Error al importar: {e}", icon=":material/block:")

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

    tab_train, tab_fi, tab_metrics, tab_corr = st.tabs(
        ["Entrenar", "Feature Importance", "Métricas", "Correlación"]
    )

    with tab_train:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Entrenar Modelo Random Forest", type="primary"):
                train_model(df)
                st.rerun()

        if st.session_state["model_trained"]:
            st.success("**Modelo Entrenado** — Random Forest listo para predicciones", icon=":material/check_circle:")
        else:
            st.info("**Modelo No Entrenado** — Entrene el modelo para comenzar", icon=":material/info:")

    with tab_fi:
        if st.session_state["trainer"] is not None:
            viewer = FeatureImportanceViewer(st.session_state["trainer"])
            viewer.render()
        else:
            st.info("Entrena el modelo para ver la importancia de variables", icon=":material/info:")

    with tab_metrics:
        if st.session_state["model_trained"]:
            summary = st.session_state.get("test_metrics")

            if summary is not None:
                st.markdown("**Métricas globales (partición de prueba)**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{summary['accuracy'] * 100:.2f}%",
                          help="Porcentaje total de equipos clasificados correctamente.")
                c2.metric("Precision (macro)", f"{summary['precision_macro'] * 100:.2f}%",
                          help="De lo que el modelo marcó en cada clase, qué proporción acertó "
                               "(promedio sin ponderar por clase).")
                c3.metric("Recall (macro)", f"{summary['recall_macro'] * 100:.2f}%",
                          help="De los casos reales de cada clase, qué proporción detectó. "
                               "Es la métrica prioritaria para el alto riesgo.")
                c4.metric("F1-Score (macro)", f"{summary['f1_macro'] * 100:.2f}%",
                          help="Media armónica entre precisión y recall (promedio por clase).")

                c5, c6, c7 = st.columns(3)
                c5.metric("Precision (weighted)", f"{summary['precision_weighted'] * 100:.2f}%",
                          help="Precisión promedio ponderada por el nº de equipos de cada clase.")
                c6.metric("Recall (weighted)", f"{summary['recall_weighted'] * 100:.2f}%",
                          help="Recall promedio ponderado por el nº de equipos de cada clase.")
                c7.metric("F1-Score (weighted)", f"{summary['f1_weighted'] * 100:.2f}%",
                          help="F1 promedio ponderado por el nº de equipos de cada clase.")

                st.info(
                    "**Prioridad: el _recall_ en las clases de alto riesgo (Alto / Muy Alto).** "
                    "No detectar un equipo realmente de alto riesgo (falso negativo) es mucho más "
                    "costoso que una falsa alarma, por lo que conviene maximizar el recall en esas clases.",
                    icon=":material/target:",
                )

            # --- Recall por clase (énfasis en alto riesgo) --------------------
            report = st.session_state.get("test_report")
            if isinstance(report, dict):
                st.markdown("**Recall por nivel de riesgo (sensibilidad)**")
                cols = st.columns(len(get_risk_levels()))
                for col, level in zip(cols, get_risk_levels()):
                    cls = report.get(level)
                    if not isinstance(cls, dict):
                        continue
                    recall = cls.get("recall", 0.0)
                    support = int(cls.get("support", 0))
                    col.metric(
                        level,
                        f"{recall * 100:.0f}%",
                        delta=f"{support} casos",
                        delta_color="off",
                        help=f"De los {support} equipos realmente '{level}', el modelo detectó el "
                             f"{recall * 100:.0f}%.",
                    )

            best_params = st.session_state.get("best_params")
            best_cv_score = st.session_state.get("best_cv_score")
            if best_params is not None:
                st.markdown("---")
                st.markdown("**Mejores Hiperparámetros (RandomizedSearchCV)**")
                if best_cv_score is not None:
                    st.metric("F1-Score (macro, validación cruzada)", f"{best_cv_score * 100:.2f}%")
                with st.expander("Ver hiperparámetros seleccionados"):
                    st.json(best_params)

        else:
            st.info("No hay métricas disponibles", icon=":material/info:")

    with tab_corr:
        if df.empty:
            st.info("No hay datos para calcular la correlación", icon=":material/info:")
        else:
            render_correlation_matrix(df)

def render_monitoring_section(df):
    st.subheader("Monitoreo")

    tab_drift, tab_cm = st.tabs(["Data Drift", "Matriz de Confusión"])

    with tab_drift:
        if st.session_state["baseline_df"] is not None and not df.empty:
            detector = DataDriftDetector(st.session_state["baseline_df"])
            detector.render_ui(df)
        else:
            st.info("No hay datos de referencia para comparar", icon=":material/info:")

    with tab_cm:
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
                st.warning("No hay datos cargados para generar la matriz de confusión dinámica.", icon=":material/warning:")
        else:
            st.info("El modelo no está entrenado. Por favor, entrene el modelo en la sección 'Modelo' para ver la matriz de confusión dinámica.", icon=":material/info:")

        st.markdown("---")

        # Render historical predictions tracking if data exists
        if not history.empty:
            eval_obj = ModelEvaluator(None, st.session_state.get("preprocessor"))
            cm_monitor = ConfusionMatrixMonitor(eval_obj)
            cm_monitor.render_historical(history)
        else:
            st.info("No hay historial de predicciones guardado.", icon=":material/info:")

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
                sac.MenuItem("Importar Data", icon="arrow-left-right"),
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
            with st.container(border=True):
                st.markdown("### :material/upload_file: Aún no hay datos para mostrar")
                st.markdown(
                    "Para comenzar a ver indicadores y gráficos:\n\n"
                    "1. Ve a **Importar/Exportar** en el menú lateral.\n"
                    "2. Sube un archivo **CSV** de equipos.\n"
                    "3. Pulsa **Guardar en Base de Datos**.\n\n"
                    "Después podrás entrenar el modelo en **Modelo** y generar predicciones."
                )
        else:
            filter_mgr = FilterManager(df)
            filtered_df = filter_mgr.render_ui()
            render_dashboard(filtered_df)

    elif page == "Predicción":
        render_prediction_form()

    elif page == "Importar Data":
        render_import()

    elif page == "Modelo":
        if df.empty:
            st.warning("No hay datos para entrenar el modelo", icon=":material/warning:")
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
            st.error("Credenciales incorrectas. Verifique su usuario y contraseña.", icon=":material/block:")
    else:
        # Render main application contents first
        main_app()
        # Add a visual separator in the sidebar
        st.sidebar.divider()
        # Render logout button below the menu navigation
        authenticator.logout(location='sidebar')

if __name__ == "__main__":
    main()
