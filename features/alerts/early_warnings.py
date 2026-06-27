import streamlit as st
from datetime import datetime
from features.database import save_alert, get_pending_alerts, resolve_alert

HIGH_VALUE_THRESHOLD = 300

class EarlyWarningSystem:
    def __init__(self):
        self.alerts = []

    def check_prediction(self, equipo_id, estado_predicted, nivel_riesgo, equipo_data=None):
        alerts_created = []

        if estado_predicted == "Crítico":
            is_high_value = False
            if equipo_data and equipo_data.get("costo_mto", 0) > HIGH_VALUE_THRESHOLD:
                is_high_value = True

            alert_data = {
                "equipo_id": equipo_id,
                "tipo_alerta": "CRITICO_PREDICHO",
                "prioridad": "ALTA" if is_high_value else "MEDIA",
                "mensaje": f"Equipo {equipo_id} predicho en estado CRÍTICO. Riesgo: {nivel_riesgo}. {'EQUIPO DE ALTO VALOR - ATENCIÓN PRIORITARIA' if is_high_value else ''}",
                "estado": "pendiente"
            }
            alert = save_alert(alert_data)
            alerts_created.append(alert)

        if nivel_riesgo == "Crítico":
            alert_data = {
                "equipo_id": equipo_id,
                "tipo_alerta": "RIESGO_CRITICO",
                "prioridad": "ALTA",
                "mensaje": f"Equipo {equipo_id} con Nivel de Riesgo Operativo CRÍTICO",
                "estado": "pendiente"
            }
            alert = save_alert(alert_data)
            alerts_created.append(alert)

        return alerts_created

    def render_alerts_panel(self):
        st.subheader("Panel de Alertas Tempranas")

        pending = get_pending_alerts()

        if not pending:
            st.success("No hay alertas pendientes")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Alertas Pendientes", len(pending))
        high_priority = sum(1 for a in pending if a.prioridad == "ALTA")
        with col2:
            st.metric("Alertas Alta Prioridad", high_priority)

        for alert in pending:
            with st.expander(f"⚠️ {alert.tipo_alerta} - {alert.equipo_id} ({alert.prioridad})"):
                st.write(f"**Mensaje:** {alert.mensaje}")
                st.write(f"**Creada:** {alert.created_at}")
                st.write(f"**Estado:** {alert.estado}")

                if st.button(f"Resolver", key=str(alert.id)):
                    resolve_alert(str(alert.id))
                    st.rerun()
