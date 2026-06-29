import streamlit as st

from features.database import save_alert, get_pending_alerts, resolve_alert

# Equipment types considered high-value (prioritized attention).
HIGH_VALUE_TYPES = ["Laptop", "Proyector"]
HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]


class EarlyWarningSystem:
    """Registers prioritized alerts based on operational-risk predictions."""

    def __init__(self):
        self.alerts = []

    def check_prediction(self, device_id, risk_level, device_data=None):
        """Create an alert when the predicted risk is high.

        Args:
            device_id: Device identifier.
            risk_level: Predicted ``operational_risk_level`` label.
            device_data: Optional dict with raw features (e.g. ``device_type``,
                ``device_brand``) used to flag high-value assets.

        Returns:
            List of created alert objects.
        """
        alerts_created = []

        if risk_level in HIGH_RISK_LEVELS:
            is_high_value = bool(
                device_data and device_data.get("device_type") in HIGH_VALUE_TYPES
            )
            priority = "ALTA" if (risk_level == "Muy Alto" or is_high_value) else "MEDIA"
            message = (
                f"Equipo {device_id} predicho con Nivel de Riesgo Operativo {risk_level}."
            )
            if is_high_value:
                message += " EQUIPO DE ALTO VALOR - ATENCIÓN PRIORITARIA"

            alert_data = {
                "device_id": device_id,
                "alert_type": "RIESGO_ALTO",
                "priority_level": priority,
                "message_text": message,
                "status_alert": "pendiente",
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
        high_priority = sum(1 for a in pending if a.priority_level == "ALTA")
        with col2:
            st.metric("Alertas Alta Prioridad", high_priority)

        for alert in pending:
            with st.expander(f"⚠️ {alert.alert_type} - {alert.device_id} ({alert.priority_level})"):
                st.write(f"**Mensaje:** {alert.message_text}")
                st.write(f"**Creada:** {alert.created_at}")
                st.write(f"**Estado:** {alert.status_alert}")

                if st.button("Resolver", key=str(alert.id)):
                    resolve_alert(str(alert.id))
                    st.rerun()
