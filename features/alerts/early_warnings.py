import streamlit as st

from features.database import save_alert, get_pending_alerts, resolve_alert

# Equipment types considered high-value (prioritized attention).
HIGH_VALUE_TYPES = ["Laptop", "Proyector"]
HIGH_RISK_LEVELS = ["Alto", "Muy Alto"]

# Estilo de severidad por prioridad (coherente con la paleta de riesgo).
# Iconos de Material Symbols (sintaxis ":material/...:").
PRIORITY_STYLE = {
    "ALTA": {"icon": ":material/crisis_alert:", "color": "#DC2626"},
    "MEDIA": {"icon": ":material/warning:", "color": "#EA580C"},
    "BAJA": {"icon": ":material/info:", "color": "#F59E0B"},
}


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
            st.success("No hay alertas pendientes. Todos los equipos bajo control.", icon=":material/check_circle:")
            return

        high_priority = sum(1 for a in pending if a.priority_level == "ALTA")
        col1, col2 = st.columns(2)
        col1.metric(
            "Alertas pendientes", len(pending),
            help="Alertas generadas por predicciones de alto riesgo aún sin resolver.",
        )
        col2.metric(
            "Alta prioridad", high_priority,
            delta=None if high_priority == 0 else f"{high_priority} urgente(s)",
            delta_color="inverse",
            help="Alertas de equipos en Riesgo Muy Alto o equipos de alto valor.",
        )

        st.markdown("---")
        # Mostrar primero las de mayor prioridad.
        order = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}
        for alert in sorted(pending, key=lambda a: order.get(a.priority_level, 9)):
            style = PRIORITY_STYLE.get(alert.priority_level, {"icon": ":material/help:", "color": "#64748B"})
            with st.container(border=True):
                head, action = st.columns([5, 1])
                with head:
                    st.markdown(
                        f"{style['icon']} **{alert.alert_type}** · `{alert.device_id}` "
                        f"<span style='color:{style['color']};font-weight:600'>"
                        f"({alert.priority_level})</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(alert.message_text)
                    st.caption(f"Creada: {alert.created_at} · Estado: {alert.status_alert}")
                with action:
                    if st.button("Resolver", key=str(alert.id), type="primary"):
                        resolve_alert(str(alert.id))
                        st.rerun()
