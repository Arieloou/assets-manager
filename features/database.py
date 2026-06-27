import streamlit as st
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import pandas as pd
import hashlib

Base = declarative_base()

@st.cache_resource
def get_engine():
    db_config = st.secrets["connections"]["postgresql"]
    connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    return create_engine(connection_string, pool_pre_ping=True)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

@st.cache_resource
def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Equipo(Base):
    __tablename__ = "equipos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id_equipo = Column(String(100), unique=True, nullable=False)
    vida_util_consumida = Column(Float)
    tasa_incidencias_tecnicas = Column(Integer)
    tiempo_inactividad_acumulado = Column(Float)
    costo_mto_reactivo_acumulado = Column(Float)
    ubicacion_activo = Column(String(50))
    estado_integridad_hardware = Column(String(50))
    tipo_equipo = Column(String(50))
    nivel_riesgo_operativo = Column(String(50))
    timestamp_registro = Column(DateTime, default=datetime.utcnow)

class HistoricalData(Base):
    __tablename__ = "historical_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipo_id = Column(UUID(as_uuid=True), ForeignKey("equipos.id"))
    data_json = Column(Text)
    imported_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String(255))

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipo_id = Column(String(100), nullable=False)
    vida_util_consumida = Column(Float)
    tasa_incidencias_tecnicas = Column(Integer)
    tiempo_inactividad_acumulado = Column(Float)
    costo_mto_reactivo_acumulado = Column(Float)
    ubicacion_activo = Column(String(50))
    tipo_equipo = Column(String(50))
    estado_integridad_predicted = Column(String(50))
    nivel_riesgo_predicted = Column(String(50))
    prediction_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipo_id = Column(String(100), nullable=False)
    tipo_alerta = Column(String(50), nullable=False)
    prioridad = Column(String(20), nullable=False)
    mensaje = Column(Text)
    estado = Column(String(20), default="pendiente")
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_path = Column(String(255))
    features = Column(Text)
    metrics_json = Column(Text)
    feature_importance_json = Column(Text)
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)

class DataDriftLog(Base):
    __tablename__ = "data_drift_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    variable_name = Column(String(100), nullable=False)
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean, default=False)
    logged_at = Column(DateTime, default=datetime.utcnow)

def create_default_user():
    session = get_session()
    existing = session.query(User).filter_by(username="admin").first()
    if not existing:
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        admin = User(
            username="admin",
            email="admin@equipos.com",
            first_name="Admin",
            last_name="User",
            password_hash=password_hash,
            role="admin"
        )
        session.add(admin)
        session.commit()
    session.close()

def verify_user(username: str, password: str) -> bool:
    session = get_session()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    user = session.query(User).filter_by(username=username, password_hash=password_hash, is_active=True).first()
    session.close()
    return user is not None

def save_equipo(equipo_data: dict) -> Equipo:
    session = get_session()
    equipo = Equipo(**equipo_data)
    session.add(equipo)
    session.commit()
    session.close()
    return equipo

def get_all_equipos() -> pd.DataFrame:
    session = get_session()
    equipos = session.query(Equipo).all()
    session.close()
    return pd.DataFrame([{
        "id": str(e.id),
        "id_equipo": e.id_equipo,
        "vida_util_consumida": e.vida_util_consumida,
        "tasa_incidencias_tecnicas": e.tasa_incidencias_tecnicas,
        "tiempo_inactividad_acumulado": e.tiempo_inactividad_acumulado,
        "costo_mto_reactivo_acumulado": e.costo_mto_reactivo_acumulado,
        "ubicacion_activo": e.ubicacion_activo,
        "estado_integridad_hardware": e.estado_integridad_hardware,
        "tipo_equipo": e.tipo_equipo,
        "nivel_riesgo_operativo": e.nivel_riesgo_operativo,
        "timestamp_registro": e.timestamp_registro
    } for e in equipos])

def save_prediction(prediction_data: dict) -> Prediction:
    session = get_session()
    prediction = Prediction(**prediction_data)
    session.add(prediction)
    session.commit()
    session.close()
    return prediction

def save_alert(alert_data: dict) -> Alert:
    session = get_session()
    alert = Alert(**alert_data)
    session.add(alert)
    session.commit()
    session.close()
    return alert

def get_pending_alerts() -> list:
    session = get_session()
    alerts = session.query(Alert).filter_by(estado="pendiente").order_by(Alert.created_at.desc()).all()
    session.close()
    return alerts

def resolve_alert(alert_id: str):
    session = get_session()
    alert = session.query(Alert).filter_by(id=uuid.UUID(alert_id)).first()
    if alert:
        alert.estado = "resuelto"
        alert.resolved_at = datetime.utcnow()
        session.commit()
    session.close()

def save_trained_model(model_data: dict) -> TrainedModel:
    session = get_session()
    session.query(TrainedModel).update({"is_active": False})
    model = TrainedModel(**model_data, is_active=True)
    session.add(model)
    session.commit()
    session.close()
    return model

def get_active_model() -> TrainedModel:
    session = get_session()
    model = session.query(TrainedModel).filter_by(is_active=True).first()
    session.close()
    return model

def log_data_drift(variable: str, ks_stat: float, p_value: float, drift: bool):
    session = get_session()
    log = DataDriftLog(
        variable_name=variable,
        ks_statistic=ks_stat,
        p_value=p_value,
        drift_detected=drift
    )
    session.add(log)
    session.commit()
    session.close()

def get_predictions_history(limit: int = 100) -> pd.DataFrame:
    session = get_session()
    predictions = session.query(Prediction).order_by(Prediction.prediction_at.desc()).limit(limit).all()
    session.close()
    return pd.DataFrame([{
        "id": str(p.id),
        "equipo_id": p.equipo_id,
        "estado_integridad_predicted": p.estado_integridad_predicted,
        "nivel_riesgo_predicted": p.nivel_riesgo_predicted,
        "prediction_at": p.prediction_at
    } for p in predictions])
