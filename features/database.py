import streamlit as st
from sqlalchemy import create_engine, Column, String, Integer, Float, Date, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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

class Device(Base):
    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(100), unique=True, nullable=False)
    device_brand = Column(String(50))
    acquisition_date = Column(Date)
    technical_incident_rate = Column(Integer)
    last_reactive_maintenance_date = Column(Date, nullable=True)
    last_preventive_maintenance_date = Column(Date)
    headquarters_location = Column(String(50))
    hardware_integrity_status = Column(String(50))
    device_type = Column(String(50))
    operational_risk_level = Column(String(50))
    registered_at = Column(DateTime, default=datetime.utcnow)

class HistoricalData(Base):
    __tablename__ = "historical_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id"))
    data_json = Column(Text)
    imported_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String(255))

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(100), nullable=False)
    device_brand = Column(String(50))
    device_type = Column(String(50))
    acquisition_date = Column(Date)
    technical_incident_rate = Column(Integer)
    days_since_reactive_maintenance = Column(Integer)
    days_since_preventive_maintenance = Column(Integer)
    headquarters_location = Column(String(50))
    hardware_integrity_status = Column(String(50))
    predicted_risk_level = Column(String(50))
    confidence_json = Column(Text)
    prediction_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(100), nullable=False)
    alert_type = Column(String(50), nullable=False)
    priority_level = Column(String(20), nullable=False)
    message_text = Column(Text)
    status_alert = Column(String(20), default="pendiente")
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
        password_hash = hashlib.sha256("admin".encode()).hexdigest()
        admin = User(
            username="admin",
            email="admin@admin.com",
            first_name="Ariel",
            last_name="Anchapaxi",
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

def save_device(device_data: dict) -> Device:
    session = get_session()
    device = Device(**device_data)
    session.add(device)
    session.commit()
    session.close()
    return device

def get_all_devices() -> pd.DataFrame:
    """Return all devices as a DataFrame using the dataset column names.

    The database column names mirror the dataset, so a DB-loaded DataFrame has
    the same shape as a freshly imported CSV.
    """
    session = get_session()
    devices = session.query(Device).all()
    session.close()
    return pd.DataFrame([{
        "id": str(device.id),
        "device_id": device.device_id,
        "device_brand": device.device_brand,
        "device_type": device.device_type,
        "acquisition_date": device.acquisition_date,
        "technical_incident_rate": device.technical_incident_rate,
        "last_reactive_maintenance_date": device.last_reactive_maintenance_date,
        "last_preventive_maintenance_date": device.last_preventive_maintenance_date,
        "headquarters_location": device.headquarters_location,
        "hardware_integrity_status": device.hardware_integrity_status,
        "operational_risk_level": device.operational_risk_level,
        "registered_at": device.registered_at,
    } for device in devices])

def clear_all_data() -> dict:
    """Delete all imported devices, predictions and alerts from the database.

    Historical data (which references ``devices`` via a foreign key) is deleted
    first to avoid FK constraint violations. Users, trained models and drift
    logs are preserved. Returns a dict with the number of rows deleted per table.
    """
    session = get_session()
    try:
        counts = {
            "historical_data": session.query(HistoricalData).delete(),
            "devices": session.query(Device).delete(),
            "predictions": session.query(Prediction).delete(),
            "alerts": session.query(Alert).delete(),
        }
        session.commit()
        return counts
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

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
    alerts = session.query(Alert).filter_by(status_alert="pendiente").order_by(Alert.created_at.desc()).all()
    session.close()
    return alerts

def resolve_alert(alert_id: str):
    session = get_session()
    alert = session.query(Alert).filter_by(id=uuid.UUID(alert_id)).first()
    if alert:
        alert.status_alert = "resuelto"
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
        "device_id": p.device_id,
        "hardware_integrity_status": p.hardware_integrity_status,
        "predicted_risk_level": p.predicted_risk_level,
        "confidence_json": p.confidence_json,
        "prediction_at": p.prediction_at
    } for p in predictions])
