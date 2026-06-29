-- Schema SQL para PostgreSQL
-- Sistema de Predicción de Fallos en Equipos Electrónicos

-- Tabla de usuarios para autenticación
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Tabla principal de equipos
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(100) UNIQUE NOT NULL,
    device_brand VARCHAR(50),
    device_type VARCHAR(50),
    acquisition_date DATE,
    technical_incident_rate INTEGER,
    last_reactive_maintenance_date DATE,
    last_preventive_maintenance_date DATE,
    headquarters_location VARCHAR(50),
    hardware_integrity_status VARCHAR(50),
    operational_risk_level VARCHAR(50),
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historial de datos importados
CREATE TABLE historical_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id UUID REFERENCES devices(id),
    data_json TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filename VARCHAR(255)
);

-- Predicciones del modelo
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(100) NOT NULL,
    device_brand VARCHAR(50),
    acquisition_date DATE,
    technical_incident_rate INTEGER,
    days_since_reactive_maintenance INTEGER,
    days_since_preventive_maintenance INTEGER,
    headquarters_location VARCHAR(50),
    hardware_integrity_status VARCHAR(50),
    device_type VARCHAR(50),
    predicted_risk_level VARCHAR(50),
    confidence_json TEXT,
    prediction_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alertas del sistema
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    priority_level VARCHAR(20) NOT NULL,
    message_text TEXT,
    status_alert VARCHAR(20) DEFAULT 'pendiente',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Modelos entrenados
CREATE TABLE trained_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_path VARCHAR(255),
    features TEXT,
    metrics_json TEXT,
    feature_importance_json TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Logs de data drift
CREATE TABLE data_drift_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variable_name VARCHAR(100) NOT NULL,
    ks_statistic FLOAT,
    p_value FLOAT,
    drift_detected BOOLEAN DEFAULT FALSE,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para optimizar consultas
CREATE INDEX idx_devices_location ON devices(headquarters_location);
CREATE INDEX idx_devices_status ON devices(hardware_integrity_status);
CREATE INDEX idx_devices_risk ON devices(operational_risk_level);
CREATE INDEX idx_predictions_date ON predictions(prediction_at);
CREATE INDEX idx_alerts_status ON alerts(status_alert);
CREATE INDEX idx_alerts_priority ON alerts(priority_level);

-- Usuario admin por defecto (password: admin123)
-- El hash se genera automáticamente con bcrypt en la aplicación
