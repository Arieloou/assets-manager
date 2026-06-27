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
CREATE TABLE equipos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_equipo VARCHAR(100) UNIQUE NOT NULL,
    vida_util_consumida FLOAT,
    tasa_incidencias_tecnicas INTEGER,
    tiempo_inactividad_acumulado FLOAT,
    costo_mto_reactivo_acumulado FLOAT,
    ubicacion_activo VARCHAR(50),
    estado_integridad_hardware VARCHAR(50),
    tipo_equipo VARCHAR(50),
    nivel_riesgo_operativo VARCHAR(50),
    timestamp_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historial de datos importados
CREATE TABLE historical_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipo_id UUID REFERENCES equipos(id),
    data_json TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filename VARCHAR(255)
);

-- Predicciones del modelo
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipo_id VARCHAR(100) NOT NULL,
    vida_util_consumida FLOAT,
    tasa_incidencias_tecnicas INTEGER,
    tiempo_inactividad_acumulado FLOAT,
    costo_mto_reactivo_acumulado FLOAT,
    ubicacion_activo VARCHAR(50),
    tipo_equipo VARCHAR(50),
    estado_integridad_predicted VARCHAR(50),
    nivel_riesgo_predicted VARCHAR(50),
    prediction_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alertas del sistema
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipo_id VARCHAR(100) NOT NULL,
    tipo_alerta VARCHAR(50) NOT NULL,
    prioridad VARCHAR(20) NOT NULL,
    mensaje TEXT,
    estado VARCHAR(20) DEFAULT 'pendiente',
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
CREATE INDEX idx_equipos_ubicacion ON equipos(ubicacion_activo);
CREATE INDEX idx_equipos_estado ON equipos(estado_integridad_hardware);
CREATE INDEX idx_predictions_fecha ON predictions(prediction_at);
CREATE INDEX idx_alerts_estado ON alerts(estado);
CREATE INDEX idx_alerts_prioridad ON alerts(prioridad);

-- Usuario admin por defecto (password: admin123)
-- El hash se genera automáticamente con bcrypt en la aplicación
