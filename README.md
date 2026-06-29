# Sistema de Predicción de Riesgo Operativo en Equipos Electrónicos

Sistema de monitoreo y predicción del **nivel de riesgo operativo** de equipos electrónicos
utilizando Random Forest.

## Modelo Random Forest

- **Target (salida única):** `operational_risk_level` — 5 clases ordinales
  (`Muy Bajo < Bajo < Medio < Alto < Muy Alto`), codificado con `OrdinalEncoder`.
- **Features de entrada:**
  1. `device_brand` (`HP`, `Dell`, `Epson`, `NEC`, `Canon`) — `OneHotEncoder`.
  2. `device_type` (`Computadora de Escritorio`, `Laptop`, `Proyector`, `Impresora`) — `OneHotEncoder`.
  3. `hardware_integrity_status` — `OrdinalEncoder`
     (`Excelente > Bueno > Desgastado > Malo > Crítico`).
  4. `headquarters_location` (`Park`, `Granados`, `Colon`) — `OneHotEncoder`.
  5. `useful_life_consumed_days` — días desde `acquisition_date` hasta hoy — `StandardScaler`.
  6. `technical_incident_rate` — cantidad de incidencias reportadas — `StandardScaler`.
  7. `days_since_last_corrective_maintenance` — días desde el último mantenimiento correctivo
     (si nunca hubo, equivale a `useful_life_consumed_days`) — `StandardScaler`.
  8. `days_since_last_preventive_maintenance` — días desde el último mantenimiento preventivo — `StandardScaler`.
- **Escalado:** `StandardScaler` aplicado a las **4 features cuantitativas**
  (`useful_life_consumed_days`, `technical_incident_rate`,
  `days_since_last_corrective_maintenance`, `days_since_last_preventive_maintenance`) dentro del
  `Preprocessor`; el modelo es un `RandomForestClassifier` puro (`random_classifier_model`).
- **Soft Output:** además de la clase predicha, el modelo devuelve la proporción de votos por
  nivel de riesgo (`predict_proba`), interpretada como la confianza de la predicción.
- **Evaluación:** `confusion_matrix` (5×5, filas = real / columnas = predicho) y
  `classification_report` (precision / recall / F1) mostrados en tarjetas.

## Requisitos

- Python 3.9+
- PostgreSQL 13+
- Dependencies listed in `requirements.txt`

## Instalación

1. **Clonar el repositorio**
```bash
git clone <repo-url>
cd project
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar base de datos PostgreSQL**

Crear la base de datos:
```sql
CREATE DATABASE equipos_db;
```

Editar `.streamlit/secrets.toml` con las credenciales:
```toml
[connections.postgresql]
host = "localhost"
port = "5432"
database = "equipos_db"
username = "postgres"
password = "tu_password"
```

5. **Inicializar la base de datos**

Ejecutar `schema.sql` en PostgreSQL o ejecutar la aplicación (crea las tablas automáticamente).

## Uso

1. **Iniciar la aplicación**
```bash
streamlit run streamlit_app.py
```

2. **Credenciales por defecto**
- Usuario: `admin`
- Contraseña: `admin123`

3. **Flujo de trabajo**
   1. Importar CSV con datos históricos de equipos
   2. Entrenar el modelo Random Forest
   3. Realizar predicciones
   4. Monitorear data drift y alertas

## Estructura del Proyecto

```
project/
├── features/              # Módulos funcionales
│   ├── auth/              # Autenticación
│   ├── data/              # Carga y exportación de datos
│   ├── model/             # Entrenamiento y predicción
│   ├── dashboard/         # Visualizaciones y KPIs
│   ├── monitoring/        # Data drift y matriz de confusión
│   └── alerts/            # Sistema de alertas tempranas
├── models/
│   └── trained_models/    # Modelos guardados
├── .streamlit/
│   └── secrets.toml       # Configuración de BD
├── streamlit_app.py       # Aplicación principal
├── config.yaml            # Configuración general
├── requirements.txt       # Dependencias Python
└── schema.sql             # Schema de BD
```

## Features Implementadas

- Login/Logout con autenticación
- Dashboard de análisis de activos
- Gráfico de dona por ubicación
- Sistema de filtros dinámicos
- Importación/Exportación CSV
- Predicción de nivel de riesgo operativo con Random Forest
- Soft Output (proporción de votos / confianza por nivel de riesgo)
- Cross-validation (k≥5)
- Feature importance
- Monitoreo de Data Drift (sobre features derivadas)
- Matriz de confusión dinámica
- Sistema de alertas tempranas (riesgo Alto / Muy Alto)
- Arquitectura basada en características
